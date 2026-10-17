"""Walk-forward validation, and the checks that tell an edge from a curve fit.

The procedure:

    1. Draw a fixed set of candidate configurations from the search space.
    2. Slide a window through the history. On each IN-SAMPLE window pick the candidate with
       the best score; then measure that candidate on the OUT-OF-SAMPLE window that follows,
       which played no part in choosing it.
    3. Stitch the out-of-sample windows together. That stitched record, not the in-sample
       one, is the estimate of what the strategy does on data it has never seen.

and what is reported alongside it:

    walk-forward efficiency   out-of-sample edge as a share of in-sample edge. Near 1 means the
                              optimiser found something real; near 0 means it fitted noise.
    rank correlation          do candidates that rank well in-sample also rank well out-of-sample?
    probability of overfit    how often the in-sample winner lands in the bottom half out-of-sample.
    trials hurdle             the t-statistic the best of N random candidates reaches by luck alone.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import FeatureStore
from .config import GOMER_COSTS, CostModel, StrategyConfig
from .metrics import summarize

SEARCH_SPACE: dict[str, list] = {
    # which side of a stretch to take, and how strictly the regime is read
    "playbook": ["reversion", "continuation", "adaptive"],
    "regime_mode": ["either", "both", "off"],
    "regime_tf_sec": [60, 300],
    "adx_max": [20.0, 25.0, 30.0],
    "bbw_max_pctile": [0.5, 0.8],
    # entry
    "entry_z": [1.5, 2.0, 2.5, 3.0],
    "rsi_low": [5.0, 10.0, 20.0],
    "use_absorption": [True, False],
    "absorb_rel_volume": [1.0, 1.3],
    "absorb_wick_frac": [0.2, 0.3],
    "entry_style": ["passive", "marketable"],
    # reversion exits
    "target_frac": [0.15, 0.25, 0.5, 1.0],
    "stop_z": [3.5, 5.0, 7.0],
    "stop_atr_mult": [1.5, 3.0],
    "max_hold_bars": [5, 10, 30],
    # continuation exits
    "cont_target_atr": [0.35, 0.5, 0.75, 1.0, 1.5],
    "cont_stop_atr": [1.5, 2.5, 3.5],
    "cont_max_hold_bars": [5, 10, 20],
}


SEARCH_SPACE_V2: dict[str, list] = {
    # Second round, after the first live trades. The playbook question is settled (see the
    # report), so this searches the continuation playbook's own settings and the filters that
    # came out of reviewing losing trades.
    "adx_max": [25.0, 30.0, 35.0],
    "entry_z": [1.5, 2.0, 2.5],
    "rsi_low": [5.0, 10.0],
    "cont_min_di_gap": [0.0, 20.0, 30.0, 40.0],
    "cont_max_wick": [1.0, 0.4, 0.25],
    "max_rel_volume": [0.0, 2.0, 1.5],
    "max_atr_bps": [0.0, 48.0, 41.0],
    "cont_target_atr": [0.5, 0.75, 1.0],
    "cont_stop_atr": [1.5, 2.0, 2.5],
    "cont_max_hold_bars": [5, 10],
    "session_warmup_bars": [0, 15, 30],
    "max_same_side": [0, 2, 3],
}


def sample_configs(base: StrategyConfig, n: int, seed: int = 1, space: dict[str, list] | None = None) -> list[StrategyConfig]:
    """`n` distinct candidates drawn uniformly from the grid. `base` itself is always the first."""
    space = space or SEARCH_SPACE
    rng = random.Random(seed)
    total = math.prod(len(v) for v in space.values())
    seen = {tuple(getattr(base, k) for k in space)}
    configs = [base]
    while len(configs) < min(n, total):
        choice = {k: rng.choice(v) for k, v in space.items()}
        key = tuple(choice.values())
        if key in seen:
            continue
        seen.add(key)
        if "rsi_low" in choice:
            choice["rsi_high"] = 100.0 - choice["rsi_low"]
        configs.append(base.with_(**choice))
    return configs


def score(stats: dict, min_trades: int, target_win_rate: float) -> float:
    """What the optimiser maximises.

    The t-statistic of the average trade (in R) rewards an edge that is both large and
    consistent and penalises one resting on a handful of trades. Candidates that also reach
    the target win rate with a significant edge are ranked ahead of all that do not, so the
    win-rate goal is pursued only among configurations that make money.
    """
    if stats["trades"] < min_trades:
        return -math.inf
    t = stats["t_stat"]
    return t + 1000.0 if stats["win_rate"] >= target_win_rate and t >= 2.0 else t


@dataclass
class Fold:
    number: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_end: pd.Timestamp
    chosen: int
    is_stats: dict
    oos_stats: dict
    rank_correlation: float
    chosen_oos_percentile: float


@dataclass
class WalkForwardResult:
    folds: list[Fold]
    configs: list[StrategyConfig]
    oos_trades: pd.DataFrame
    oos_stats: dict
    is_stats_mean: dict
    efficiency: float
    rank_correlation: float
    probability_of_overfit: float
    trials_hurdle: float
    all_trades: list[pd.DataFrame] = field(repr=False, default_factory=list)

    def parameter_table(self, keys: list[str] | None = None) -> pd.DataFrame:
        keys = keys or list(SEARCH_SPACE)
        return pd.DataFrame([{k: getattr(self.configs[f.chosen], k) for k in keys} for f in self.folds], index=[f.number for f in self.folds])


def _window(trades: pd.DataFrame, start, end) -> pd.DataFrame:
    return trades[(trades["entry_time"] >= start) & (trades["entry_time"] < end)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5:
        return float("nan")
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


_WORKER_STORE: FeatureStore | None = None


def _init_worker(directory: str, symbols: list[str] | None) -> None:
    global _WORKER_STORE
    from .data import load_dir

    _WORKER_STORE = FeatureStore(load_dir(directory, symbols))


def _run_candidate(job: tuple[StrategyConfig, CostModel, float]) -> pd.DataFrame:
    cfg, costs, starting_equity = job
    return _WORKER_STORE.run(cfg, costs, starting_equity=starting_equity, fixed_risk=True, compound=False).trades


def run_candidates(
    directory: str,
    configs: list[StrategyConfig],
    costs: CostModel = GOMER_COSTS,
    starting_equity: float = 100_000.0,
    symbols: list[str] | None = None,
    workers: int | None = None,
) -> list[pd.DataFrame]:
    """Runs every candidate once over the whole history, in parallel, and returns its trades.

    Sizing is fixed and non-compounding here, so a trade's size and R-multiple do not depend
    on what happened in any other window. A window's result is then simply the trades opened
    inside it.
    """
    import multiprocessing as mp
    import os

    workers = workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    # Sorting by indicator settings lets each worker reuse its cached indicator frames.
    order = sorted(range(len(configs)), key=lambda i: configs[i].indicator_key())
    jobs = [(configs[i], costs, starting_equity) for i in order]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(directory, symbols)) as pool:
        results = pool.map(_run_candidate, jobs, chunksize=max(1, len(jobs) // (workers * 4)))
    trades: list[pd.DataFrame] = [None] * len(configs)  # type: ignore[list-item]
    for i, frame in zip(order, results):
        trades[i] = frame
    return trades


def walk_forward(
    all_trades: list[pd.DataFrame],
    configs: list[StrategyConfig],
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
    is_days: int = 20,
    oos_days: int = 5,
    min_trades: int = 40,
    target_win_rate: float = 0.80,
    starting_equity: float = 100_000.0,
) -> WalkForwardResult:
    """`all_trades[i]` is candidate i's trades over the whole history (see run_candidates)."""
    first, last = first_day.floor("D"), last_day.floor("D") + pd.Timedelta(days=1)
    folds: list[Fold] = []
    oos_parts: list[pd.DataFrame] = []
    start = first
    number = 1
    while start + pd.Timedelta(days=is_days + oos_days) <= last:
        is_end = start + pd.Timedelta(days=is_days)
        oos_end = is_end + pd.Timedelta(days=oos_days)
        is_stats = [summarize(_window(t, start, is_end), None, starting_equity) for t in all_trades]
        oos_stats = [summarize(_window(t, is_end, oos_end), None, starting_equity) for t in all_trades]
        scores = np.array([score(s, min_trades, target_win_rate) for s in is_stats])
        chosen = int(np.argmax(scores))

        # Among candidates with enough trades on both sides: does in-sample rank predict out-of-sample rank?
        usable = [i for i in range(len(configs)) if np.isfinite(scores[i]) and oos_stats[i]["trades"] >= max(5, min_trades // 4)]
        is_edge = np.array([is_stats[i]["t_stat"] for i in usable])
        oos_edge = np.array([oos_stats[i].get("expectancy_r", 0.0) for i in usable])
        correlation = _spearman(is_edge, oos_edge)
        percentile = float((oos_edge <= oos_stats[chosen].get("expectancy_r", 0.0)).mean()) if chosen in usable and len(usable) > 1 else float("nan")

        folds.append(Fold(number, start, is_end, oos_end, chosen, is_stats[chosen], oos_stats[chosen], correlation, percentile))
        oos_parts.append(_window(all_trades[chosen], is_end, oos_end))
        start += pd.Timedelta(days=oos_days)
        number += 1

    oos_trades = pd.concat(oos_parts, ignore_index=True) if oos_parts else all_trades[0].iloc[0:0]
    oos_trades = oos_trades.sort_values("exit_time").reset_index(drop=True)
    equity = None
    if len(oos_trades):
        equity = (starting_equity + oos_trades.set_index("exit_time")["pnl"].cumsum()).groupby(level=0).last()
    oos_summary = summarize(oos_trades, equity, starting_equity)

    is_r = float(np.mean([f.is_stats.get("expectancy_r", 0.0) for f in folds])) if folds else 0.0
    is_mean = {
        "win_rate": float(np.mean([f.is_stats["win_rate"] for f in folds])) if folds else float("nan"),
        "expectancy_r": is_r,
        "expectancy_bps": float(np.mean([f.is_stats.get("expectancy_bps", 0.0) for f in folds])) if folds else 0.0,
    }
    percentiles = [f.chosen_oos_percentile for f in folds if f.chosen_oos_percentile == f.chosen_oos_percentile]
    correlations = [f.rank_correlation for f in folds if f.rank_correlation == f.rank_correlation]
    return WalkForwardResult(
        folds=folds,
        configs=configs,
        oos_trades=oos_trades,
        oos_stats=oos_summary,
        is_stats_mean=is_mean,
        efficiency=oos_summary.get("expectancy_r", 0.0) / is_r if is_r > 0 else float("nan"),
        rank_correlation=float(np.mean(correlations)) if correlations else float("nan"),
        probability_of_overfit=float(np.mean([p < 0.5 for p in percentiles])) if percentiles else float("nan"),
        trials_hurdle=math.sqrt(2.0 * math.log(max(2, len(configs)))),
        all_trades=all_trades,
    )


def select_final(result: WalkForwardResult, min_trades: int, target_win_rate: float, starting_equity: float = 100_000.0) -> tuple[int, dict]:
    """The configuration to deploy: the best score over the whole history, by the same rule the folds used."""
    stats = [summarize(t, None, starting_equity) for t in result.all_trades]
    scores = [score(s, min_trades, target_win_rate) for s in stats]
    chosen = int(np.argmax(scores))
    return chosen, stats[chosen]


def shuffle_signals(features: dict[str, pd.DataFrame], seed: int = 0) -> dict[str, pd.DataFrame]:
    """The same number of long and short entries per symbol, at random bars.

    Running the strategy's exits on these is the benchmark that matters for a high win rate:
    a small target and a wide stop win most of the time on ANY entry. The signal has earned
    its place only if it beats this, and by more than chance.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for symbol, frame in features.items():
        frame = frame.copy()
        ready = frame[["atr", "vwap", "ema", "stop_long", "stop_short"]].notna().all(axis=1).to_numpy()
        positions = np.flatnonzero(ready)
        for column in ("long_signal", "short_signal"):
            count = int(frame[column].sum())
            flags = np.zeros(len(frame), dtype=bool)
            if count and len(positions):
                flags[rng.choice(positions, size=min(count, len(positions)), replace=False)] = True
            frame[column] = flags
        out[symbol] = frame
    return out
