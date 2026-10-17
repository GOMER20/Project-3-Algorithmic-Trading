"""After a session: what went wrong on the losing trades, and whether the algorithm should change.

A loss is not automatically a mistake. This strategy takes a small profit often and a larger
loss sometimes, so some losses are part of it working. What a review has to find out is which
kind each loss was:

  expected    the backtester, replaying the very same minutes, takes the same trade and
              loses it too. The rules did what they were built to do.
  execution   the backtester wins the trade that AT lost live. Something between the rules
              and the exchange went differently: a slow fill, a missed target, a bad price.
  unplanned   the backtester would not have taken the trade at all.

Then it asks whether the session as a whole fell short of the long-run success rate by more
than luck allows. Only that, or execution problems, is a reason to change the algorithm.
Changing rules after every losing trade fits the algorithm to noise and makes it worse, so
single losses are recorded and explained here, and the rules change when the record says so.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .backtest import Backtester
from .config import StrategyConfig
from .strategy import LONG, RegimeMeanReversion

OHLCV = ["open", "high", "low", "close", "volume"]


def binomial_tail(wins: int, n: int, p: float) -> float:
    """Chance of `wins` or fewer successes in `n` tries when each succeeds with probability `p`."""
    return sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(0, wins + 1))


def review_session(trades: pd.DataFrame, bars: dict[str, pd.DataFrame], cfg: StrategyConfig, expected: dict | None = None) -> dict:
    """Compares a live session with the backtester's replay of the same minutes.

    `trades` is the journal's record; `bars` maps symbol to one-minute OHLCV covering the
    session and the warm-up before it; `expected` is the backtest's holdout statistics.
    Returns the findings as a dict, with a Markdown write-up under "text".
    """
    if trades is None or len(trades) == 0:
        return {"trades": 0, "update_recommended": False, "text": "## Loss review\n\nNo trades, so nothing to review.\n"}
    live = trades.copy()
    live["entry_time"] = pd.to_datetime(live["entry_time"], utc=True)
    live["exit_time"] = pd.to_datetime(live["exit_time"], utc=True)
    live = live.sort_values("entry_time").reset_index(drop=True)

    strategy = RegimeMeanReversion(cfg)
    features = {s: strategy.features(b[OHLCV]) for s, b in bars.items() if len(b) >= cfg.warmup_bars}
    start = live["entry_time"].min().floor("min") - pd.Timedelta(minutes=2)
    end = live["exit_time"].max().ceil("min") + pd.Timedelta(minutes=1)
    replay = Backtester(cfg, starting_equity=100_000.0, fixed_risk=True, compound=False).run(features, start, end).trades if features else pd.DataFrame()
    replay = replay.assign(entry_time=pd.to_datetime(replay["entry_time"], utc=True)) if len(replay) else replay

    # The backtester stamps a trade with its signal bar; live, the order goes in a minute later.
    used: set[int] = set()
    matches: list[int | None] = []
    for trade in live.itertuples():
        found = None
        if len(replay):
            minute = trade.entry_time.floor("min")
            candidates = replay[(replay["symbol"] == trade.symbol) & (replay["side"] == trade.side) & ~replay.index.isin(used)]
            gap = (candidates["entry_time"] + pd.Timedelta(minutes=1) - minute).abs()
            if len(gap) and gap.min() <= pd.Timedelta(minutes=1):
                found = int(gap.idxmin())
                used.add(found)
        matches.append(found)

    losers = []
    for row, match in zip(live.itertuples(), matches):
        if row.pnl > 0:
            continue
        twin = replay.loc[match] if match is not None else None
        kind = "unplanned" if twin is None else "expected" if twin["pnl"] <= 0 else "execution"
        signal_bar = row.entry_time.floor("min") - pd.Timedelta(minutes=1)
        frame = features.get(row.symbol)
        context = {}
        if frame is not None and signal_bar in frame.index:
            f = frame.loc[signal_bar]
            lean = float(f["plus_di"] - f["minus_di"]) * row.side
            wick = float(f["upper_wick"] if row.side == LONG else f["lower_wick"])
            context = {"lean": round(lean, 1), "adx": round(float(f["adx"]), 1), "wick": round(wick, 2),
                       "rel_volume": round(float(f["rel_volume"]), 2), "atr_bps": round(1e4 * float(f["atr"]) / float(f["close"]), 1)}
        together = int(((live["side"] == row.side) & (live["entry_time"] <= row.entry_time) & (live["exit_time"] > row.entry_time)).sum()) - 1
        losers.append({
            "symbol": row.symbol, "side": "long" if row.side == LONG else "short", "time": row.entry_time.isoformat(), "pnl": float(row.pnl),
            "exit": row.exit_reason, "kind": kind, "same_side_open": together, **context,
            "replay_exit": None if twin is None else twin["exit_reason"], "replay_pnl": None if twin is None else float(twin["pnl"]),
        })

    n, wins = len(live), int((live["pnl"] > 0).sum())
    replay_n = len(replay)
    replay_wins = int((replay["pnl"] > 0).sum()) if replay_n else 0
    long_run = float(expected["win_rate"]) if expected and expected.get("win_rate") else None
    chance = binomial_tail(wins, n, long_run) if long_run else None
    by_kind = {k: sum(1 for x in losers if x["kind"] == k) for k in ("expected", "execution", "unplanned")}
    clustered = sum(1 for x in losers if x["same_side_open"] >= 2)
    missed = replay_n - len(used)

    execution_problem = by_kind["execution"] + by_kind["unplanned"] > max(1, 0.10 * n)
    below_by_more_than_luck = chance is not None and n >= 30 and chance < 0.01
    update = bool(execution_problem or below_by_more_than_luck)

    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "n/a"  # noqa: E731
    lines = ["## Loss review", ""]
    lines.append(f"{n - wins} of {n} trades lost. Replaying the same minutes, the backtester made {replay_n} trades and won {pct(replay_wins, replay_n)} "
                 f"(live: {pct(wins, n)}" + (f"; long-run expectation {100 * long_run:.1f}%" if long_run else "") + ").")
    lines.append("")
    if losers:
        lines += [
            f"- **{by_kind['expected']} expected**: the backtester took the same trade and lost it too. The rules did what they were built to do.",
            f"- **{by_kind['execution']} execution**: the backtester won a trade AT lost live.",
            f"- **{by_kind['unplanned']} unplanned**: the backtester would not have taken the trade.",
        ]
        if clustered:
            lines.append(f"- {clustered} of the losing trades were opened while two or more other positions were already open in the same direction: one market move, counted several times.")
        if missed:
            lines.append(f"- The backtester also made {missed} trades in this window that AT did not.")
        lines.append("")
    if execution_problem:
        verdict = "**Execution needs looking at.** Too many losing trades do not match the backtest. The trades are listed in loss_review.json; do not trust this session's success rate until they are explained."
    elif below_by_more_than_luck:
        verdict = (f"**The algorithm needs re-tuning.** A success rate this far below {100 * long_run:.1f}% over {n} trades would happen by chance about "
                   f"{100 * chance:.2f}% of the time. Re-run the research before the next session.")
    elif chance is not None and n >= 30:
        verdict = (f"**No mistake found, and no change indicated.** The losses match the backtest, and a session at or below this success rate happens by chance about "
                   f"{100 * chance:.0f}% of the time.")
    else:
        verdict = "**No mistake found.** The losses match the backtest. There are too few trades here to judge the success rate against the long run."
    lines += [verdict, ""]
    return {
        "trades": n, "wins": wins, "win_rate": wins / n, "replay_trades": replay_n, "replay_win_rate": replay_wins / replay_n if replay_n else None,
        "long_run_win_rate": long_run, "chance_of_this_or_worse": chance, "losers_by_kind": by_kind, "losers_clustered": clustered,
        "replay_trades_missed": missed, "update_recommended": update, "losers": losers, "text": "\n".join(lines),
    }


def record(review: dict, session: str, path: Path) -> None:
    """Appends one line per session to the running record that decides when the rules change."""
    row = {k: review.get(k) for k in ("trades", "wins", "win_rate", "replay_win_rate", "long_run_win_rate", "chance_of_this_or_worse", "losers_by_kind", "update_recommended")}
    with open(path, "a") as handle:
        handle.write(json.dumps({"session": session, **row}, default=float) + "\n")
