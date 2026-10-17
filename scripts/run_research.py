"""The whole research run: walk-forward on the main history, then every check on a holdout.

    .venv/bin/python scripts/run_research.py [--candidates 240] [--workers 7]

Writes reports/backtest_report.md, reports/backtest_results.json, reports/equity.png and the
chosen configuration to configs/at.json (which the live bot loads).

The holdout history comes from a different random seed than the one the parameters were
chosen on, and is touched exactly once, at the end, with the parameters already frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mrbot.backtest import Backtester, FeatureStore  # noqa: E402
from mrbot.config import GOMER_COSTS, STRESS_COSTS, StrategyConfig  # noqa: E402
from mrbot.data import load_dir, random_walk_bars  # noqa: E402
from mrbot.metrics import format_stats, summarize  # noqa: E402
from mrbot.walkforward import SEARCH_SPACE, SEARCH_SPACE_V2, run_candidates, sample_configs, select_final, shuffle_signals, walk_forward  # noqa: E402

EQUITY = 100_000.0


def line(stats: dict) -> str:
    if not stats.get("trades"):
        return "| 0 | – | – | – | – | – |"
    return (
        f"| {stats['trades']:,} | {100 * stats['win_rate']:.1f}% | {stats['expectancy_bps']:+.2f} | {stats['t_stat']:+.1f} "
        f"| {stats['profit_factor']:.2f} | {100 * stats['total_return']:+.1f}% |"
    )


HEAD = "| trades | win rate | avg trade (bps) | t-stat | profit factor | return* |\n|---|---|---|---|---|---|"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/sessions"))
    parser.add_argument("--holdout", default=str(ROOT / "data/sessions_holdout"))
    parser.add_argument("--candidates", type=int, default=300)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--is-days", type=int, default=28, help="calendar days chosen on (28 = 20 sessions)")
    parser.add_argument("--oos-days", type=int, default=7, help="calendar days tested on (7 = 5 sessions)")
    parser.add_argument("--min-trades", type=int, default=40)
    parser.add_argument("--target-win-rate", type=float, default=0.90)
    parser.add_argument("--space", choices=["v1", "v2"], default="v2", help="v1: all three playbooks from the brief; v2: continuation settings and loss-review filters")
    parser.add_argument("--base", default=str(ROOT / "configs/at_v1.json"), help="v2: the configuration being improved on (candidate 0)")
    args = parser.parse_args()
    started = time.time()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "configs").mkdir(exist_ok=True)

    brief = StrategyConfig()  # the brief, taken literally
    previous = StrategyConfig(**json.loads(Path(args.base).read_text())) if args.space == "v2" else brief
    configs = sample_configs(previous, args.candidates, seed=2, space=SEARCH_SPACE_V2) if args.space == "v2" else sample_configs(brief, args.candidates, seed=1)
    print(f"Running {len(configs)} candidates over the main history ...", flush=True)
    all_trades = run_candidates(args.data, configs, GOMER_COSTS, EQUITY, workers=args.workers)

    main_bars = load_dir(args.data)
    index = next(iter(main_bars.values())).index
    wf = walk_forward(all_trades, configs, index[0], index[-1], args.is_days, args.oos_days, args.min_trades, args.target_win_rate, EQUITY)
    chosen, chosen_stats = select_final(wf, args.min_trades * 3, args.target_win_rate, EQUITY)
    final = configs[chosen]
    (ROOT / "configs/at.json").write_text(json.dumps(final.to_dict(), indent=2) + "\n")
    print(f"Walk-forward done in {time.time() - started:.0f}s. Final candidate #{chosen}: playbook={final.playbook}", flush=True)

    # ---- holdout: frozen parameters, different seed, used once ---------------------------
    holdout = FeatureStore(load_dir(args.holdout))
    features = holdout.features(final)
    run = lambda cfg, costs=GOMER_COSTS, feats=None, **kw: Backtester(cfg, costs, EQUITY, compound=False, **kw).run(feats or holdout.features(cfg))  # noqa: E731
    hold = run(final, feats=features)
    hold_stress = run(final, STRESS_COSTS, feats=features)
    hold_brief = run(brief)
    hold_strict = run(brief.with_(regime_mode="both"))
    hold_previous = run(previous) if previous != brief else None

    def steadiness(trades: pd.DataFrame) -> dict:
        """How the success rate holds up over short stretches and whole sessions, not just on average."""
        if not len(trades):
            return {}
        t = trades.assign(when=pd.to_datetime(trades["entry_time"]), won=trades["pnl"] > 0).set_index("when").sort_index()
        half_hours = t.groupby(pd.Grouper(freq="30min")).agg(n=("won", "size"), rate=("won", "mean"), pnl=("pnl", "sum")).query("n >= 8")
        days = t.groupby(t.index.floor("D")).agg(n=("won", "size"), rate=("won", "mean"), pnl=("pnl", "sum")).query("n >= 20")
        return {
            "overall": float(t.won.mean()),
            "half_hours": int(len(half_hours)), "half_hour_p5": float(half_hours.rate.quantile(0.05)), "half_hour_p25": float(half_hours.rate.quantile(0.25)),
            "half_hour_median": float(half_hours.rate.median()), "half_hours_below_70": float((half_hours.rate < 0.70).mean()),
            "half_hours_losing": float((half_hours.pnl < 0).mean()),
            "sessions": int(len(days)), "session_min": float(days.rate.min()), "session_median": float(days.rate.median()), "session_max": float(days.rate.max()),
            "sessions_below_80": int((days.rate < 0.80).sum()), "sessions_losing": int((days.pnl < 0).sum()), "trades_per_session": float(days.n.median()),
        }

    steady = steadiness(hold.trades)
    steady_previous = steadiness(hold_previous.trades) if hold_previous is not None else {}

    random_runs = [run(final, feats=shuffle_signals(features, seed)).stats for seed in range(5)]
    random_win = float(np.mean([s["win_rate"] for s in random_runs if s["trades"]]))
    random_bps = float(np.mean([s["expectancy_bps"] for s in random_runs if s["trades"]]))

    ablations = {
        "all filters (as chosen)": final,
        "without the regime gate": final.with_(regime_mode="off", bbw_max_pctile=1.0),
        "without RSI(2)": final.with_(rsi_low=101.0, rsi_high=-1.0),
        "without the VWAP band": final.with_(entry_z=-1e9),
        "without the order-flow filter": final.with_(use_absorption=False),
        "with the order-flow filter": final.with_(use_absorption=True),
        "without the loss-review filters": final.with_(cont_min_di_gap=0.0, cont_max_wick=1.0, max_rel_volume=0.0, max_atr_bps=0.0),
        "without the same-side limit": final.with_(max_same_side=0),
    }
    ablation_stats = {name: run(cfg).stats for name, cfg in ablations.items() if name == "all filters (as chosen)" or cfg != final}

    null = FeatureStore({f"RW{i}": random_walk_bars(30_000, seed=100 + i, sigma_bps=20.0 + 3 * i) for i in range(8)})
    null_run = Backtester(final, GOMER_COSTS, EQUITY, compound=False).run(null.features(final))

    moods = {}
    if len(hold.trades):
        mood_at = pd.concat([f["mood"].rename(s) for s, f in holdout.bars.items()], axis=1)
        tags = [mood_at[sym].asof(ts) for sym, ts in zip(hold.trades["symbol"], hold.trades["entry_time"])]
        for mood, group in hold.trades.assign(mood=tags).groupby("mood"):
            moods[mood] = summarize(group, None, EQUITY)

    # ---- chart ---------------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        if len(wf.oos_trades):
            oos = wf.oos_trades.set_index("exit_time")["pnl"].cumsum()
            axes[0].plot(oos.index, 100 * oos.values / EQUITY, color="#1f6feb")
        axes[0].set_title("Walk-forward: stitched out-of-sample P&L (% of starting equity)")
        axes[1].plot(hold.equity.index, 100 * (hold.equity.values / EQUITY - 1), color="#1a7f37", label="practice-market costs")
        axes[1].plot(hold_stress.equity.index, 100 * (hold_stress.equity.values / EQUITY - 1), color="#bf8700", label="stress costs")
        axes[1].plot(hold_brief.equity.index, 100 * (hold_brief.equity.values / EQUITY - 1), color="#cf222e", label="brief, literal")
        axes[1].set_title("Holdout (unseen seed), frozen parameters")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(ROOT / "reports/equity.png", dpi=130)
    except Exception as err:  # the chart is a convenience; the numbers are in the report either way
        print(f"(chart skipped: {err})")

    # ---- report --------------------------------------------------------------------------
    by_playbook = {}
    for playbook in [p for p in SEARCH_SPACE["playbook"] if any(c.playbook == p for c in configs)]:
        stats = [summarize(t, None, EQUITY) for t, c in zip(all_trades, configs) if c.playbook == playbook]
        stats = [s for s in stats if s["trades"] >= args.min_trades]
        by_playbook[playbook] = {
            "candidates": len(stats),
            "profitable": int(sum(s["expectancy"] > 0 for s in stats)),
            "median_win_rate": float(np.median([s["win_rate"] for s in stats])) if stats else float("nan"),
            "median_bps": float(np.median([s["expectancy_bps"] for s in stats])) if stats else float("nan"),
            "best_t": float(max((s["t_stat"] for s in stats), default=float("nan"))),
        }

    changed = {k: v for k, v in final.to_dict().items() if getattr(brief, k) != v}
    s = hold.stats
    md = [
        "# AT — backtest and walk-forward report",
        "",
        f"History: Gomer Trading's practice-market model, one-minute bars, {len(main_bars)} symbols, US exchange hours with an "
        f"overnight price gap each morning (as the app's practice market now runs). Parameters were chosen on "
        f"{index.normalize().nunique()} sessions and then tested once on a separate "
        f"{next(iter(holdout.bars.values())).index.normalize().nunique()}-session history from a different random seed "
        f"that played no part in choosing them.",
        "",
        "\\* Return is on a fixed $100,000 base with no compounding, so periods can be compared.",
        "",
        "## 1. The brief, taken literally",
        "",
        "Fade stretched prices in a ranging regime; target the mean; stop at 3.5 deviations.",
        "",
        "| regime gate " + HEAD.split("\n")[0],
        "|---" + HEAD.split("\n")[1],
        f"| as written (ADX < 20 **or** Hurst < 0.5) {line(hold_brief.stats)}",
        f"| strict (ADX < 20 **and** Hurst < 0.5) {line(hold_strict.stats)}",
        "",
        "## 2. What the search found, by playbook (main history)",
        "",
        "| playbook | candidates with enough trades | profitable | median win rate | median avg trade (bps) | best t-stat |",
        "|---|---|---|---|---|---|",
        *[
            f"| {name} | {d['candidates']} | {d['profitable']} | {100 * d['median_win_rate']:.1f}% | {d['median_bps']:+.2f} | {d['best_t']:+.1f} |"
            for name, d in by_playbook.items()
        ],
        "",
        "## 3. Walk-forward",
        "",
        f"{len(wf.folds)} folds: choose on {args.is_days} days, test on the next {args.oos_days}. {len(configs)} candidates.",
        "",
        "| fold | chosen playbook | in-sample win rate | in-sample bps | out-of-sample trades | out-of-sample win rate | out-of-sample bps |",
        "|---|---|---|---|---|---|---|",
        *[
            f"| {f.number} | {configs[f.chosen].playbook} | {100 * f.is_stats['win_rate']:.1f}% | {f.is_stats.get('expectancy_bps', 0):+.2f} "
            f"| {f.oos_stats['trades']:,} | {100 * f.oos_stats['win_rate']:.1f}% | {f.oos_stats.get('expectancy_bps', 0):+.2f} |"
            for f in wf.folds
        ],
        "",
        "Stitched out-of-sample record:",
        "",
        "```",
        format_stats(wf.oos_stats),
        "```",
        "",
        "Overfitting checks:",
        "",
        f"- **Walk-forward efficiency {wf.efficiency:.2f}** (out-of-sample edge ÷ in-sample edge; near 1 is good, near 0 is a curve fit).",
        f"- **Rank correlation {wf.rank_correlation:+.2f}** between a candidate's in-sample and out-of-sample results.",
        f"- **Probability of overfit {100 * wf.probability_of_overfit:.0f}%** (how often the in-sample winner fell in the bottom half out-of-sample).",
        f"- **Trials hurdle t = {wf.trials_hurdle:.1f}**: the best of {len(configs)} worthless candidates would reach about this by luck.",
        f"- Out-of-sample folds profitable: {sum(f.oos_stats.get('expectancy', 0) > 0 for f in wf.folds)} of {len(wf.folds)}.",
        "",
        "## 4. Holdout: frozen parameters on a history they have never seen",
        "",
        "| costs " + HEAD.split("\n")[0],
        "|---" + HEAD.split("\n")[1],
        f"| practice market (1¢ taker slippage, no commission) {line(hold.stats)}",
        f"| stress (double slippage, $0.005/share commission, pessimistic queue) {line(hold_stress.stats)}",
        *([f"| the previous configuration, for comparison {line(hold_previous.stats)}"] if hold_previous is not None else []),
        "",
        "```",
        format_stats(hold.stats),
        "```",
        "",
        f"Orders sent {hold.orders_placed:,}, filled {100 * hold.stats['fill_rate']:.0f}%. Circuit-breaker halts: {hold.stats['breaker_halts']}.",
        "",
        "### How steady is the success rate?",
        "",
        "An average hides bad stretches. The same holdout, cut into half-hours (with at least 8 trades) and into sessions:",
        "",
        "| | this configuration |" + (" previous configuration |" if steady_previous else ""),
        "|---|---|" + ("---|" if steady_previous else ""),
        *[
            f"| {label} | {fmt(steady.get(key))} |" + (f" {fmt(steady_previous.get(key))} |" if steady_previous else "")
            for label, key, fmt in (() if not steady else (
                ("overall success rate", "overall", lambda v: f"{100 * v:.1f}%"),
                ("typical half-hour (median)", "half_hour_median", lambda v: f"{100 * v:.0f}%"),
                ("a poor half-hour (1 in 4 is worse)", "half_hour_p25", lambda v: f"{100 * v:.0f}%"),
                ("a bad half-hour (1 in 20 is worse)", "half_hour_p5", lambda v: f"{100 * v:.0f}%"),
                ("half-hours under 70%", "half_hours_below_70", lambda v: f"{100 * v:.1f}%"),
                ("half-hours that lost money", "half_hours_losing", lambda v: f"{100 * v:.1f}%"),
                ("worst session", "session_min", lambda v: f"{100 * v:.0f}%"),
                ("typical session (median)", "session_median", lambda v: f"{100 * v:.0f}%"),
                ("sessions under 80%", "sessions_below_80", lambda v: f"{v} of {steady.get('sessions')}"),
                ("sessions that lost money", "sessions_losing", lambda v: f"{v} of {steady.get('sessions')}"),
                ("trades per session (median)", "trades_per_session", lambda v: f"{v:.0f}"),
            ))
        ],
        "",
        "## 5. Is the win rate the signal, or just the exit?",
        "",
        "A small target and a wide stop win most of the time on any entry. Same exits, same number of entries, random bars:",
        "",
        "| entries | win rate | avg trade (bps) |",
        "|---|---|---|",
        f"| the strategy's signals | {100 * s['win_rate']:.1f}% | {s['expectancy_bps']:+.2f} |",
        f"| random bars (5 runs) | {100 * random_win:.1f}% | {random_bps:+.2f} |",
        "",
        "## 6. What each filter contributes (holdout)",
        "",
        "| variant " + HEAD.split("\n")[0],
        "|---" + HEAD.split("\n")[1],
        *[f"| {name} {line(st)}" for name, st in ablation_stats.items()],
        "",
        "## 7. A market with nothing to predict",
        "",
        "Eight random walks with a spread, 30,000 minutes each. A correct backtest should lose roughly its costs here.",
        "",
        HEAD,
        line(null_run.stats),
        "",
        "## 8. By the practice market's hidden mood (diagnostic; the strategy never sees this)",
        "",
        "| mood " + HEAD.split("\n")[0],
        "|---" + HEAD.split("\n")[1],
        *[f"| {mood} {line(st)}" for mood, st in moods.items()],
        "",
        "## 9. Chosen configuration",
        "",
        "Differences from the brief's literal settings:",
        "",
        "```json",
        json.dumps(changed, indent=2),
        "```",
        "",
        "![equity](equity.png)",
        "",
    ]
    (ROOT / "reports/backtest_report.md").write_text("\n".join(md))
    results = {
        "final_config": final.to_dict(),
        "walk_forward": {
            "oos": wf.oos_stats, "efficiency": wf.efficiency, "rank_correlation": wf.rank_correlation,
            "probability_of_overfit": wf.probability_of_overfit, "trials_hurdle": wf.trials_hurdle,
            "folds": [{"fold": f.number, "playbook": configs[f.chosen].playbook, "is": f.is_stats, "oos": f.oos_stats} for f in wf.folds],
        },
        "holdout": hold.stats, "holdout_stress": hold_stress.stats, "holdout_brief": hold_brief.stats, "holdout_brief_strict": hold_strict.stats,
        "holdout_previous": hold_previous.stats if hold_previous is not None else None, "steadiness": steady, "steadiness_previous": steady_previous,
        "random_entry": {"win_rate": random_win, "expectancy_bps": random_bps},
        "ablation": ablation_stats, "null_market": null_run.stats, "by_playbook": by_playbook, "by_mood": moods,
    }
    (ROOT / "reports/backtest_results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nDone in {time.time() - started:.0f}s. Report: reports/backtest_report.md\n")
    print(format_stats(hold.stats, "HOLDOUT, frozen parameters"))


if __name__ == "__main__":
    main()
