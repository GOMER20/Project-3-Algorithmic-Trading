"""Performance statistics, computed the same way for a backtest and for a live session."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a win rate. Honest about small samples in a way a bare percentage is not."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((1.0 - equity / peak).max())


def longest_losing_streak(pnl: np.ndarray) -> int:
    longest = current = 0
    for value in pnl:
        current = current + 1 if value <= 0 else 0
        longest = max(longest, current)
    return longest


def summarize(trades: pd.DataFrame, equity: pd.Series | None, starting_equity: float, periods_per_year: int = 365) -> dict:
    """One flat dict of statistics. `trades` needs pnl, r_multiple, notional, bars_held, exit_reason, fees."""
    n = len(trades)
    out: dict = {"trades": n}
    if n == 0:
        out.update(win_rate=float("nan"), expectancy=0.0, net_pnl=0.0, total_return=0.0, profit_factor=float("nan"), t_stat=0.0)
        if equity is not None and len(equity):
            out["max_drawdown"] = max_drawdown(equity)
        return out

    pnl = trades["pnl"].to_numpy(dtype=float)
    r = trades["r_multiple"].to_numpy(dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    low, high = wilson_interval(len(wins), n)
    gross_win, gross_loss = wins.sum(), -losses.sum()
    bps = 1e4 * pnl / trades["notional"].to_numpy(dtype=float)

    out.update(
        win_rate=len(wins) / n,
        win_rate_ci_low=low,
        win_rate_ci_high=high,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        payoff_ratio=float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else float("nan"),
        profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        expectancy=float(pnl.mean()),
        expectancy_bps=float(bps.mean()),
        expectancy_r=float(r.mean()),
        # How many standard errors the average trade is from zero. Below about 2 the edge is not established.
        t_stat=float(r.mean() / (r.std(ddof=1) / math.sqrt(n))) if n > 1 and r.std(ddof=1) > 0 else 0.0,
        net_pnl=float(pnl.sum()),
        total_return=float(pnl.sum() / starting_equity),
        fees=float(trades["fees"].sum()),
        worst_trade=float(pnl.min()),
        worst_trade_r=float(r.min()),
        best_trade=float(pnl.max()),
        largest_loss_vs_avg_win=float(-pnl.min() / wins.mean()) if len(wins) and pnl.min() < 0 else 0.0,
        longest_losing_streak=longest_losing_streak(pnl),
        avg_bars_held=float(trades["bars_held"].mean()),
        long_share=float((trades["side"] > 0).mean()),
        exits={reason: {"trades": int(len(g)), "win_rate": float((g["pnl"] > 0).mean()), "pnl": float(g["pnl"].sum())} for reason, g in trades.groupby("exit_reason")},
    )
    if equity is not None and len(equity) > 1:
        out["max_drawdown"] = max_drawdown(equity)
        daily = equity.resample("1D").last().dropna()
        returns = daily.pct_change().dropna()
        days = max(1, len(daily))
        out["days"] = days
        out["trades_per_day"] = n / days
        out["sharpe"] = float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year)) if len(returns) > 2 and returns.std(ddof=1) > 0 else float("nan")
        out["worst_day"] = float(returns.min()) if len(returns) else 0.0
        out["profitable_days"] = float((returns > 0).mean()) if len(returns) else float("nan")
    return out


def format_stats(stats: dict, title: str = "") -> str:
    """A compact plain-text block for the terminal and for reports."""
    if stats.get("trades", 0) == 0:
        return f"{title}\n  no trades"
    lines = [title] if title else []
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines += [
        f"  trades               {stats['trades']}" + (f"  ({stats['trades_per_day']:.1f} per day over {stats['days']} days)" if "trades_per_day" in stats else ""),
        f"  win rate             {pct(stats['win_rate'])}   (95% interval {pct(stats['win_rate_ci_low'])} to {pct(stats['win_rate_ci_high'])})",
        f"  average win / loss   ${stats['avg_win']:,.2f} / ${stats['avg_loss']:,.2f}   (payoff ratio {stats['payoff_ratio']:.2f})",
        f"  profit factor        {stats['profit_factor']:.2f}",
        f"  expectancy           ${stats['expectancy']:,.2f} per trade   ({stats['expectancy_bps']:.2f} bps, {stats['expectancy_r']:.3f} R, t = {stats['t_stat']:.2f})",
        f"  net profit           ${stats['net_pnl']:,.2f}   ({pct(stats['total_return'])} of starting equity)",
        f"  worst trade          ${stats['worst_trade']:,.2f}   ({stats['worst_trade_r']:.2f} R, {stats['largest_loss_vs_avg_win']:.1f}x the average win)",
        f"  longest losing run   {stats['longest_losing_streak']} trades",
        f"  average hold         {stats['avg_bars_held']:.1f} bars",
    ]
    if "max_drawdown" in stats:
        lines.append(f"  max drawdown         {pct(stats['max_drawdown'])}")
    if stats.get("sharpe") == stats.get("sharpe") and "sharpe" in stats:  # needs several days; absent or NaN for a single session
        lines.append(f"  daily Sharpe (ann.)  {stats['sharpe']:.2f}   worst day {100 * stats['worst_day']:+.1f}%, profitable days {pct(stats['profitable_days'])}")
    exits = ", ".join(f"{reason} {d['trades']} ({pct(d['win_rate'])} won)" for reason, d in sorted(stats["exits"].items()))
    lines.append(f"  exits                {exits}")
    return "\n".join(lines)
