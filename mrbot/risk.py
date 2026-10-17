"""Position sizing and the daily-loss circuit breaker. Shared by the backtest and the live bot."""
from __future__ import annotations

import math
from collections import deque
from typing import Callable

from .config import StrategyConfig
from .strategy import EntryIntent


def kelly_risk_fraction(r_multiples: list[float]) -> float | None:
    """The Kelly fraction of equity to put at risk, from a sample of closed trades.

    Trades are measured in R (profit divided by the amount risked). With win probability p,
    average win b and average loss a (both in R), the growth-optimal stake is p/a - (1-p)/b.
    Returns None when the sample cannot support an estimate (no winners or no losers).
    """
    wins = [r for r in r_multiples if r > 0]
    losses = [-r for r in r_multiples if r <= 0]
    if not wins or not losses:
        return None
    p = len(wins) / len(r_multiples)
    avg_win = sum(wins) / len(wins)
    avg_loss = max(sum(losses) / len(losses), 1e-9)
    return p / avg_loss - (1.0 - p) / avg_win


class RiskManager:
    """Decides how many shares a signal gets, and whether the strategy may trade at all.

    Sizing is fixed-fractional until `kelly_min_trades` trades have closed; from then on the
    risk per trade is `kelly_fraction` of the Kelly stake estimated from the most recent
    trades, kept between `risk_floor` and `risk_cap`. A sample that shows no edge sizes at the
    floor instead of betting more to win it back.

    The circuit breaker compares equity with where the session started. Once the loss reaches
    `max_daily_loss` the strategy is halted until the next session; the caller is expected to
    cancel working orders and flatten when `update()` returns True.
    """

    def __init__(self, cfg: StrategyConfig, session_of: Callable[[object], object] | None = None):
        self.cfg = cfg
        self.session_of = session_of or (lambda ts: ts.floor("D") if hasattr(ts, "floor") else ts)
        self.session: object = None
        self.session_start_equity = 0.0
        self.halted = False
        self.halts: list[tuple[object, float]] = []
        self.r_history: deque[float] = deque(maxlen=cfg.kelly_lookback)

    # --- circuit breaker -----------------------------------------------------------------
    def update(self, ts: object, equity: float) -> bool:
        """Call at every mark. Returns True on the mark that trips the breaker."""
        session = self.session_of(ts)
        if session != self.session:
            self.session = session
            self.session_start_equity = equity
            self.halted = False
        if self.halted or self.session_start_equity <= 0:
            return False
        drawdown = 1.0 - equity / self.session_start_equity
        if drawdown >= self.cfg.max_daily_loss:
            self.halted = True
            self.halts.append((ts, drawdown))
            return True
        return False

    # --- sizing --------------------------------------------------------------------------
    def risk_fraction(self) -> float:
        cfg = self.cfg
        if cfg.kelly_fraction <= 0 or len(self.r_history) < cfg.kelly_min_trades:
            return min(cfg.risk_per_trade, cfg.risk_cap)
        kelly = kelly_risk_fraction(list(self.r_history))
        if kelly is None:
            return min(cfg.risk_per_trade, cfg.risk_cap)
        return min(cfg.risk_cap, max(cfg.risk_floor, cfg.kelly_fraction * kelly))

    def size(self, intent: EntryIntent, equity: float, buying_power: float | None = None) -> int:
        """Whole shares for this entry, or 0 when the trade should be skipped."""
        if self.halted or equity <= 0 or intent.risk_per_share <= 0:
            return 0
        by_risk = equity * self.risk_fraction() / intent.risk_per_share
        by_notional = equity * self.cfg.max_notional_frac / intent.limit_price
        shares = min(by_risk, by_notional)
        if buying_power is not None:
            shares = min(shares, 0.98 * buying_power / intent.limit_price)
        return max(0, math.floor(shares))

    def record_trade(self, r_multiple: float) -> None:
        self.r_history.append(r_multiple)
