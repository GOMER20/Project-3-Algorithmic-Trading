"""Every tunable number in one place.

`StrategyConfig` is what the walk-forward optimiser searches over; `CostModel` describes the
exchange. Both are frozen so a configuration can be hashed, cached and written into a report
exactly as it was run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any


@dataclass(frozen=True)
class StrategyConfig:
    # --- which side of a stretch to take ------------------------------------------------
    playbook: str = "reversion"
    """'reversion'    fade a stretch, and only in a ranging regime (the original brief)
    'continuation' trade WITH a stretch, and only in a trending regime
    'adaptive'     let the regime gate choose between the two, bar by bar"""

    # --- timeframes ---------------------------------------------------------------------
    base_tf_sec: int = 60
    """Entry timeframe: the bars signals are computed on."""
    regime_tf_sec: int = 300
    """Higher timeframe the regime gate is measured on. Equal to base_tf_sec disables resampling."""

    # --- 1. regime gate -----------------------------------------------------------------
    adx_period: int = 14
    adx_max: float = 20.0
    hurst_window: int = 100
    hurst_max: float = 0.5
    regime_mode: str = "either"
    """'adx', 'hurst', 'either' (ADX below adx_max OR Hurst below hurst_max), 'both', or 'off'."""
    bb_period: int = 20
    bb_k: float = 2.0
    bbw_lookback: int = 100
    bbw_max_pctile: float = 0.5
    """Volatility must be consolidating: Bollinger band width in the lower part of its own
    recent range. 1.0 disables the check."""

    # --- 2. entry confluence ------------------------------------------------------------
    rsi_period: int = 2
    rsi_low: float = 10.0
    rsi_high: float = 90.0
    vwap_window: int = 30
    """Bars in the rolling VWAP and its deviation bands. 0 anchors the VWAP to the session."""
    entry_z: float = 2.5
    use_absorption: bool = True
    absorb_rel_volume: float = 1.3
    """Effort: the signal bar's volume relative to its 20-bar average."""
    absorb_wick_frac: float = 0.3
    """Result: the share of the bar's range left behind as a rejection wick."""
    sr_lookback: int = 20
    """Bars that define support and resistance (the recent extreme being tested)."""
    session_warmup_bars: int = 0
    """No entries until this many bars into a session (counted from any break in the bars)."""
    max_bar_atr: float = 0.0
    """Reject a signal bar whose range exceeds this many ATRs (a gap or news shock, which is a
    re-pricing rather than a dislocation). 0 disables."""

    # --- 3. execution -------------------------------------------------------------------
    entry_offset_atr: float = 0.0
    """How far behind the touch the entry limit rests, in ATRs. 0 joins the bid (ask for shorts)."""
    entry_style: str = "passive"
    """'passive' rests at the near touch and earns the spread; 'marketable' sends a limit
    through the far touch and pays the taker's slippage for a certain fill."""
    entry_ttl_bars: int = 2
    """Unfilled entry orders are cancelled after this many bars."""

    # --- 4. exits -----------------------------------------------------------------------
    target: str = "nearest"
    """'vwap', 'ema', or 'nearest' (whichever of the two the price reaches first)."""
    ema_period: int = 20
    target_frac: float = 1.0
    """Share of the distance from entry to the mean that is asked for. 1.0 is the mean itself."""
    dynamic_target: bool = True
    """Re-price the take-profit to the current mean on every bar, instead of freezing it at entry."""
    stop_z: float = 3.5
    stop_atr_mult: float = 1.5
    atr_period: int = 14
    max_hold_bars: int = 30
    """Time stop. A dislocation that has not closed by now was a re-pricing."""

    # --- 4b. exits for the continuation playbook ------------------------------------------
    cont_target_atr: float = 0.75
    """Take profit this many ATRs beyond the entry."""
    cont_stop_atr: float = 2.5
    """Hard stop this many ATRs behind the entry."""
    cont_max_hold_bars: int = 10
    # Filters that came out of reviewing losing trades. Each one removes a kind of entry that
    # lost more often than the rest; 0 (or 1.0 for the wick) switches a filter off.
    cont_min_di_gap: float = 0.0
    """The directional index must lean this far the trade's way (+DI minus -DI for a long)."""
    cont_max_wick: float = 1.0
    """Largest rejection wick against the trade allowed on the signal bar, as a share of its range."""
    max_rel_volume: float = 0.0
    """Skip signal bars with more than this multiple of normal volume (a climax, not a trend)."""
    max_atr_bps: float = 0.0
    """Skip symbols and moments where one-minute ATR exceeds this many basis points of price."""

    # --- sizing -------------------------------------------------------------------------
    risk_per_trade: float = 0.004
    """Fixed-fractional risk: share of equity lost if the hard stop is hit."""
    kelly_fraction: float = 0.25
    """Fraction of the Kelly bet used once enough trades exist to estimate the edge. 0 = fixed fractional only."""
    kelly_min_trades: int = 40
    kelly_lookback: int = 150
    risk_floor: float = 0.001
    risk_cap: float = 0.01
    max_notional_frac: float = 0.25
    """Largest single position as a share of equity, whatever the stop distance says."""
    max_open_positions: int = 4
    max_same_side: int = 0
    """Most positions allowed in one direction at once (0 = no separate limit). Several longs
    opened together are one bet on the market, and they stop out together."""
    session_flatten_bars: int = 3
    """No entries in a session's last bars; anything still open then is closed at market."""

    # --- circuit breaker ----------------------------------------------------------------
    max_daily_loss: float = 0.02
    allow_short: bool = True

    def with_(self, **changes: Any) -> "StrategyConfig":
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def warmup_bars(self) -> int:
        """Base bars of history needed before every indicator is defined."""
        per_regime_bar = max(1, self.regime_tf_sec // self.base_tf_sec)
        # Each term is when that indicator first has a value: the Bollinger-width percentile needs
        # its band plus half its lookback, Hurst its window plus its longest lag (16).
        regime = max(self.bb_period + max(10, self.bbw_lookback // 2), self.hurst_window + 16, 3 * self.adx_period) + 2
        base = max(self.vwap_window, self.sr_lookback, self.ema_period * 3, self.atr_period * 3, 25)
        # Both sets are computed over the same bars, so the longer one decides (plus a small margin).
        return max(regime * per_regime_bar, base) + 10

    @property
    def live_window(self) -> int:
        """Bars a live bot computes on. More than the warm-up, so the smoothed indicators (ADX
        above all, which is smoothed twice) have settled to the values a full history gives."""
        return self.warmup_bars + 60 * max(1, self.regime_tf_sec // self.base_tf_sec)

    # Fields that change indicator values, as opposed to thresholds applied to them.
    INDICATOR_FIELDS = (
        "base_tf_sec", "regime_tf_sec", "adx_period", "hurst_window", "bb_period", "bb_k", "bbw_lookback",
        "rsi_period", "vwap_window", "sr_lookback", "ema_period", "atr_period",
    )

    def indicator_key(self) -> tuple:
        return tuple(getattr(self, name) for name in self.INDICATOR_FIELDS)


@dataclass(frozen=True)
class CostModel:
    """What trading costs on the exchange being simulated.

    The defaults are Gomer Trading's execution rules: a taker pays one cent a share on top of
    the touch, a resting limit order fills at its own price when the opposite quote reaches
    it, and commission is whatever the competition charges per share.
    """

    taker_slippage: float = 0.01
    """Dollars per share paid by any order that takes liquidity."""
    commission_per_share: float = 0.0
    taker_fee_bps: float = 0.0
    maker_fee_bps: float = 0.0
    """Percentage fees, for venues that charge them (negative = rebate)."""
    stop_slippage_bps: float = 1.0
    """Extra slippage on stop-outs, which trade in a moving market."""
    maker_through_ticks: int = 0
    """Require the opposite quote to trade this many ticks THROUGH a resting limit before it
    counts as filled. 0 is the exchange's literal rule; 1 is a pessimistic queue assumption."""
    tick_size: float = 0.01
    fallback_spread_bps: float = 1.6
    """Spread assumed when the data has no bid/ask columns."""

    def with_(self, **changes: Any) -> "CostModel":
        return replace(self, **changes)


GOMER_COSTS = CostModel()
"""The practice market as configured by default (no commission)."""

STRESS_COSTS = CostModel(taker_slippage=0.02, commission_per_share=0.005, stop_slippage_bps=3.0, maker_through_ticks=1)
"""A deliberately harsh version, to see how much of the edge is cost assumption."""
