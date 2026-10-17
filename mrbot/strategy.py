"""Regime-filtered statistical mean reversion.

One class holds the whole trading rule, and both the backtester and the live bot call it, so
what is tested is what trades:

    indicators(bars)  raw, threshold-free indicator columns (expensive, cacheable)
    signals(frame)    the regime gate and entry confluence applied to those columns
    entry(row, ...)   the order to send for a signal bar, with its stop and target
    manage(row, pos)  what to do with an open position at each bar close

A trade needs every independent condition at once:

    1. Regime gate   volatility consolidating (Bollinger width low in its own range) and the
                     higher timeframe not trending (ADX below threshold, or Hurst below 0.5)
    2. Stretch       RSI(2) at an extreme AND price more than `entry_z` deviations from VWAP
    3. Absorption    heavy volume at support/resistance that failed to extend the move
    4. Room          the mean is far enough away to pay for the trade, and the stop is defined

Exits: take profit at the mean (VWAP or EMA), a hard stop beyond `stop_z` deviations or
`stop_atr_mult` ATRs through support/resistance, and a time stop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import indicators as ind
from .config import StrategyConfig

LONG, SHORT = 1, -1
REVERSION, CONTINUATION = 1, 2
REQUIRED = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class EntryIntent:
    side: int
    limit_price: float
    stop_price: float
    target_price: float
    atr: float
    z: float
    kind: int = REVERSION
    marketable: bool = False
    """True when the limit is priced through the far touch and will trade on arrival."""

    @property
    def risk_per_share(self) -> float:
        return abs(self.limit_price - self.stop_price)


@dataclass
class Position:
    symbol: str
    side: int
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: object
    bars_held: int = 0
    kind: int = REVERSION
    initial_risk: float = 0.0
    """Dollars lost if the hard stop is hit at its price; the unit that R-multiples are measured in."""


@dataclass(frozen=True)
class ExitIntent:
    target_price: float
    """Where the take-profit limit should rest for the coming bar."""
    exit_now: bool = False
    reason: str = ""


def round_tick(price: float, tick: float, direction: int) -> float:
    """Rounds to the tick grid: direction -1 rounds down, +1 rounds up."""
    scaled = price / tick
    stepped = math.floor(scaled + 1e-9) if direction < 0 else math.ceil(scaled - 1e-9)
    return round(stepped * tick, 6)


class RegimeMeanReversion:
    def __init__(self, cfg: StrategyConfig | None = None, tick_size: float = 0.01):
        self.cfg = cfg or StrategyConfig()
        self.tick = tick_size

    # ------------------------------------------------------------------ indicators
    def indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Raw indicator columns. Depends only on the indicator periods, not on any threshold.

        Sessions. A break in the bars (the market closing overnight, or the feed being down)
        starts a new session, and the price usually jumps across it. That jump is a
        re-pricing, not a move anyone could have traded, so indicators are computed on a
        series with each jump taken out (every earlier price is scaled by the jump's ratio)
        and the price levels are then scaled back. The first bar of a day therefore reads as
        continuing from yesterday's close, instead of as a huge one-minute move.
        """
        cfg = self.cfg
        missing = [c for c in REQUIRED if c not in bars.columns]
        if missing:
            raise ValueError(f"bars is missing columns: {missing}")
        out = bars.copy()
        step = out.index.to_series().diff().dt.total_seconds()
        new_session = (step > 2 * cfg.base_tf_sec).fillna(False)
        out["bars_in_session"] = out.groupby(new_session.cumsum()).cumcount() + 1
        jump = (out["open"] / out["close"].shift(1)).where(new_session, 1.0).fillna(1.0)
        scale = jump.cumprod()  # real price / gap-adjusted price; constant within a session

        work = out[REQUIRED].copy()
        for column in ("open", "high", "low", "close"):
            work[column] = out[column] / scale
        high, low, close, volume = work["high"], work["low"], work["close"], work["volume"]
        typical = (high + low + close) / 3.0

        out["rsi"] = ind.rsi(close, cfg.rsi_period)
        atr = ind.atr(high, low, close, cfg.atr_period)
        ema = ind.ema(close, cfg.ema_period)
        if cfg.vwap_window > 0:
            vwap, vwap_sd = ind.rolling_vwap(typical, volume, cfg.vwap_window)
        else:
            session = pd.Series(out.index.floor("D"), index=out.index)
            vwap, vwap_sd = ind.session_vwap(typical, volume, session)
        out["z"] = (close - vwap) / vwap_sd.replace(0.0, np.nan)

        # Order flow: effort (volume) against result (how much of the bar's range held).
        span = (high - low).replace(0.0, np.nan)
        body_low = np.minimum(work["open"], close)
        body_high = np.maximum(work["open"], close)
        out["lower_wick"] = ((body_low - low) / span).fillna(0.0)
        out["upper_wick"] = ((high - body_high) / span).fillna(0.0)
        out["rel_volume"] = volume / volume.rolling(20).mean().shift(1)
        delta = ind.volume_delta(out)
        out["delta_ratio"] = (delta / volume.replace(0.0, np.nan)).fillna(0.0)
        out["bar_atr"] = (high - low) / atr

        # Support and resistance: the extremes of the recent range, including the bar testing them.
        support = low.rolling(cfg.sr_lookback).min()
        resistance = high.rolling(cfg.sr_lookback).max()
        prior_support = low.shift(1).rolling(cfg.sr_lookback).min()
        prior_resistance = high.shift(1).rolling(cfg.sr_lookback).max()
        out["at_support"] = low <= prior_support + 0.25 * atr
        out["at_resistance"] = high >= prior_resistance - 0.25 * atr

        # Price levels go back to real prices; ratios and oscillators need no conversion.
        for name, level in (("atr", atr), ("ema", ema), ("vwap", vwap), ("vwap_sd", vwap_sd), ("support", support), ("resistance", resistance)):
            out[name] = level * scale

        regime = self._regime(work)
        for column in regime.columns:
            out[column] = regime[column]
        return out

    def _regime(self, bars: pd.DataFrame) -> pd.DataFrame:
        """ADX, Bollinger-width percentile and Hurst on the regime timeframe, aligned to base bars.

        A higher-timeframe bar is only used once it has closed: each value is shifted to the
        label of the NEXT bar before being forward-filled, so the bar still forming never leaks
        into a decision, and a live window that ends mid-bar gives the same answer as a backtest.
        """
        cfg = self.cfg
        resampled = cfg.regime_tf_sec > cfg.base_tf_sec
        if resampled:
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            htf = bars[REQUIRED].resample(f"{cfg.regime_tf_sec}s", label="left", closed="left").agg(agg).dropna(subset=["close"])
        else:
            htf = bars[REQUIRED]
        frame = ind.adx(htf["high"], htf["low"], htf["close"], cfg.adx_period)
        frame["bbw_pct"] = ind.percentile_rank(ind.bollinger_width(htf["close"], cfg.bb_period, cfg.bb_k), cfg.bbw_lookback)
        frame["hurst"] = ind.hurst(htf["close"], cfg.hurst_window)
        if resampled:
            frame = frame.shift(1).reindex(bars.index, method="ffill")
        return frame

    # ------------------------------------------------------------------ signals
    def signals(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Applies the thresholds. Cheap, so a parameter search can reuse one indicator frame."""
        cfg = self.cfg
        out = frame.copy()
        everywhere = pd.Series(True, index=out.index)

        # 1. Regime. "Ranging" is the brief's gate; "trending" is its complement on the same measures.
        adx_low = out["adx"] < cfg.adx_max
        hurst_low = out["hurst"] < cfg.hurst_max
        measured = out[["adx", "hurst", "bbw_pct"]].notna().all(axis=1)
        not_trending = {"adx": adx_low, "hurst": hurst_low, "either": adx_low | hurst_low, "both": adx_low & hurst_low, "off": everywhere}[cfg.regime_mode]
        quiet = out["bbw_pct"] <= cfg.bbw_max_pctile if cfg.bbw_max_pctile < 1.0 else everywhere
        out["regime_ok"] = (not_trending & quiet & measured) if cfg.regime_mode != "off" else quiet.fillna(False)
        out["trending"] = (~not_trending & measured) if cfg.regime_mode != "off" else everywhere

        # 2. Stretch: the short-term oscillator and the VWAP band agree.
        ready = out[["rsi", "atr", "ema", "vwap", "z", "support", "resistance"]].notna().all(axis=1)
        if cfg.session_warmup_bars > 0:
            ready = ready & (out["bars_in_session"] > cfg.session_warmup_bars)
        calm_bar = out["bar_atr"] <= cfg.max_bar_atr if cfg.max_bar_atr > 0 else everywhere
        stretched_down = ready & calm_bar & (out["rsi"] < cfg.rsi_low) & (out["z"] < -cfg.entry_z)
        stretched_up = ready & calm_bar & (out["rsi"] > cfg.rsi_high) & (out["z"] > cfg.entry_z)

        # 3. Order flow: was the effort absorbed (fade it) or did it carry the bar (go with it)?
        if cfg.use_absorption:
            effort = out["rel_volume"] >= cfg.absorb_rel_volume
            held_up = out["lower_wick"] >= cfg.absorb_wick_frac
            held_down = out["upper_wick"] >= cfg.absorb_wick_frac
            if {"buy_volume", "sell_volume"}.issubset(out.columns):
                # With true aggressor volume: net selling that failed to close the bar down.
                held_up = held_up | ((out["delta_ratio"] <= -0.15) & (out["close"] >= out["open"]))
                held_down = held_down | ((out["delta_ratio"] >= 0.15) & (out["close"] <= out["open"]))
            absorbed_at_support = out["at_support"] & effort & held_up
            absorbed_at_resistance = out["at_resistance"] & effort & held_down
            carried_down = effort & ~held_up
            carried_up = effort & ~held_down
        else:
            absorbed_at_support = absorbed_at_resistance = carried_down = carried_up = everywhere

        # 4. What reviewing losing continuation trades showed: they cluster where the directional
        #    index barely favours the move, where the signal bar was pushed back (a wick against
        #    it), on volume climaxes, and in the most volatile names and moments.
        lean = out["plus_di"] - out["minus_di"]
        strong_up = lean >= cfg.cont_min_di_gap if cfg.cont_min_di_gap > 0 else everywhere
        strong_down = -lean >= cfg.cont_min_di_gap if cfg.cont_min_di_gap > 0 else everywhere
        clean_up = out["upper_wick"] <= cfg.cont_max_wick if cfg.cont_max_wick < 1.0 else everywhere
        clean_down = out["lower_wick"] <= cfg.cont_max_wick if cfg.cont_max_wick < 1.0 else everywhere
        orderly = everywhere
        if cfg.max_rel_volume > 0:
            orderly = orderly & (out["rel_volume"] <= cfg.max_rel_volume)
        if cfg.max_atr_bps > 0:
            orderly = orderly & (1e4 * out["atr"] / out["close"] <= cfg.max_atr_bps)

        fade_long = stretched_down & absorbed_at_support & out["regime_ok"]
        fade_short = stretched_up & absorbed_at_resistance & out["regime_ok"]
        follow_long = stretched_up & carried_up & out["trending"] & strong_up & clean_up & orderly
        follow_short = stretched_down & carried_down & out["trending"] & strong_down & clean_down & orderly
        if cfg.playbook == "reversion":
            follow_long = follow_short = ~everywhere
        elif cfg.playbook == "continuation":
            fade_long = fade_short = ~everywhere
        elif cfg.playbook != "adaptive":
            raise ValueError(f"unknown playbook {cfg.playbook!r}")

        out["long_signal"] = fade_long | follow_long
        out["short_signal"] = (fade_short | follow_short) if cfg.allow_short else ~everywhere
        out["kind"] = np.where(follow_long | follow_short, CONTINUATION, np.where(fade_long | fade_short, REVERSION, 0))

        # Hard stops for the reversion playbook, fixed when the order is sent: whichever of the
        # two tail conditions is nearer. (Continuation stops are a multiple of ATR from the entry.)
        out["stop_long"] = np.maximum(out["vwap"] - cfg.stop_z * out["vwap_sd"], out["support"] - cfg.stop_atr_mult * out["atr"])
        out["stop_short"] = np.minimum(out["vwap"] + cfg.stop_z * out["vwap_sd"], out["resistance"] + cfg.stop_atr_mult * out["atr"])
        return out

    def features(self, bars: pd.DataFrame) -> pd.DataFrame:
        return self.signals(self.indicators(bars))

    # ------------------------------------------------------------------ decisions
    def mean_for(self, side: int, vwap: float, ema: float, reference: float) -> float | None:
        """The mean a position on `side` is reaching for, per cfg.target.

        With 'nearest' the trade is over as soon as the price touches EITHER mean, so this
        returns None (nothing left to reach) once any candidate is at or behind `reference`.
        """
        candidates = {"vwap": [vwap], "ema": [ema], "nearest": [vwap, ema]}[self.cfg.target]
        if any(m != m or (m - reference) * side <= 0 for m in candidates):
            return None
        return min(candidates) if side == LONG else max(candidates)

    def entry(self, row, bid: float, ask: float) -> EntryIntent | None:
        """The order for a signal bar, or None when the signal has no room to pay for itself.

        `row` is the signal bar's feature row (a Series or dict with the feature names). A
        passive entry joins the bid for a long and the ask for a short, optionally stepped
        back by `entry_offset_atr`, so it earns the spread instead of paying it.
        """
        cfg = self.cfg
        if row["long_signal"]:
            side = LONG
        elif row["short_signal"]:
            side = SHORT
        else:
            return None
        kind = int(row.get("kind", REVERSION) or REVERSION)
        atr = float(row["atr"])
        spread = max(ask - bid, self.tick)
        marketable = cfg.entry_style == "marketable"
        if marketable:
            # Through the far touch by two ticks: trades on arrival, but cannot chase a runaway quote.
            limit = round_tick(ask + 2 * self.tick, self.tick, +1) if side == LONG else round_tick(bid - 2 * self.tick, self.tick, -1)
        elif side == LONG:
            limit = round_tick(bid - cfg.entry_offset_atr * atr, self.tick, -1)
        else:
            limit = round_tick(ask + cfg.entry_offset_atr * atr, self.tick, +1)

        if kind == CONTINUATION:
            target = limit + side * cfg.cont_target_atr * atr
            stop = limit - side * cfg.cont_stop_atr * atr
        else:
            mean = self.mean_for(side, float(row["vwap"]), float(row["ema"]), limit)
            if mean is None:
                return None
            target = limit + cfg.target_frac * (mean - limit)
            stop = float(row["stop_long"] if side == LONG else row["stop_short"])
        target = round_tick(target, self.tick, -1 if side == LONG else +1)
        stop = round_tick(stop, self.tick, -1 if side == LONG else +1)
        # The target has to clear the spread it will be measured against, and the stop has to be
        # far enough away that ordinary quote noise cannot touch it.
        if (target - limit) * side < 2 * spread:
            return None
        if (limit - stop) * side < max(2 * spread, 0.25 * atr) or stop <= 0:
            return None
        return EntryIntent(side, limit, stop, target, atr, float(row["z"]), kind, marketable)

    def manage(self, row, position: Position) -> ExitIntent:
        """Called at each bar close while a position is open. `row` needs close, vwap and ema."""
        cfg = self.cfg
        close = float(row["close"])
        if position.kind == CONTINUATION:
            if position.bars_held >= cfg.cont_max_hold_bars:
                return ExitIntent(position.target_price, exit_now=True, reason="time")
            # A move that has fallen back through its own average is no longer a trend.
            if (close - float(row["ema"])) * position.side < 0:
                return ExitIntent(position.target_price, exit_now=True, reason="trend_over")
            return ExitIntent(position.target_price)

        if position.bars_held >= cfg.max_hold_bars:
            return ExitIntent(position.target_price, exit_now=True, reason="time")
        if not cfg.dynamic_target:
            return ExitIntent(position.target_price)
        mean = self.mean_for(position.side, float(row["vwap"]), float(row["ema"]), close)
        if mean is None:
            # The mean has come to the price (or past it): the reversion this trade was for is over.
            return ExitIntent(position.target_price, exit_now=True, reason="mean")
        target = position.entry_price + cfg.target_frac * (mean - position.entry_price)
        if (target - close) * position.side <= 0:
            return ExitIntent(position.target_price, exit_now=True, reason="mean")
        return ExitIntent(round_tick(target, self.tick, -1 if position.side == LONG else +1))
