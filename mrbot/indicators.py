"""Technical indicators, vectorised over pandas Series.

Every function is causal: the value at row t uses rows up to and including t and nothing
later. That property is what lets the backtest compute indicators once over a whole history
and still claim no look-ahead, and it is asserted in tests/test_indicators.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (an EMA with alpha = 1/period), as used by RSI, ATR and ADX."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    change = close.diff()
    gain = wilder(change.clip(lower=0.0), period)
    loss = wilder((-change).clip(lower=0.0), period)
    out = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    # No losses in the window is the definition of fully overbought; no movement at all is neutral.
    out = out.where(loss > 0, 100.0).where((gain > 0) | (loss > 0), 50.0)
    return out.where(gain.notna())


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return wilder(true_range(high, low, close), period)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with its +DI / -DI components."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    smoothed_tr = wilder(true_range(high, low, close), period).replace(0.0, np.nan)
    plus_di = 100.0 * wilder(plus_dm, period) / smoothed_tr
    minus_di = 100.0 * wilder(minus_dm, period) / smoothed_tr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return pd.DataFrame({"adx": wilder(dx.fillna(0.0).where(plus_di.notna()), period), "plus_di": plus_di, "minus_di": minus_di})


def bollinger_width(close: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    """Band width as a share of the middle band."""
    mid = close.rolling(period).mean()
    return 2.0 * k * close.rolling(period).std(ddof=0) / mid


def percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
    """Where the current value sits in its own trailing window, in (0, 1]."""
    return series.rolling(lookback, min_periods=max(10, lookback // 2)).rank(pct=True)


def rolling_vwap(price: pd.Series, volume: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Volume-weighted average price over a rolling window, with its volume-weighted standard deviation."""
    weight = volume.clip(lower=0.0) + 1e-9
    total = weight.rolling(window).sum()
    vwap = (price * weight).rolling(window).sum() / total
    variance = (price * price * weight).rolling(window).sum() / total - vwap * vwap
    return vwap, np.sqrt(variance.clip(lower=0.0))


def session_vwap(price: pd.Series, volume: pd.Series, session: pd.Series, min_bars: int = 15) -> tuple[pd.Series, pd.Series]:
    """VWAP anchored at the start of each session, with its deviation. Undefined for the first `min_bars`."""
    weight = volume.clip(lower=0.0) + 1e-9
    total = weight.groupby(session).cumsum()
    vwap = (price * weight).groupby(session).cumsum() / total
    variance = (price * price * weight).groupby(session).cumsum() / total - vwap * vwap
    age = price.groupby(session).cumcount()
    ready = age >= min_bars
    return vwap.where(ready), np.sqrt(variance.clip(lower=0.0)).where(ready)


def hurst(close: pd.Series, window: int = 100, lags: tuple[int, ...] = (1, 2, 4, 8, 16)) -> pd.Series:
    """Rolling Hurst exponent from how the dispersion of returns scales with horizon.

    For a random walk the standard deviation of a tau-bar return grows as tau^0.5. A slope
    below 0.5 means moves are partly given back (mean reversion); above 0.5 means they are
    extended (trend). The slope is fitted by least squares across `lags` on each row.
    """
    log_price = np.log(close)
    x = np.log(np.array(lags, dtype=float))
    x_centred = x - x.mean()
    columns = []
    for lag in lags:
        dispersion = log_price.diff(lag).rolling(window, min_periods=window).std(ddof=0)
        columns.append(np.log(dispersion.replace(0.0, np.nan)))
    y = pd.concat(columns, axis=1).to_numpy()
    slope = ((y - y.mean(axis=1, keepdims=True)) * x_centred).sum(axis=1) / (x_centred**2).sum()
    return pd.Series(slope, index=close.index)


def volume_delta(bars: pd.DataFrame) -> pd.Series:
    """Signed volume: buyer-initiated minus seller-initiated.

    Uses true aggressor-side volume when the feed provides it (`buy_volume` / `sell_volume`).
    Otherwise it falls back to bulk volume classification, which splits each bar's volume by
    where it closed inside its range. The fallback is derived from price, so it carries no
    information beyond the candle itself; it is an approximation, not order flow.
    """
    if {"buy_volume", "sell_volume"}.issubset(bars.columns):
        return bars["buy_volume"] - bars["sell_volume"]
    span = (bars["high"] - bars["low"]).replace(0.0, np.nan)
    location = ((bars["close"] - bars["low"]) / span).fillna(0.5)
    return bars["volume"] * (2.0 * location - 1.0)
