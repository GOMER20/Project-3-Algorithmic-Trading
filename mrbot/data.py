"""Loading bar history, and a null market for checking the backtester against."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def to_frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Bars with a `time` column in epoch seconds (the exchange's format) -> a UTC-indexed frame."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=BAR_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    frame.index = pd.to_datetime(frame.pop("time"), unit="s", utc=True)
    frame.index.name = "time"
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def load_csv(path: str | Path) -> pd.DataFrame:
    return to_frame(pd.read_csv(path))


def load_dir(directory: str | Path, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Every `<SYMBOL>.csv` in a directory, as written by scripts/generate_history.ts."""
    directory = Path(directory)
    files = sorted(directory.glob("*.csv"))
    if symbols:
        files = [f for f in files if f.stem in symbols]
    if not files:
        raise FileNotFoundError(f"No bar files in {directory}. Generate them first (see README).")
    return {f.stem: load_csv(f) for f in files}


def random_walk_bars(
    minutes: int = 20_000,
    seed: int = 0,
    price: float = 150.0,
    sigma_bps: float = 12.0,
    spread_bps: float = 1.6,
    ticks_per_bar: int = 20,
    start: str = "2026-01-05",
) -> pd.DataFrame:
    """One-minute bars from a driftless random walk with a bid/ask spread.

    This market has nothing to predict: every move is permanent. A strategy run on it should
    make nothing before costs and lose its costs after them. If a backtest shows a profit
    here, the backtest is wrong (look-ahead, or fills that are too generous).
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, sigma_bps / 1e4 / np.sqrt(ticks_per_bar), size=minutes * ticks_per_bar)
    mid = price * np.exp(np.cumsum(steps))
    spread = np.maximum(0.01, np.round(mid * spread_bps / 1e4, 2))
    bid = np.round(mid - spread / 2, 2)
    ask = np.round(bid + spread, 2)
    at_ask = rng.random(mid.size) >= 0.5
    last = np.where(at_ask, ask, bid)
    size = rng.integers(40, 800, size=mid.size)

    shape = (minutes, ticks_per_bar)
    last, bid, ask, size, at_ask = (a.reshape(shape) for a in (last, bid, ask, size, at_ask))
    close = last[:, -1]
    open_ = np.concatenate([[last[0, 0]], close[:-1]])
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(last.max(axis=1), open_),
            "low": np.minimum(last.min(axis=1), open_),
            "close": close,
            "volume": size.sum(axis=1),
            "bid_low": bid.min(axis=1),
            "bid_high": bid.max(axis=1),
            "ask_low": ask.min(axis=1),
            "ask_high": ask.max(axis=1),
            "bid_close": bid[:, -1],
            "ask_close": ask[:, -1],
            "buy_volume": (size * at_ask).sum(axis=1),
            "sell_volume": (size * ~at_ask).sum(axis=1),
        },
        index=pd.date_range(start, periods=minutes, freq="1min", tz="UTC", name="time"),
    )
    return frame
