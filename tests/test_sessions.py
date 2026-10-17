"""Market sessions: overnight gaps must not look like signals, and nothing is held over a close."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mrbot.backtest import Backtester
from mrbot.config import StrategyConfig
from mrbot.data import load_dir, random_walk_bars
from mrbot.review import review_session
from mrbot.strategy import RegimeMeanReversion

ROOT = Path(__file__).resolve().parent.parent
CFG = StrategyConfig(playbook="continuation", regime_tf_sec=60, use_absorption=False, entry_z=2.0, rsi_low=5.0, rsi_high=95.0,
                     adx_max=25.0, entry_style="marketable", kelly_fraction=0.0)
PRICES = ["open", "high", "low", "close", "bid_low", "bid_high", "ask_low", "ask_high", "bid_close", "ask_close"]


def two_sessions(gap: float) -> pd.DataFrame:
    """780 one-minute bars as two exchange sessions on consecutive days, the second opening `gap` away from the first's close."""
    bars = random_walk_bars(780, seed=5, sigma_bps=25.0)
    first = pd.date_range("2026-01-05 13:30", periods=390, freq="1min", tz="UTC")
    bars.index = first.append(first + pd.Timedelta(days=1)).rename("time")
    bars.iloc[390:, bars.columns.get_indexer(PRICES)] *= 1.0 + gap
    return bars


def test_an_overnight_gap_changes_levels_but_not_signals():
    strategy = RegimeMeanReversion(CFG)
    smooth, gapped = strategy.features(two_sessions(0.0)), strategy.features(two_sessions(0.04))
    for column in ("rsi", "z", "adx", "hurst", "bbw_pct", "upper_wick", "lower_wick", "bar_atr"):
        np.testing.assert_allclose(gapped[column].to_numpy(), smooth[column].to_numpy(), rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=column)
    for column in ("long_signal", "short_signal", "kind"):
        assert (gapped[column] == smooth[column]).all(), column
    # Price levels in the second session sit 4% higher, as the prices themselves do.
    for column in ("atr", "ema", "vwap", "support", "resistance"):
        np.testing.assert_allclose(gapped[column].iloc[400:].to_numpy(), 1.04 * smooth[column].iloc[400:].to_numpy(), rtol=1e-6, err_msg=column)
    assert list(gapped["bars_in_session"].iloc[[0, 389, 390, 391]]) == [1, 390, 1, 2]


def test_without_the_adjustment_the_gap_would_have_fired():
    """The reason for the adjustment: read naively, a 4% open is the biggest 'move' of the day."""
    bars = two_sessions(0.04)
    close = bars["close"]
    naive_z = (close - close.rolling(30).mean()) / close.rolling(30).std()
    adjusted = RegimeMeanReversion(CFG).features(bars)
    assert naive_z.iloc[390] > 4 and abs(adjusted["z"].iloc[390]) < 4
    assert not adjusted["long_signal"].iloc[390] or adjusted["z"].iloc[390] > CFG.entry_z  # any signal there is the session's own


def test_nothing_is_opened_late_or_held_over_the_close():
    cfg = CFG.with_(regime_mode="off", session_warmup_bars=30)
    bars = two_sessions(0.04)
    result = Backtester(cfg, compound=False).run({"X": RegimeMeanReversion(cfg).features(bars)})
    trades = result.trades
    assert len(trades) > 10
    minute_of_session = lambda ts: int((pd.Timestamp(ts) - pd.Timestamp(ts).normalize() - pd.Timedelta(hours=13, minutes=30)).total_seconds() // 60)  # noqa: E731
    entered = trades["entry_time"].map(minute_of_session)
    assert entered.min() >= 30  # the first half hour of each session is left alone
    assert entered.max() <= 390 - cfg.session_flatten_bars - 1
    assert (pd.to_datetime(trades["entry_time"]).dt.normalize() == pd.to_datetime(trades["exit_time"]).dt.normalize()).all()


def test_review_tells_expected_losses_from_execution_problems():
    path, data = ROOT / "configs/at.json", ROOT / "data/sessions_holdout"
    if not path.exists() or not data.exists():
        pytest.skip("generate history and run scripts/run_research.py first")
    cfg = StrategyConfig(**json.loads(path.read_text())).with_(kelly_fraction=0.0)
    bars = {s: b.iloc[: 390 * 2] for s, b in load_dir(data, ["TSLA", "NVDA", "AAPL", "META", "JPM", "SHOP"]).items()}
    strategy = RegimeMeanReversion(cfg)
    session = bars["TSLA"].index[390]
    # Plain OHLCV, as the app's chart history is: the review can only replay what it can fetch.
    ohlcv = {s: b[["open", "high", "low", "close", "volume"]] for s, b in bars.items()}
    replay = Backtester(cfg, compound=False).run({s: strategy.features(b) for s, b in ohlcv.items()}, start=session).trades
    assert len(replay) > 20 and (replay["pnl"] <= 0).any()

    # A live session that went exactly as the backtest says: every loss is an expected one.
    faithful = review_session(replay, ohlcv, cfg, {"win_rate": 0.86})
    assert faithful["losers_by_kind"]["execution"] == 0 and faithful["losers_by_kind"]["unplanned"] == 0
    assert faithful["losers_by_kind"]["expected"] == int((replay["pnl"] <= 0).sum())
    assert not faithful["update_recommended"] and "No mistake found" in faithful["text"]

    # The same session, but live lost twelve trades the backtest wins (more than one in ten):
    # that is an execution problem, not bad luck.
    spoiled = replay.copy()
    winners = spoiled.index[spoiled["pnl"] > 0][:12]
    spoiled.loc[winners, "pnl"] = -100.0
    flagged = review_session(spoiled, ohlcv, cfg, {"win_rate": 0.86})
    assert flagged["losers_by_kind"]["execution"] == 12 and flagged["update_recommended"]
    assert "Execution needs looking at" in flagged["text"]
