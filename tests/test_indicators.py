"""Indicators and signals must not see the future, and a live window must agree with a full history."""
import numpy as np
import pandas as pd
import pytest

from mrbot import indicators as ind
from mrbot.config import StrategyConfig
from mrbot.data import random_walk_bars
from mrbot.strategy import RegimeMeanReversion

BARS = random_walk_bars(3000, seed=3)
CONFIGS = [
    StrategyConfig(),
    StrategyConfig(playbook="continuation", regime_tf_sec=60, use_absorption=False, entry_z=2.0, rsi_low=5.0, rsi_high=95.0, adx_max=25.0),
    StrategyConfig(playbook="adaptive", vwap_window=0),
]
SIGNAL_COLUMNS = ["rsi", "atr", "ema", "vwap", "z", "adx", "bbw_pct", "hurst", "long_signal", "short_signal", "kind", "stop_long", "stop_short"]


def test_rsi_bounds_and_direction():
    rising = pd.Series(np.arange(1.0, 60.0))
    assert ind.rsi(rising, 2).iloc[-1] == 100.0
    assert ind.rsi(rising[::-1].reset_index(drop=True), 2).iloc[-1] < 1.0
    values = ind.rsi(BARS["close"], 2).dropna()
    assert values.between(0, 100).all()


def test_hurst_tells_trend_from_reversion():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=6000)
    trending = pd.Series(100 + np.cumsum(pd.Series(noise).rolling(8, min_periods=1).mean()))  # positively autocorrelated steps
    reverting = pd.Series(100 + np.cumsum(np.diff(noise, prepend=0.0)) * 0.5)  # every step is half undone
    walk = pd.Series(100 + np.cumsum(noise) * 0.1)
    h = lambda s: ind.hurst(s, 500).dropna().median()  # noqa: E731
    assert h(trending) > 0.6 > 0.4 > h(reverting)
    assert 0.4 < h(walk) < 0.6


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.playbook)
def test_no_lookahead(cfg):
    """Changing the future must not change anything already computed."""
    strategy = RegimeMeanReversion(cfg)
    full = strategy.features(BARS)
    cut = 2200
    tampered = BARS.copy()
    tampered.iloc[cut:, tampered.columns.get_indexer(["open", "high", "low", "close"])] *= 1.5
    tampered.iloc[cut:, tampered.columns.get_loc("volume")] *= 7
    again = strategy.features(tampered)
    pd.testing.assert_frame_equal(full[SIGNAL_COLUMNS].iloc[:cut], again[SIGNAL_COLUMNS].iloc[:cut])


@pytest.mark.parametrize("cfg", CONFIGS[:2], ids=lambda c: c.playbook)
def test_live_window_matches_backtest(cfg):
    """The bot computes on a trailing window. Its newest row must be the backtest's row."""
    strategy = RegimeMeanReversion(cfg)
    full = strategy.features(BARS)
    window = cfg.live_window
    for end in (1500, 1873, 2404, 2999):
        live = strategy.features(BARS.iloc[end - window : end + 1]).iloc[-1]
        back = full.iloc[end]
        assert bool(live["long_signal"]) == bool(back["long_signal"]) and bool(live["short_signal"]) == bool(back["short_signal"])
        for column in ("rsi", "atr", "vwap", "z", "ema", "adx"):
            assert live[column] == pytest.approx(back[column], rel=2e-3, abs=2e-3), column
