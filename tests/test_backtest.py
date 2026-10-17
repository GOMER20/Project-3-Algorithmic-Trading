"""The backtester's fill rules, costs and safety checks."""
import numpy as np
import pandas as pd

from mrbot.backtest import Backtester
from mrbot.config import GOMER_COSTS, CostModel, StrategyConfig
from mrbot.data import random_walk_bars
from mrbot.risk import RiskManager, kelly_risk_fraction
from mrbot.strategy import LONG, EntryIntent, RegimeMeanReversion


def flat_market(n=60, price=100.0):
    index = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")
    frame = pd.DataFrame({"open": price, "high": price + 0.01, "low": price - 0.01, "close": price, "volume": 1000.0}, index=index)
    frame["bid_close"], frame["ask_close"] = price - 0.01, price + 0.01
    frame["bid_low"], frame["bid_high"] = price - 0.01, price - 0.01
    frame["ask_low"], frame["ask_high"] = price + 0.01, price + 0.01
    for column, value in (("atr", 0.2), ("vwap", price + 1.0), ("ema", price + 1.0), ("z", -3.0), ("stop_long", price - 1.0), ("stop_short", price + 1.0)):
        frame[column] = value
    frame["long_signal"], frame["short_signal"], frame["kind"] = False, False, 0
    return frame


CFG = StrategyConfig(dynamic_target=False, target_frac=0.5, kelly_fraction=0.0, max_hold_bars=1000)


def run(frame, costs=GOMER_COSTS, cfg=CFG):
    return Backtester(cfg, costs, 100_000.0).run({"X": frame})


def signal_at(frame, i):
    frame.iloc[i, frame.columns.get_loc("long_signal")] = True
    frame.iloc[i, frame.columns.get_loc("kind")] = 1
    return frame


def test_resting_buy_needs_the_ask_to_reach_it():
    frame = signal_at(flat_market(), 10)  # limit rests at the bid, 99.99; the ask never gets there
    result = run(frame)
    assert result.orders_placed == 1 and result.orders_filled == 0 and len(result.trades) == 0

    frame = signal_at(flat_market(), 10)
    frame.iloc[11, frame.columns.get_loc("ask_low")] = 99.99  # the ask touches it in the next bar
    frame.iloc[11, frame.columns.get_loc("bid_low")] = 99.97
    result = run(frame)
    assert result.orders_filled == 1
    assert result.trades.iloc[0]["entry_price"] == 99.99  # a maker fills at its own price, no slippage


def test_a_signal_cannot_fill_on_its_own_bar():
    frame = signal_at(flat_market(), 10)
    frame.iloc[10, frame.columns.get_loc("ask_low")] = 99.0  # the signal bar itself traded far below the limit
    assert run(frame).orders_filled == 0


def test_stop_is_assumed_before_target_and_pays_slippage():
    frame = signal_at(flat_market(), 10)
    cols = {c: frame.columns.get_loc(c) for c in frame.columns}
    frame.iloc[11, cols["ask_low"]] = 99.99
    frame.iloc[11, cols["bid_low"]] = 99.97
    frame.iloc[12, cols["bid_low"]] = 98.9  # through the stop at 99.00 ...
    frame.iloc[12, cols["bid_high"]] = 101.0  # ... and through the target, in the same bar
    trade = run(frame).trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] < 99.0 - GOMER_COSTS.taker_slippage + 1e-9
    assert trade["pnl"] < 0


def test_target_fills_as_maker_and_fees_are_charged():
    frame = signal_at(flat_market(), 10)
    cols = {c: frame.columns.get_loc(c) for c in frame.columns}
    frame.iloc[11, cols["ask_low"]] = 99.99
    frame.iloc[11, cols["bid_low"]] = 99.97
    frame.iloc[13, cols["bid_high"]] = 100.6
    free = run(frame).trades.iloc[0]
    assert free["exit_reason"] == "target" and free["exit_price"] == 100.49  # halfway from 99.99 to the mean at 101, rounded down
    paid = run(frame, CostModel(commission_per_share=0.01)).trades.iloc[0]
    assert np.isclose(paid["fees"], 2 * 0.01 * paid["quantity"])
    assert np.isclose(free["pnl"] - paid["pnl"], paid["fees"])


def test_circuit_breaker_flattens_and_halts_for_the_day():
    frame = flat_market(300)
    cols = {c: frame.columns.get_loc(c) for c in frame.columns}
    frame["stop_long"] = 50.0  # a stop too far away to help: only the breaker can end this
    for i in (10, 100, 200):
        signal_at(frame, i)
    frame.iloc[11, cols["ask_low"]] = 99.99
    for i in range(12, 300):  # then the market falls 20% and stays there
        for column in ("open", "high", "low", "close", "bid_close", "ask_close", "bid_low", "bid_high", "ask_low", "ask_high"):
            frame.iloc[i, cols[column]] *= 0.8
    cfg = CFG.with_(risk_per_trade=0.5, risk_cap=0.5, max_notional_frac=0.5)
    result = run(frame, cfg=cfg)
    assert list(result.trades["exit_reason"]) == ["breaker"]
    assert result.stats["breaker_halts"] == 1
    assert result.orders_placed == 1  # the later signals on the same day were refused


def test_random_walk_has_no_edge():
    """Nothing to predict means nothing to earn. A profit here would mean the backtest is leaking."""
    store = {f"RW{i}": random_walk_bars(15_000, seed=40 + i, sigma_bps=22.0) for i in range(4)}
    for playbook in ("reversion", "continuation"):
        cfg = StrategyConfig(playbook=playbook, regime_mode="off", bbw_max_pctile=1.0, use_absorption=False, entry_z=2.0, kelly_fraction=0.0)
        strategy = RegimeMeanReversion(cfg)
        result = Backtester(cfg, GOMER_COSTS, 100_000.0, compound=False).run({s: strategy.features(b) for s, b in store.items()})
        assert result.stats["trades"] > 200
        assert result.stats["t_stat"] < 2.0, f"{playbook} shows an edge on a random walk"


def test_sizing_and_kelly():
    cfg = StrategyConfig(risk_per_trade=0.01, max_notional_frac=1.0, kelly_min_trades=10, kelly_fraction=0.5, risk_cap=0.05)
    risk = RiskManager(cfg, session_of=lambda s: s)
    intent = EntryIntent(LONG, 100.0, 99.0, 101.0, 0.5, -3.0)
    assert risk.size(intent, 100_000.0) == 1000  # 1% of equity over $1 of risk a share
    assert risk.size(intent, 100_000.0, buying_power=50_000.0) == 490  # capped by buying power
    assert kelly_risk_fraction([1.0] * 6 + [-1.0] * 4) == 0.6 / 1 - 0.4 / 1
    for r in [1.0] * 6 + [-1.0] * 4:
        risk.record_trade(r)
    assert np.isclose(risk.risk_fraction(), 0.05)  # half of Kelly's 20% is far above the cap
    for _ in range(150):
        risk.record_trade(-1.0)
    risk.record_trade(0.1)
    assert risk.risk_fraction() == cfg.risk_floor  # no edge in the sample: smallest size, not a bigger bet
