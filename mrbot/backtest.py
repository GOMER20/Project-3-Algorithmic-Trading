"""Event-driven backtester with the exchange's fill rules, fees and slippage.

Time moves one base bar at a time across all symbols together, so equity, position limits
and the daily-loss breaker are shared the way they are live. Within a step:

    A. Orders already working are matched against the NEW bar:
         - the hard stop first (if a bar could have hit both the stop and the target, the
           stop is assumed to have come first)
         - then the resting take-profit limit
         - then resting entry limits; an entry filled in a bar can also be stopped in that
           same bar, but cannot take profit in it
    B. At the bar's close the strategy decides, using only that bar and earlier:
         - mark to liquidation value, update the circuit breaker
         - re-price or close open positions
         - place new entry limits for signal bars

Nothing decided at a close can fill before the next bar, and resting limits fill only when
the OPPOSITE quote reaches them (the exchange's rule), which is stricter than "the bar's low
touched my price". Takers pay the configured slippage and commission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import GOMER_COSTS, CostModel, StrategyConfig
from .metrics import summarize
from .risk import RiskManager
from .strategy import LONG, EntryIntent, Position, RegimeMeanReversion

FEATURE_COLUMNS = ["open", "high", "low", "close", "atr", "vwap", "ema", "z", "stop_long", "stop_short"]
QUOTE_COLUMNS = ["bid_low", "bid_high", "ask_low", "ask_high", "bid_close", "ask_close"]
TRADE_COLUMNS = [
    "symbol", "side", "kind", "quantity", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
    "pnl", "gross", "fees", "r_multiple", "notional", "bars_held", "exit_maker", "z",
]


@dataclass
class _Pending:
    intent: EntryIntent
    quantity: int
    ttl: int


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    stats: dict
    cfg: StrategyConfig
    costs: CostModel
    halts: list = field(default_factory=list)
    orders_placed: int = 0
    orders_filled: int = 0


def _instant(value, tz) -> pd.Timestamp:
    """A window boundary in the index's own timezone; a bare date means midnight UTC."""
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo else ts
    return ts.tz_localize("UTC").tz_convert(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def with_quotes(frame: pd.DataFrame, costs: CostModel) -> pd.DataFrame:
    """Adds bid/ask columns to plain OHLCV bars, assuming a fixed spread.

    The assumption is the unfavourable one: the bar's low printed on the bid and its high on
    the ask, so a resting buy needs the price to fall a full spread below it to fill.
    """
    if set(QUOTE_COLUMNS).issubset(frame.columns):
        return frame
    out = frame.copy()
    spread = np.maximum(costs.tick_size, out["close"] * costs.fallback_spread_bps / 1e4)
    out["bid_close"] = out["close"] - spread / 2
    out["ask_close"] = out["close"] + spread / 2
    out["bid_low"] = out["low"]
    out["ask_low"] = out["low"] + spread
    out["ask_high"] = out["high"]
    out["bid_high"] = out["high"] - spread
    return out


class Backtester:
    def __init__(
        self,
        cfg: StrategyConfig,
        costs: CostModel = GOMER_COSTS,
        starting_equity: float = 100_000.0,
        fixed_risk: bool = False,
        compound: bool = True,
    ):
        """`fixed_risk` turns the Kelly overlay off. `compound=False` sizes every trade from the
        starting equity instead of current equity, which keeps results comparable across time."""
        self.compound = compound
        self.cfg = cfg if not fixed_risk else cfg.with_(kelly_fraction=0.0)
        self.costs = costs
        self.starting_equity = starting_equity
        self.strategy = RegimeMeanReversion(self.cfg, costs.tick_size)

    def run(self, features: dict[str, pd.DataFrame], start=None, end=None) -> BacktestResult:
        """`features` maps symbol to the output of RegimeMeanReversion.features() over its bars."""
        cfg, costs, strategy = self.cfg, self.costs, self.strategy
        index = None
        for frame in features.values():
            index = frame.index if index is None else index.union(frame.index)
        lo = 0 if start is None else int(index.searchsorted(_instant(start, index.tz)))
        hi = len(index) if end is None else int(index.searchsorted(_instant(end, index.tz)))
        lo = max(lo, 1)

        arrays: dict[str, dict[str, np.ndarray]] = {}
        signals_at: dict[int, list[str]] = {}
        for symbol, frame in features.items():
            frame = with_quotes(frame, costs).reindex(index)
            data = {c: frame[c].to_numpy(dtype=float) for c in FEATURE_COLUMNS + QUOTE_COLUMNS}
            data["long_signal"] = frame["long_signal"].fillna(False).to_numpy(dtype=bool)
            data["short_signal"] = frame["short_signal"].fillna(False).to_numpy(dtype=bool)
            data["kind"] = frame["kind"].fillna(0).to_numpy(dtype=int) if "kind" in frame else np.ones(len(frame), dtype=int)
            arrays[symbol] = data
            for i in np.flatnonzero((data["long_signal"] | data["short_signal"])[lo:hi]) + lo:
                signals_at.setdefault(int(i), []).append(symbol)

        sessions = index.floor("D").asi8  # one id per UTC day, whatever the index resolution
        # Bars left before trading stops (the close, or a break in the data). Exchange hours are
        # known in advance, so using them here is a timetable, not a look ahead.
        gap = np.zeros(len(index), dtype=bool)
        if len(index) > 1:
            gap[1:] = np.asarray((index[1:] - index[:-1]).total_seconds()) > 2 * cfg.base_tf_sec
        run = np.cumsum(gap)
        sizes = np.bincount(run)
        bars_left = sizes[run] - (np.arange(len(index)) - (np.cumsum(sizes) - sizes)[run]) - 1
        risk = RiskManager(cfg, session_of=lambda s: s)
        cash = self.starting_equity
        positions: dict[str, Position] = {}
        pending: dict[str, _Pending] = {}
        entry_fees: dict[str, float] = {}
        entry_z: dict[str, float] = {}
        trades: list[tuple] = []
        equity = np.empty(hi - lo)
        placed = filled = 0
        through = costs.maker_through_ticks * costs.tick_size

        def fee(quantity: int, price: float, maker: bool) -> float:
            bps = costs.maker_fee_bps if maker else costs.taker_fee_bps
            return quantity * costs.commission_per_share + quantity * price * bps / 1e4

        def close(symbol: str, price: float, reason: str, i: int, maker: bool) -> None:
            nonlocal cash
            position = positions.pop(symbol)
            exit_fee = fee(position.quantity, price, maker)
            cash += position.side * position.quantity * price - exit_fee
            gross = position.side * position.quantity * (price - position.entry_price)
            fees = entry_fees.pop(symbol) + exit_fee
            pnl = gross - fees
            r_multiple = pnl / position.initial_risk if position.initial_risk > 0 else 0.0
            risk.record_trade(r_multiple)
            trades.append((
                symbol, position.side, position.kind, position.quantity, position.entry_time, position.entry_price, index[i], price, reason,
                pnl, gross, fees, r_multiple, position.quantity * position.entry_price, position.bars_held, maker, entry_z.pop(symbol),
            ))

        def open_position(symbol: str, intent: EntryIntent, quantity: int, price: float, i: int, maker: bool) -> Position:
            nonlocal cash, filled
            filled += 1
            paid = fee(quantity, price, maker)
            cash -= intent.side * quantity * price + paid
            entry_fees[symbol] = paid
            entry_z[symbol] = intent.z
            positions[symbol] = Position(
                symbol=symbol, side=intent.side, quantity=quantity, entry_price=price, stop_price=intent.stop_price,
                target_price=intent.target_price, entry_time=index[i], kind=intent.kind,
                initial_risk=quantity * abs(price - intent.stop_price),
            )
            return positions[symbol]

        def taker_exit_price(position: Position, data: dict, i: int) -> float:
            touch = data["bid_close"][i] if position.side == LONG else data["ask_close"][i]
            return touch - position.side * costs.taker_slippage

        def stop_price(position: Position, touch_at_open: float) -> float:
            # A stop becomes a market order. If the market opened beyond it, it trades there.
            level = min(position.stop_price, touch_at_open) if position.side == LONG else max(position.stop_price, touch_at_open)
            return level - position.side * (costs.taker_slippage + level * costs.stop_slippage_bps / 1e4)

        for i in range(lo, hi):
            # ---- A. match working orders against bar i ---------------------------------
            if positions:
                for symbol in list(positions):
                    data = arrays[symbol]
                    if data["close"][i] != data["close"][i]:
                        continue  # no bar for this symbol at this time
                    position = positions[symbol]
                    if position.side == LONG:
                        if data["bid_low"][i] <= position.stop_price:
                            close(symbol, stop_price(position, data["bid_close"][i - 1]), "stop", i, False)
                        elif data["bid_high"][i] >= position.target_price + through:
                            close(symbol, position.target_price, "target", i, True)
                    else:
                        if data["ask_high"][i] >= position.stop_price:
                            close(symbol, stop_price(position, data["ask_close"][i - 1]), "stop", i, False)
                        elif data["ask_low"][i] <= position.target_price - through:
                            close(symbol, position.target_price, "target", i, True)

            if pending:
                for symbol in list(pending):
                    data = arrays[symbol]
                    if data["close"][i] != data["close"][i]:
                        continue
                    order = pending[symbol]
                    intent = order.intent
                    reached = data["ask_low"][i] <= intent.limit_price - through if intent.side == LONG else data["bid_high"][i] >= intent.limit_price + through
                    if not reached:
                        order.ttl -= 1
                        if order.ttl <= 0:
                            del pending[symbol]
                        continue
                    del pending[symbol]
                    position = open_position(symbol, intent, order.quantity, intent.limit_price, i, True)
                    # The same bar that filled the entry may have gone on to the stop.
                    stopped = data["bid_low"][i] <= position.stop_price if intent.side == LONG else data["ask_high"][i] >= position.stop_price
                    if stopped:
                        close(symbol, stop_price(position, position.stop_price), "stop", i, False)

            # ---- B. decisions at the close of bar i ------------------------------------
            value = cash
            for symbol, position in positions.items():
                data = arrays[symbol]
                mark = data["bid_close"][i] if position.side == LONG else data["ask_close"][i]
                if mark != mark:
                    mark = position.entry_price
                value += position.side * position.quantity * mark
            equity[i - lo] = value

            if risk.update(sessions[i], value):
                pending.clear()
                for symbol in list(positions):
                    close(symbol, taker_exit_price(positions[symbol], arrays[symbol], i), "breaker", i, False)
                equity[i - lo] = cash

            closing = bars_left[i] < cfg.session_flatten_bars
            if closing:
                pending.clear()
                for symbol in list(positions):
                    close(symbol, taker_exit_price(positions[symbol], arrays[symbol], i), "session_end", i, False)
                equity[i - lo] = cash

            if positions:
                for symbol in list(positions):
                    data = arrays[symbol]
                    if data["close"][i] != data["close"][i]:
                        continue
                    position = positions[symbol]
                    position.bars_held += 1
                    decision = strategy.manage({"close": data["close"][i], "vwap": data["vwap"][i], "ema": data["ema"][i]}, position)
                    touch = data["bid_close"][i] if position.side == LONG else data["ask_close"][i]
                    if decision.exit_now or (decision.target_price - touch) * position.side <= 0:
                        # Not exit_now: the price closed beyond the resting target without the quote reaching it.
                        close(symbol, taker_exit_price(position, data, i), decision.reason or "target_taker", i, False)
                    else:
                        position.target_price = decision.target_price

            for symbol in () if closing else signals_at.get(i, ()):
                if risk.halted or symbol in positions or symbol in pending:
                    continue
                if len(positions) + len(pending) >= cfg.max_open_positions:
                    continue
                data = arrays[symbol]
                if cfg.max_same_side > 0:
                    side = LONG if data["long_signal"][i] else -LONG
                    if sum(p.side == side for p in positions.values()) + sum(o.intent.side == side for o in pending.values()) >= cfg.max_same_side:
                        continue
                row = {name: data[name][i] for name in ("long_signal", "short_signal", "kind", "atr", "vwap", "ema", "z", "stop_long", "stop_short")}
                intent = strategy.entry(row, data["bid_close"][i], data["ask_close"][i])
                if intent is None:
                    continue
                committed = sum(p.quantity * p.entry_price for p in positions.values()) + sum(o.quantity * o.intent.limit_price for o in pending.values())
                basis = equity[i - lo] if self.compound else min(self.starting_equity, equity[i - lo])
                quantity = risk.size(intent, basis, buying_power=max(0.0, basis - committed))
                if quantity <= 0:
                    continue
                placed += 1
                if intent.marketable:
                    # Crosses the spread on arrival: a taker fill at the touch plus slippage, capped by the limit.
                    touch = data["ask_close"][i] if intent.side == LONG else data["bid_close"][i]
                    price = touch + intent.side * costs.taker_slippage
                    price = min(price, intent.limit_price) if intent.side == LONG else max(price, intent.limit_price)
                    open_position(symbol, intent, quantity, price, i, False)
                else:
                    pending[symbol] = _Pending(intent, quantity, cfg.entry_ttl_bars)

        last = hi - 1
        for symbol in list(positions):
            if lo <= last:
                close(symbol, taker_exit_price(positions[symbol], arrays[symbol], last), "session_end", last, False)
        if hi > lo:
            equity[-1] = cash

        trade_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        equity_series = pd.Series(equity, index=index[lo:hi])
        stats = summarize(trade_frame, equity_series, self.starting_equity)
        stats["orders_placed"] = placed
        stats["fill_rate"] = filled / placed if placed else float("nan")
        stats["breaker_halts"] = len(risk.halts)
        return BacktestResult(trade_frame, equity_series, stats, cfg, costs, list(risk.halts), placed, filled)


class FeatureStore:
    """Holds raw bars and builds strategy features on demand, caching the expensive part.

    Indicators depend only on periods and windows, so a parameter search that varies
    thresholds reuses one indicator frame per symbol and pays only for the boolean signals.
    Indicators are causal, so computing them once over the whole history and then slicing by
    date leaks nothing into an earlier window.
    """

    def __init__(self, bars: dict[str, pd.DataFrame]):
        self.bars = bars
        self._indicators: dict[tuple, dict[str, pd.DataFrame]] = {}

    def features(self, cfg: StrategyConfig) -> dict[str, pd.DataFrame]:
        strategy = RegimeMeanReversion(cfg)
        key = cfg.indicator_key()
        if key not in self._indicators:
            self._indicators[key] = {symbol: strategy.indicators(frame) for symbol, frame in self.bars.items()}
        return {symbol: strategy.signals(frame) for symbol, frame in self._indicators[key].items()}

    def run(self, cfg: StrategyConfig, costs: CostModel = GOMER_COSTS, start=None, end=None, **kwargs) -> BacktestResult:
        return Backtester(cfg, costs, **kwargs).run(self.features(cfg), start, end)
