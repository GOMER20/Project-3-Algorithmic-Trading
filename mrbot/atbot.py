"""AT, connected to Gomer Trading through the bot gateway.

Same rules as the backtest (RegimeMeanReversion + RiskManager), driven by the gateway's
streams: best bid/offer for prices, execution reports for what happened to AT's orders.

How a trade runs:

  entry    a limit order; a marketable one is sent immediate-or-cancel, so it never rests
  stop     as soon as any of the entry fills, a STOP order for exactly that many shares
           rests at the exchange. It is there whether or not AT is still running.
  target   AT watches the quote and, when it reaches the target, sends an
           immediate-or-cancel limit at the target price. (The gateway has no bracket
           orders, and a resting take-profit beside a resting stop could sell twice.)
  other    a time stop, "the trend is over", the daily-loss breaker and the end of the
           session all close at market, retrying until the position is flat.

Once a minute AT compares its own picture with the exchange's account. A position it does
not recognise is closed; an order it does not recognise is cancelled.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .bot import BOT_NAME, DEFAULT_SYMBOLS, Journal
from .config import StrategyConfig
from .data import to_frame
from .gateway import DesktopHistory, GatewayClient, GatewayError
from .risk import RiskManager
from .strategy import LONG, EntryIntent, Position, RegimeMeanReversion

log = logging.getLogger("at.bot")


@dataclass
class _Order:
    ref: str
    symbol: str
    role: str
    """'entry', 'stop' or 'exit'."""
    qty: int
    reason: str = ""
    cum: int = 0
    value: float = 0.0
    """Dollars traded so far (average price times shares)."""
    commission: float = 0.0
    done: bool = False


@dataclass
class _Slot:
    symbol: str
    bars: pd.DataFrame
    minute: int = 0
    bid: float = 0.0
    ask: float = 0.0
    intent: EntryIntent | None = None
    entry: _Order | None = None
    stop: _Order | None = None
    orders: dict[str, _Order] = field(default_factory=dict)
    position: Position | None = None
    ttl: int = 0
    must_exit: str | None = None
    """Set once AT has decided to get out at market; exits are re-sent until the position is flat."""
    busy: bool = False
    """An exit order is in flight; do not send another until it reports back."""
    mismatches: int = 0

    @property
    def idle(self) -> bool:
        return self.entry is None and self.position is None

    @property
    def remaining(self) -> int:
        """Shares of the current trade still held."""
        if self.entry is None:
            return 0
        return self.entry.cum - sum(o.cum for o in self.orders.values() if o.role != "entry")


class GatewayBot:
    def __init__(self, client: GatewayClient, history: DesktopHistory, strategy_cfg: StrategyConfig,
                 symbols: list[str] | None = None, journal: Journal | None = None, history_bars: int = 1000, run_dir: str = "runs",
                 tag: str = ""):
        """`tag` goes into every order reference. Copies of AT trading the same account in different
        competitions need different tags: references must be unique across the whole account."""
        self.client, self.history = client, history
        self.strategy_cfg = strategy_cfg
        self.strategy = RegimeMeanReversion(strategy_cfg)
        self.risk = RiskManager(strategy_cfg, session_of=lambda ts: int(ts // 86_400))
        self.journal = journal or Journal(Path(run_dir) / time.strftime("%Y-%m-%d_%H%M"))
        self.symbols = symbols or list(DEFAULT_SYMBOLS)
        self.history_bars = history_bars
        if history_bars < strategy_cfg.warmup_bars:
            raise ValueError(f"history_bars={history_bars} is less than the {strategy_cfg.warmup_bars} bars this strategy needs to warm up")
        self.window = min(history_bars, strategy_cfg.live_window)
        self.slots: dict[str, _Slot] = {}
        self.orders: dict[str, _Order] = {}
        self.account: dict = {}
        self.allow_short = strategy_cfg.allow_short
        self.clock = 0.0
        self.stopping = False
        self.orders_sent = 0
        self.rejects = 0
        self._account_minute = 0
        self.tag = tag
        self._run_id = format(int(time.time()) % 0xFFFFFF, "x")
        self._counter = itertools.count(1)

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        session = await self.client.connect()
        self.competition = session["competition"]
        self.account = await self.client.account()
        self.allow_short = self.allow_short and bool(self.account.get("allowShorting", True))
        histories = await asyncio.gather(*(self.history.bars(s, self.history_bars) for s in self.symbols))
        for symbol, rows in zip(self.symbols, histories):
            # The newest bar is still forming; it is taken in when its minute ends.
            self.slots[symbol] = _Slot(symbol, to_frame(rows[:-1] if rows else rows))
        await self._clean_slate()
        shortest = min((len(slot.bars) for slot in self.slots.values()), default=0)
        if shortest < self.strategy_cfg.warmup_bars:
            # The app keeps bars only for the hours it was open, so after a quiet spell there may be little history.
            self.journal.note(
                f"The app has {shortest} minutes of price history and the indicators need {self.strategy_cfg.warmup_bars}. "
                f"{BOT_NAME} will watch for about {self.strategy_cfg.warmup_bars - shortest} minutes before its first trade."
            )
        ack = await self.client.subscribe(self.symbols)
        for symbol in ack.get("unknown", []):
            self.slots.pop(symbol, None)
        self.journal.note(
            f"{BOT_NAME} logged on as {session['traderId']} in '{self.competition['name']}', trading {len(self.slots)} symbols with the "
            f"{self.strategy_cfg.playbook} playbook. Equity ${float(self.account.get('totalEquity', 0)):,.2f}."
        )

    async def _clean_slate(self) -> None:
        """Before trading: nothing working and nothing held from an earlier run."""
        for order in await self.client.open_orders():
            await self.client.cancel(order_id=order["orderId"])
        self.account = await self.client.account()
        for held in self.account.get("positions", []):
            slot = self.slots.get(held["ticker"])
            if slot is not None and int(held["quantity"]) != 0:
                self._adopt(slot, int(held["quantity"]), float(held["averagePrice"]), "left from an earlier run")

    async def run(self, until_wall: float | None = None) -> None:
        """Trades until wall time passes `until_wall` or `stopping` is set."""
        while not self.stopping and (until_wall is None or time.time() < until_wall):
            await self._pump(5.0)

    async def _pump(self, timeout: float) -> None:
        try:
            event = await asyncio.wait_for(self.client.events.get(), timeout)
        except asyncio.TimeoutError:
            return
        try:
            await self.handle(event)
        except GatewayError as err:
            log.warning("gateway refused %s: %s", event.get("op"), err)

    async def shutdown(self, reason: str = "session_end") -> None:
        """Cancels working entries, closes every position at market, and waits until flat."""
        self.stopping = True
        for slot in self.slots.values():
            if slot.position is not None:
                slot.must_exit = slot.must_exit or reason
            elif slot.entry is not None and not slot.entry.done:
                await self._safely(self.client.cancel(slot.entry.ref))
        deadline = time.time() + 30
        while time.time() < deadline and any(not s.idle for s in self.slots.values()) and self.client.connected:
            await self._pump(1.0)
        if self.client.connected:
            for order in await self._safely(self.client.open_orders()) or []:
                await self._safely(self.client.cancel(order_id=order["orderId"]))
            self.account = await self._safely(self.client.account()) or self.account
        if self.clock:
            self.journal.mark(self.clock, float(self.account.get("totalEquity", 0.0)))
        left = [s.symbol for s in self.slots.values() if s.position is not None]
        self.journal.note(
            f"{BOT_NAME} stopped ({reason}). {len(self.journal.trades)} trades, {self.orders_sent} orders sent, {self.rejects} refused."
            + (f" STILL HOLDING {', '.join(left)}: their stop orders remain at the exchange." if left else "")
        )

    @staticmethod
    async def _safely(awaitable):
        try:
            return await awaitable
        except GatewayError as err:
            log.warning("%s", err)
            return None

    # ------------------------------------------------------------------ events
    async def handle(self, event: dict) -> None:
        op = event.get("op")
        if op == "bbo":
            await self._on_bbo(event)
        elif op == "exec":
            await self._on_exec(event)
        elif op == "disconnected" and not self.stopping:
            await self._reconnect()

    async def _reconnect(self) -> None:
        delay = 1.0
        while not self.stopping:
            try:
                await self.client.connect()
                await self.client.subscribe(list(self.slots))
                # Execution reports sent while we were away are gone; ask where every live order stands.
                for order in [o for o in self.orders.values() if not o.done]:
                    try:
                        await self._on_exec(await self.client.request("order.status", clOrdId=order.ref))
                    except GatewayError as err:
                        if err.code == "UNKNOWN_ORDER":
                            order.done = True
                self.journal.note("Reconnected to the gateway.")
                return
            except (GatewayError, OSError) as err:
                log.warning("gateway not reachable (%s); retrying in %.0fs", err, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)

    async def _on_bbo(self, quote: dict) -> None:
        slot = self.slots.get(quote["s"])
        if slot is None:
            return
        slot.bid, slot.ask = float(quote["b"]), float(quote["a"])
        ts = quote["t"] / 1000.0
        self.clock = max(self.clock, ts)
        minute = int(ts // 60) * 60
        if slot.minute == 0:
            slot.minute = minute
        elif minute > slot.minute:
            slot.minute = minute
            await self._on_minute(slot)
        position = slot.position
        if position is None or slot.busy or slot.remaining <= 0:
            return
        if slot.must_exit:
            await self._send_exit(slot, None, slot.must_exit)
        elif ((slot.bid if position.side == LONG else slot.ask) - position.target_price) * position.side >= 0:
            await self._send_exit(slot, position.target_price, "target")

    async def _on_minute(self, slot: _Slot) -> None:
        if self._account_minute != slot.minute:
            self._account_minute = slot.minute
            await self._refresh_account()
        try:
            rows = await self.history.bars(slot.symbol, 3)
        except Exception as err:
            log.warning("no bars for %s this minute: %s", slot.symbol, err)
            return
        last = int(slot.bars.index[-1].timestamp()) if len(slot.bars) else 0
        done = [row for row in rows if last < row["time"] < slot.minute]
        if done:
            slot.bars = pd.concat([slot.bars, to_frame(done)]).tail(self.history_bars)
            await self._decide(slot)

    async def _refresh_account(self) -> None:
        """Once a minute: equity for sizing and the breaker, and a check of AT's picture against the exchange's."""
        self.account = await self.client.account()
        equity = float(self.account.get("totalEquity", 0.0))
        self.journal.mark(self.clock, equity)
        if self.risk.update(self.clock, equity):
            self.journal.note(f"Circuit breaker: the session's loss reached {100 * self.strategy_cfg.max_daily_loss:.1f}%. Flattening and halting.")
            for slot in self.slots.values():
                if slot.position is not None:
                    slot.must_exit = "breaker"
                elif slot.entry is not None and not slot.entry.done:
                    await self._safely(self.client.cancel(slot.entry.ref))
        held = {p["ticker"]: p for p in self.account.get("positions", [])}
        for slot in self.slots.values():
            theirs = int(held[slot.symbol]["quantity"]) if slot.symbol in held else 0
            mine = slot.remaining * (slot.position.side if slot.position else 0)
            if theirs == mine or slot.busy or (slot.entry is not None and not slot.entry.done):
                slot.mismatches = 0
                continue
            # The account can be a moment ahead of the execution reports still queued for us,
            # so only a difference that is still there a minute later is acted on.
            slot.mismatches += 1
            if slot.mismatches < 2:
                continue
            slot.mismatches = 0
            if theirs == 0:
                self.journal.note(f"{slot.symbol}: the exchange shows no position where {BOT_NAME} expected {mine}; closing its record.")
                await self._finish(slot, "reconciled")
            elif slot.position is None:
                self._adopt(slot, theirs, float(held[slot.symbol]["averagePrice"]), f"which {BOT_NAME} did not open")
            else:
                # Close what AT knows about; whatever is left over is picked up as an unrecognised position.
                self.journal.note(f"{slot.symbol}: holding {theirs}, expected {mine}; closing at market.")
                slot.must_exit = "mismatch"

    def _adopt(self, slot: _Slot, quantity: int, price: float, why: str) -> None:
        """Takes on a position AT has no record of, in order to close it and account for it."""
        self.journal.note(f"{slot.symbol}: found a position of {quantity} {why}; closing it.")
        entry = self._order(slot, "entry", abs(quantity))
        entry.cum, entry.value, entry.done = abs(quantity), abs(quantity) * price, True
        self.orders.pop(entry.ref, None)  # there is no order at the exchange behind this record
        slot.entry = entry
        slot.position = Position(slot.symbol, LONG if quantity > 0 else -LONG, abs(quantity), price, price, price, self.clock)
        slot.must_exit = "orphan"

    # ------------------------------------------------------------------ orders
    def _order(self, slot: _Slot, role: str, qty: int, reason: str = "") -> _Order:
        order = _Order(f"AT{self.tag}-{self._run_id}-{next(self._counter)}", slot.symbol, role, qty, reason)
        self.orders[order.ref] = slot.orders[order.ref] = order
        return order

    async def _send(self, order: _Order, side: str, type: str, **kwargs) -> bool:
        self.orders_sent += 1
        try:
            await self.client.new_order(order.ref, order.symbol, side, type, order.qty, **kwargs)
            return True
        except GatewayError as err:
            order.done = True
            self.orders.pop(order.ref, None)
            self.rejects += 1
            log.info("%s %s refused: %s", order.symbol, order.role, err)
            if order.role == "entry" and "short" in err.text.lower():
                self.allow_short = False
            if err.code == "SANDBOX_CLOSED" and not self.risk.halted:
                self.risk.halted = True
                self.journal.note(f"'{self.competition['name']}' has ended or not started; {BOT_NAME} stopped opening trades in it.")
            return False

    async def _on_exec(self, report: dict) -> None:
        order = self.orders.get(report.get("clOrdId") or "")
        if order is None:
            return
        if report.get("avgPx") is not None:
            order.cum = int(report["cumQty"])
            order.value = float(report["avgPx"]) * order.cum
        order.commission = float(report.get("commission") or 0.0)
        order.done = report["ordStatus"] in ("FILLED", "CANCELED", "REJECTED")
        if order.done:
            self.orders.pop(order.ref, None)
        slot = self.slots[order.symbol]
        if slot.orders.get(order.ref) is not order:
            return  # belongs to a trade that has already been closed and journalled
        if order.role == "entry":
            await self._entry_progress(slot, order)
        else:
            if order.role == "exit" and order.done:
                slot.busy = False
            if slot.entry.done and slot.remaining <= 0:
                await self._finish(slot, order.reason or "stop")
            elif order.done and slot.remaining > 0:
                await self._protect(slot)  # part of the position is still there: the stop must match it

    async def _entry_progress(self, slot: _Slot, order: _Order) -> None:
        if order.cum > 0:
            intent, price = slot.intent, order.value / order.cum
            if slot.position is None:
                slot.position = Position(slot.symbol, intent.side, order.cum, price, intent.stop_price, intent.target_price, self.clock, kind=intent.kind)
            slot.position.quantity, slot.position.entry_price = order.cum, price
            slot.position.initial_risk = order.cum * abs(price - intent.stop_price)
            await self._protect(slot)
        elif order.done:
            self._clear(slot)

    async def _protect(self, slot: _Slot) -> None:
        """Makes sure a STOP order for exactly the shares held is resting at the exchange."""
        position, want, stop = slot.position, slot.remaining, slot.stop
        if position is None or want <= 0:
            return
        if stop is not None and not stop.done:
            if stop.qty - stop.cum == want:
                return
            await self._safely(self.client.cancel(stop.ref))
        slot.stop = self._order(slot, "stop", want, "stop")
        if not await self._send(slot.stop, "SELL" if position.side == LONG else "BUY", "STOP", stop_price=position.stop_price):
            # No stop means no position: get out now rather than sit unprotected.
            slot.stop = None
            slot.must_exit = slot.must_exit or "unprotected"

    async def _send_exit(self, slot: _Slot, limit: float | None, reason: str) -> None:
        """An immediate-or-cancel order for everything still held: at `limit`, or at market."""
        position, remaining = slot.position, slot.remaining
        if position is None or remaining <= 0 or slot.busy:
            return
        if not self.allow_short and slot.stop is not None and not slot.stop.done:
            # Where shorting is off, the resting stop already commits these shares; free them first.
            await self._safely(self.client.cancel(slot.stop.ref))
        order = self._order(slot, "exit", remaining, reason)
        slot.busy = True
        side = "SELL" if position.side == LONG else "BUY"
        ok = await self._send(order, side, "LIMIT" if limit is not None else "MARKET", tif="IOC", **({"price": limit} if limit is not None else {}))
        if not ok:
            slot.busy = False
            await self._protect(slot)

    async def _finish(self, slot: _Slot, reason: str) -> None:
        """The trade is over: withdraw its stop, journal it, free the slot."""
        position, entry = slot.position, slot.entry
        if slot.stop is not None and not slot.stop.done:
            await self._safely(self.client.cancel(slot.stop.ref))
        if position is not None and entry is not None and entry.cum > 0:
            outs = [o for o in slot.orders.values() if o.role != "entry" and o.cum > 0]
            shares = sum(o.cum for o in outs)
            exit_value = sum(o.value for o in outs)
            if shares < entry.cum:
                # Closed without our orders seeing all of it (reconciled): value the rest at the touch.
                exit_value += (entry.cum - shares) * (slot.bid if position.side == LONG else slot.ask)
            gross = position.side * (exit_value - entry.value)
            fees = sum(o.commission for o in slot.orders.values())
            pnl = gross - fees
            r_multiple = pnl / position.initial_risk if position.initial_risk > 0 else 0.0
            if reason not in ("orphan", "reconciled"):
                self.risk.record_trade(r_multiple)
            self.journal.trade(
                symbol=slot.symbol, side=position.side, kind=position.kind, quantity=entry.cum,
                entry_time=pd.Timestamp(position.entry_time, unit="s", tz="UTC"), entry_price=round(entry.value / entry.cum, 4),
                exit_time=pd.Timestamp(self.clock, unit="s", tz="UTC"), exit_price=round(exit_value / entry.cum, 4), exit_reason=reason,
                pnl=round(pnl, 4), gross=round(gross, 4), fees=round(fees, 4), r_multiple=r_multiple,
                notional=entry.value, bars_held=position.bars_held,
            )
            log.info("%s %s closed by %s: %+.2f", slot.symbol, "long" if position.side == LONG else "short", reason, pnl)
        self._clear(slot)

    @staticmethod
    def _clear(slot: _Slot) -> None:
        slot.intent = slot.entry = slot.stop = slot.position = slot.must_exit = None
        slot.orders = {}
        slot.ttl = 0
        slot.busy = False

    # ------------------------------------------------------------------ decisions
    async def _decide(self, slot: _Slot) -> None:
        if len(slot.bars) < self.strategy_cfg.warmup_bars:
            return
        row = self.strategy.features(slot.bars.tail(self.window)).iloc[-1]
        position = slot.position
        if position is not None:
            position.bars_held += 1
            decision = self.strategy.manage(row, position)
            if decision.exit_now:
                slot.must_exit = slot.must_exit or decision.reason
                await self._send_exit(slot, None, slot.must_exit)
            else:
                position.target_price = decision.target_price
            return
        if slot.entry is not None:
            slot.ttl -= 1
            if slot.ttl <= 0 and not slot.entry.done:
                await self._safely(self.client.cancel(slot.entry.ref))
            return
        if self.stopping or self.risk.halted or not (row["long_signal"] or row["short_signal"]):
            return
        if row["short_signal"] and not self.allow_short:
            return
        if sum(not s.idle for s in self.slots.values()) >= self.strategy_cfg.max_open_positions:
            return
        intent = self.strategy.entry(row, slot.bid, slot.ask)
        if intent is None:
            return
        cap = self.strategy_cfg.max_same_side
        if cap > 0 and sum(1 for s in self.slots.values() if not s.idle and s.intent is not None and s.intent.side == intent.side) >= cap:
            return
        quantity = self.risk.size(intent, float(self.account.get("totalEquity", 0.0)), float(self.account.get("buyingPower", 0.0)))
        if quantity <= 0:
            return
        slot.intent, slot.ttl = intent, self.strategy_cfg.entry_ttl_bars
        slot.entry = self._order(slot, "entry", quantity)
        if not await self._send(slot.entry, "BUY" if intent.side == LONG else "SELL", "LIMIT", price=intent.limit_price, tif="IOC" if intent.marketable else "GTC"):
            self._clear(slot)

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict:
        return {
            "name": BOT_NAME,
            "playbook": self.strategy_cfg.playbook,
            "competition": getattr(self, "competition", {}).get("name"),
            "connected": self.client.connected,
            "clock": self.clock,
            "halted": self.risk.halted,
            "equity": self.account.get("totalEquity"),
            "orders_sent": self.orders_sent,
            "orders_refused": self.rejects,
            "trades": len(self.journal.trades),
            "positions": [
                {"symbol": s.symbol, "side": s.position.side, "quantity": s.remaining, "entry": s.position.entry_price,
                 "stop": s.position.stop_price, "target": s.position.target_price, "bars_held": s.position.bars_held}
                for s in self.slots.values() if s.position
            ],
        }
