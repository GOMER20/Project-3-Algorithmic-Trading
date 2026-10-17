"""AT's order handling on the bot gateway, driven by a scripted exchange.

The gateway bot is also exercised against the real app (see README); these tests pin down
the parts that must never go wrong: a position always has a stop that matches it, nothing
is left working after a trade, and the journal's P&L is the fills' P&L.
"""
import asyncio

import pandas as pd
import pytest

from mrbot.atbot import GatewayBot, _Slot
from mrbot.bot import Journal
from mrbot.config import StrategyConfig
from mrbot.data import random_walk_bars
from mrbot.gateway import GatewayError
from mrbot.strategy import CONTINUATION, LONG, EntryIntent, ExitIntent

CFG = StrategyConfig(playbook="continuation", regime_tf_sec=60, kelly_fraction=0.0, risk_per_trade=0.004, max_notional_frac=0.25)
INTENT = EntryIntent(LONG, 100.04, 99.04, 100.54, 0.5, 2.5, CONTINUATION, True)


class FakeGateway:
    """Records what AT sends. The test plays the exchange by feeding execution reports back."""

    def __init__(self):
        self.sent: list[dict] = []
        self.cancelled: list[str] = []
        self.positions: list[dict] = []
        self.refuse: set[str] = set()
        self.connected = True
        self.events: asyncio.Queue = asyncio.Queue()

    async def new_order(self, ref, symbol, side, type, qty, price=None, stop_price=None, tif="GTC"):
        if type in self.refuse:
            raise GatewayError("RISK", f"{type} refused for the test")
        self.sent.append({"ref": ref, "symbol": symbol, "side": side, "type": type, "qty": qty, "price": price, "stop": stop_price, "tif": tif})
        return {"clOrdId": ref}

    async def cancel(self, cl_ord_id=None, order_id=None):
        self.cancelled.append(cl_ord_id or order_id)
        return True

    async def account(self):
        return {"totalEquity": 100_000.0, "buyingPower": 100_000.0, "allowShorting": True, "positions": self.positions}

    async def open_orders(self):
        return []


class FakeHistory:
    async def bars(self, symbol, limit):
        return []


def report(order: dict, cum: int, px: float | None, status: str) -> dict:
    return {"op": "exec", "clOrdId": order["ref"], "cumQty": cum, "avgPx": px, "ordStatus": status, "commission": 0.0}


@pytest.fixture
def rig(tmp_path):
    gateway = FakeGateway()
    bot = GatewayBot(gateway, FakeHistory(), CFG, ["X"], Journal(tmp_path))
    bars = random_walk_bars(400, seed=1)[["open", "high", "low", "close", "volume"]]
    slot = bot.slots["X"] = _Slot("X", bars, bid=100.00, ask=100.02)
    bot.account = {"totalEquity": 100_000.0, "buyingPower": 100_000.0}
    bot.clock = bars.index[-1].timestamp() + 60
    # The strategy's arithmetic is tested elsewhere; here it says "go long" and names its levels.
    bot.strategy.features = lambda _bars: pd.DataFrame([{"long_signal": True, "short_signal": False}])
    bot.strategy.entry = lambda _row, _bid, _ask: INTENT
    bot.strategy.manage = lambda _row, position: ExitIntent(position.target_price)
    return bot, gateway, slot


def run(coroutine):
    return asyncio.run(coroutine)


async def open_long(bot, gateway, slot, fill=100.03):
    await bot._decide(slot)
    entry = gateway.sent[-1]
    assert (entry["side"], entry["type"], entry["tif"], entry["price"]) == ("BUY", "LIMIT", "IOC", 100.04)
    assert entry["qty"] == 249  # a quarter of equity at the limit price is the binding limit, not the 0.4% risk
    await bot.handle(report(entry, entry["qty"], fill, "FILLED"))
    return entry


def test_fill_then_stop_then_target(rig):
    bot, gateway, slot = rig

    async def scenario():
        entry = await open_long(bot, gateway, slot)
        stop = gateway.sent[-1]
        assert (stop["side"], stop["type"], stop["qty"], stop["stop"], stop["tif"]) == ("SELL", "STOP", entry["qty"], 99.04, "GTC")
        await bot.handle({"op": "bbo", "s": "X", "b": 100.40, "a": 100.42, "t": bot.clock * 1000 + 500})
        assert gateway.sent[-1] is stop  # not at the target yet: nothing sent
        await bot.handle({"op": "bbo", "s": "X", "b": 100.54, "a": 100.56, "t": bot.clock * 1000 + 500})
        target = gateway.sent[-1]
        assert (target["side"], target["type"], target["tif"], target["price"], target["qty"]) == ("SELL", "LIMIT", "IOC", 100.54, entry["qty"])
        await bot.handle({"op": "bbo", "s": "X", "b": 100.55, "a": 100.57, "t": bot.clock * 1000 + 500})
        assert gateway.sent[-1] is target  # one exit in flight at a time
        await bot.handle(report(target, entry["qty"], 100.54, "FILLED"))
        return entry, stop

    entry, stop = run(scenario())
    assert slot.idle and stop["ref"] in gateway.cancelled  # the stop is withdrawn, never left behind
    trade = bot.journal.trades[0]
    assert trade["exit_reason"] == "target" and trade["pnl"] == pytest.approx(entry["qty"] * (100.54 - 100.03))


def test_stop_out(rig):
    bot, gateway, slot = rig

    async def scenario():
        entry = await open_long(bot, gateway, slot)
        await bot.handle(report(gateway.sent[-1], entry["qty"], 99.03, "FILLED"))
        return entry

    entry = run(scenario())
    trade = bot.journal.trades[0]
    assert slot.idle and trade["exit_reason"] == "stop" and trade["pnl"] == pytest.approx(entry["qty"] * (99.03 - 100.03))
    assert trade["r_multiple"] == pytest.approx(-1.0, abs=0.02)


def test_partial_exit_resizes_the_stop(rig):
    bot, gateway, slot = rig

    async def scenario():
        entry = await open_long(bot, gateway, slot)
        first_stop = gateway.sent[-1]
        await bot.handle({"op": "bbo", "s": "X", "b": 100.54, "a": 100.56, "t": bot.clock * 1000 + 500})
        target = gateway.sent[-1]
        await bot.handle(report(target, 100, 100.54, "PARTIALLY_FILLED"))
        await bot.handle(report(target, 100, 100.54, "CANCELED"))  # immediate-or-cancel: the rest is cancelled
        return entry, first_stop

    entry, first_stop = run(scenario())
    new_stop = gateway.sent[-1]
    assert first_stop["ref"] in gateway.cancelled
    assert (new_stop["type"], new_stop["qty"]) == ("STOP", entry["qty"] - 100)  # the stop always matches what is held
    assert slot.remaining == entry["qty"] - 100 and not slot.busy and not bot.journal.trades


def test_entry_that_never_fills_frees_the_slot(rig):
    bot, gateway, slot = rig

    async def scenario():
        await bot._decide(slot)
        await bot.handle(report(gateway.sent[-1], 0, None, "CANCELED"))

    run(scenario())
    assert slot.idle and len(gateway.sent) == 1 and not bot.journal.trades


def test_time_stop_closes_at_market_until_flat(rig):
    bot, gateway, slot = rig

    async def scenario():
        entry = await open_long(bot, gateway, slot)
        bot.strategy.manage = lambda _row, position: ExitIntent(position.target_price, exit_now=True, reason="time")
        await bot._decide(slot)
        first = gateway.sent[-1]
        assert (first["type"], first["tif"], first["qty"]) == ("MARKET", "IOC", entry["qty"])
        await bot.handle(report(first, 200, 100.10, "PARTIALLY_FILLED"))
        await bot.handle(report(first, 200, 100.10, "CANCELED"))
        await bot.handle({"op": "bbo", "s": "X", "b": 100.08, "a": 100.10, "t": bot.clock * 1000 + 500})
        second = gateway.sent[-1]
        assert (second["type"], second["qty"]) == ("MARKET", entry["qty"] - 200)  # tries again for what is left
        await bot.handle(report(second, entry["qty"] - 200, 100.07, "FILLED"))
        return entry

    entry = run(scenario())
    trade = bot.journal.trades[0]
    assert slot.idle and trade["exit_reason"] == "time"
    assert trade["pnl"] == pytest.approx(200 * 100.10 + (entry["qty"] - 200) * 100.07 - entry["qty"] * 100.03)


def test_no_stop_means_no_position(rig):
    bot, gateway, slot = rig
    gateway.refuse.add("STOP")

    async def scenario():
        entry = await open_long(bot, gateway, slot)
        await bot.handle({"op": "bbo", "s": "X", "b": 100.01, "a": 100.03, "t": bot.clock * 1000 + 500})
        return entry

    entry = run(scenario())
    out = gateway.sent[-1]
    assert slot.must_exit == "unprotected" and (out["side"], out["type"], out["qty"]) == ("SELL", "MARKET", entry["qty"])


def test_unrecognised_position_is_closed_and_journalled(rig):
    bot, gateway, slot = rig
    gateway.positions = [{"ticker": "X", "quantity": -50, "averagePrice": 101.0}]

    async def scenario():
        await bot._refresh_account()
        assert slot.idle  # one look is not enough: the account can be ahead of our execution reports
        await bot._refresh_account()
        assert slot.must_exit == "orphan"
        await bot.handle({"op": "bbo", "s": "X", "b": 100.00, "a": 100.02, "t": bot.clock * 1000 + 500})
        cover = gateway.sent[-1]
        assert (cover["side"], cover["type"], cover["qty"]) == ("BUY", "MARKET", 50)
        await bot.handle(report(cover, 50, 100.03, "FILLED"))

    run(scenario())
    trade = bot.journal.trades[0]
    assert slot.idle and trade["exit_reason"] == "orphan" and trade["pnl"] == pytest.approx(50 * (101.0 - 100.03))


def test_two_copies_on_one_account_keep_to_their_own_orders(tmp_path):
    """AT trades its sandbox and the Demo Trading Challenge as one account. The exchange sends
    every execution report to every session of the account, so each copy must ignore the other's."""
    sandbox, demo = FakeGateway(), FakeGateway()
    bars = random_walk_bars(400, seed=1)[["open", "high", "low", "close", "volume"]]
    bots = []
    for tag, gateway, folder in (("", sandbox, "a"), ("D", demo, "b")):
        bot = GatewayBot(gateway, FakeHistory(), CFG, ["X"], Journal(tmp_path / folder), tag=tag)
        bot.slots["X"] = _Slot("X", bars, bid=100.00, ask=100.02)
        bot.account = {"totalEquity": 100_000.0, "buyingPower": 100_000.0}
        bot.strategy.features = lambda _bars: pd.DataFrame([{"long_signal": True, "short_signal": False}])
        bot.strategy.entry = lambda _row, _bid, _ask: INTENT
        bots.append(bot)

    async def scenario():
        for bot in bots:
            await bot._decide(bot.slots["X"])
        mine, theirs = sandbox.sent[-1], demo.sent[-1]
        assert mine["ref"] != theirs["ref"] and theirs["ref"].startswith("ATD-")
        # The sandbox copy hears about the Demo copy's fill, and must not act on it.
        await bots[0].handle(report(theirs, theirs["qty"], 100.03, "FILLED"))
        assert bots[0].slots["X"].position is None and len(sandbox.sent) == 1
        await bots[1].handle(report(theirs, theirs["qty"], 100.03, "FILLED"))
        assert bots[1].slots["X"].position is not None and demo.sent[-1]["type"] == "STOP"

    run(scenario())
