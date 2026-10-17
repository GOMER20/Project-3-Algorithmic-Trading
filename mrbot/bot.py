"""AT: the live trading bot.

It listens to the exchange's quote stream, and on every completed one-minute bar it runs the
same RegimeMeanReversion rules and RiskManager the backtest uses, then sends orders.

How it stays safe when things go wrong:

  - The exchange is the source of truth. Once a minute, and whenever one of its orders
    changes, the bot re-reads its positions and working orders and corrects its own picture.
  - Every entry carries a server-side stop-loss (and, when the target is fixed, a server-side
    take-profit). If the bot or the connection dies, the exits are still working.
  - A position the bot does not recognise, or one facing the wrong way, is closed at once.
  - Losing `max_daily_loss` of the session's starting equity cancels everything, flattens,
    and halts until the next session.
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .data import to_frame
from .exchange import ExchangeError, GomerClient
from .metrics import format_stats, summarize
from .risk import RiskManager
from .strategy import CONTINUATION, LONG, EntryIntent, Position, RegimeMeanReversion

BOT_NAME = "AT"
log = logging.getLogger("at.bot")

DEFAULT_SYMBOLS = ["AAPL", "META", "NVDA", "AMD", "TSLA", "PLTR", "COIN", "GME", "SHOP", "MU", "JPM", "SPY"]


@dataclass
class BotConfig:
    base_url: str = "http://127.0.0.1:47831"
    trader_id: str = ""
    password: str = ""
    competition: str | None = None
    """Competition id or name. None picks the active competition this trader is enrolled in."""
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    history_bars: int = 1000
    run_dir: str = "runs"


@dataclass
class _Slot:
    """Everything the bot knows about one symbol."""

    symbol: str
    bars: pd.DataFrame
    minute: int = 0
    bid: float = 0.0
    ask: float = 0.0
    entry_id: str | None = None
    intent: EntryIntent | None = None
    ttl: int = 0
    position: Position | None = None
    target_id: str | None = None
    """The bot's own take-profit limit, when the target follows the mean."""
    exit_ids: set[str] = field(default_factory=set)
    """Every order the bot has sent to reduce this position, working or not (targets included)."""
    exit_reason: str | None = None

    @property
    def idle(self) -> bool:
        return self.entry_id is None and self.position is None


class Journal:
    """The session's record: every closed trade and a mark of equity each minute."""

    FIELDS = ["symbol", "side", "kind", "quantity", "entry_time", "entry_price", "exit_time", "exit_price", "exit_reason",
              "pnl", "gross", "fees", "r_multiple", "notional", "bars_held"]

    def __init__(self, directory: str | Path, name: str = BOT_NAME):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.trades: list[dict] = []
        self.marks: list[tuple[float, float]] = []
        self.starting_equity = 0.0
        self.events: list[str] = []
        trades_file, equity_file = self.directory / "trades.csv", self.directory / "equity.csv"
        if trades_file.exists() and equity_file.exists():
            # The bot was restarted inside a session: carry on from what is already on record.
            saved = pd.read_csv(trades_file, parse_dates=["entry_time", "exit_time"])
            self.trades = saved.to_dict("records")
            self.marks = [(float(t), float(e)) for t, e in pd.read_csv(equity_file).itertuples(index=False)]
            if self.marks:
                self.starting_equity = self.marks[0][1]
            self.events.append(f"Restarted; resuming a session with {len(self.trades)} trades already on record.")
        else:
            with open(trades_file, "w", newline="") as handle:
                csv.writer(handle).writerow(self.FIELDS)
            with open(equity_file, "w", newline="") as handle:
                csv.writer(handle).writerow(["time", "equity"])

    def trade(self, **row) -> None:
        self.trades.append(row)
        with open(self.directory / "trades.csv", "a", newline="") as handle:
            csv.writer(handle).writerow([row[k] for k in self.FIELDS])

    def mark(self, ts: float, equity: float) -> None:
        if not self.marks:
            self.starting_equity = equity
        self.marks.append((ts, equity))
        with open(self.directory / "equity.csv", "a", newline="") as handle:
            csv.writer(handle).writerow([round(ts, 3), round(equity, 2)])

    def note(self, text: str) -> None:
        self.events.append(text)
        log.info(text)

    def stats(self) -> dict:
        trades = pd.DataFrame(self.trades, columns=self.FIELDS)
        equity = None
        if self.marks:
            equity = pd.Series([e for _, e in self.marks], index=pd.to_datetime([t for t, _ in self.marks], unit="s", utc=True))
        out = summarize(trades, equity, self.starting_equity or 1.0)
        if self.marks:
            out["starting_equity"] = self.starting_equity
            out["ending_equity"] = self.marks[-1][1]
            out["session_return"] = self.marks[-1][1] / self.starting_equity - 1.0 if self.starting_equity else 0.0
        return out

    def write_report(self, expected: dict | None = None, title: str | None = None) -> Path:
        """Writes report.md and stats.json. `expected` is the backtest's holdout statistics, for comparison."""
        stats = self.stats()
        (self.directory / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
        lines = [f"# {title or self.name + ' — live session report'}", ""]
        if self.marks:
            start = pd.Timestamp(self.marks[0][0], unit="s", tz="UTC")
            end = pd.Timestamp(self.marks[-1][0], unit="s", tz="UTC")
            lines += [
                f"Session {start:%Y-%m-%d %H:%M} to {end:%H:%M} UTC.",
                f"Equity ${stats['starting_equity']:,.2f} → ${stats['ending_equity']:,.2f} ({100 * stats['session_return']:+.2f}%).",
                "",
            ]
        if stats.get("trades"):
            wins = round(stats["win_rate"] * stats["trades"])
            lines += [
                f"## Success rate: {100 * stats['win_rate']:.1f}%",
                "",
                f"{wins:,} of {stats['trades']:,} trades made money. Net result ${stats['net_pnl']:+,.2f}; "
                f"the average winner made ${stats['avg_win']:,.2f} and the average loser lost ${-stats['avg_loss']:,.2f}.",
                "",
            ]
        else:
            lines += ["## No trades were made this session", ""]
        lines += ["```", format_stats(stats), "```", ""]
        if expected and stats.get("trades"):
            lines += [
                "## Against the backtest",
                "",
                "| | live session | backtest holdout |",
                "|---|---|---|",
                f"| win rate | {100 * stats['win_rate']:.1f}% (95% interval {100 * stats['win_rate_ci_low']:.0f}–{100 * stats['win_rate_ci_high']:.0f}%) | {100 * expected['win_rate']:.1f}% |",
                f"| average trade | {stats['expectancy_bps']:+.2f} bps | {expected['expectancy_bps']:+.2f} bps |",
                f"| payoff ratio | {stats['payoff_ratio']:.2f} | {expected['payoff_ratio']:.2f} |",
                f"| profit factor | {stats['profit_factor']:.2f} | {expected['profit_factor']:.2f} |",
                "",
            ]
        if self.events:
            lines += ["## Events", ""] + [f"- {e}" for e in self.events] + [""]
        path = self.directory / "report.md"
        path.write_text("\n".join(lines))
        return path


class TradingBot:
    def __init__(self, client: GomerClient, strategy_cfg: StrategyConfig, cfg: BotConfig, journal: Journal | None = None):
        self.client = client
        self.cfg = cfg
        self.strategy_cfg = strategy_cfg
        self.strategy = RegimeMeanReversion(strategy_cfg)
        self.risk = RiskManager(strategy_cfg, session_of=lambda ts: int(ts // 86_400))
        self.journal = journal or Journal(Path(cfg.run_dir) / time.strftime("%Y-%m-%d_%H%M%S"))
        if cfg.history_bars < strategy_cfg.warmup_bars:
            # Otherwise the indicators would never be ready and the bot would sit silent all session.
            raise ValueError(f"history_bars={cfg.history_bars} is less than the {strategy_cfg.warmup_bars} bars this strategy needs to warm up")
        self.window = min(cfg.history_bars, strategy_cfg.live_window)
        self.slots: dict[str, _Slot] = {}
        self.competition_id = ""
        self.allow_shorting = True
        self.portfolio: dict = {}
        self.positions: dict[str, dict] = {}
        self.open_orders: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.clock = 0.0
        """Exchange time: the timestamp of the newest quote, in epoch seconds."""
        self.busy = False
        self.stopping = False
        self.orders_sent = 0

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        user = await self.client.login()
        competitions = await self.client.competitions()
        wanted = self.cfg.competition
        matches = [c for c in competitions if c.get("enrolled", True) and (c["id"] == wanted or c["name"] == wanted)] if wanted else [
            c for c in competitions if c["status"] == "ACTIVE" and c.get("enrolled", True)
        ]
        if not matches:
            raise RuntimeError(f"{BOT_NAME} is not enrolled in " + (f"a competition called {wanted!r}" if wanted else "any active competition"))
        competition = matches[0]
        self.competition_id = competition["id"]
        self.allow_shorting = bool(competition.get("allowShorting", True))
        await self._sync()
        histories = await asyncio.gather(*(self.client.bars(s, self.cfg.history_bars) for s in self.cfg.symbols))
        for symbol, rows in zip(self.cfg.symbols, histories):
            # The newest bar is still forming; it is added when its minute ends.
            self.slots[symbol] = _Slot(symbol, to_frame(rows[:-1] if rows else rows))
        self.journal.note(
            f"{BOT_NAME} signed in as {user.get('traderId', self.cfg.trader_id)}, trading {len(self.slots)} symbols in "
            f"'{competition['name']}' with the {self.strategy_cfg.playbook} playbook. Equity ${self.portfolio.get('totalEquity', 0):,.2f}."
        )

    async def run(self, until_clock: float | None = None, until_wall: float | None = None) -> None:
        """Trades until the exchange clock passes `until_clock`, wall time passes `until_wall`, or stop() is requested."""
        stream = self.client.stream(list(self.slots)).__aiter__()
        while not self.stopping:
            if until_wall is not None and time.time() >= until_wall:
                break
            try:
                message = await asyncio.wait_for(stream.__anext__(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except StopAsyncIteration:
                break
            await self.handle(message)
            if until_clock is not None and self.clock >= until_clock:
                break
        await stream.aclose()

    async def shutdown(self, reason: str = "session_end") -> None:
        """Cancels working entries and closes every position at market."""
        self.stopping = True
        await self._flatten_all(reason)
        await self._sync()
        if self.clock:
            self.journal.mark(self.clock, float(self.portfolio.get("totalEquity", 0.0)))
        self.journal.note(f"{BOT_NAME} stopped ({reason}). {len(self.journal.trades)} trades, {self.orders_sent} orders sent.")

    # ------------------------------------------------------------------ messages
    async def handle(self, message: dict) -> None:
        self.busy = True
        try:
            kind = message.get("type")
            if kind == "quotes":
                await self._on_quotes(message["data"])
            elif kind == "quote":
                await self._on_quotes([message["data"]])
            elif kind == "order":
                await self._on_order(message["data"])
            elif kind == "connected":
                await self._sync()
                for slot in self.slots.values():
                    await self._reconcile(slot)
        except ExchangeError as err:
            log.warning("exchange error while handling %s: %s", message.get("type"), err)
        finally:
            self.busy = False

    async def _on_quotes(self, quotes: list[dict]) -> None:
        rolled: list[_Slot] = []
        for quote in quotes:
            slot = self.slots.get(quote["symbol"])
            if slot is None:
                continue
            slot.bid, slot.ask = float(quote["bid"]), float(quote["ask"])
            ts = quote["ts"] / 1000.0
            self.clock = max(self.clock, ts)
            minute = int(ts // 60) * 60
            if slot.minute == 0:
                slot.minute = minute
            elif minute > slot.minute:
                slot.minute = minute
                rolled.append(slot)
        if rolled:
            await self._on_minute(rolled)

    async def _on_minute(self, slots: list[_Slot]) -> None:
        """A minute has ended for these symbols: take in the finished bar, re-sync, decide."""
        fetched = await asyncio.gather(*(self.client.bars(s.symbol, 3) for s in slots), return_exceptions=True)
        fresh: list[_Slot] = []
        for slot, rows in zip(slots, fetched):
            if isinstance(rows, Exception):
                log.warning("no bars for %s this minute: %s", slot.symbol, rows)
                continue
            last = int(slot.bars.index[-1].timestamp()) if len(slot.bars) else 0
            done = [row for row in rows if last < row["time"] < slot.minute]
            if done:
                slot.bars = pd.concat([slot.bars, to_frame(done)]).tail(self.cfg.history_bars)
                fresh.append(slot)

        await self._sync()
        equity = float(self.portfolio.get("totalEquity", 0.0))
        self.journal.mark(self.clock, equity)
        if self.risk.update(self.clock, equity):
            self.journal.note(f"Circuit breaker: session loss reached {100 * self.strategy_cfg.max_daily_loss:.1f}%. Flattening and halting.")
            await self._flatten_all("breaker")
            await self._sync()
        for slot in self.slots.values():
            await self._reconcile(slot)
        for slot in fresh:
            try:
                await self._decide(slot)
            except ExchangeError as err:
                log.warning("%s: %s", slot.symbol, err)

    async def _on_order(self, order: dict) -> None:
        self.orders[order["id"]] = order
        slot = self.slots.get(order.get("ticker", ""))
        if slot is None or order.get("status") == "OPEN" and not order.get("filledQuantity"):
            return
        mine = order["id"] in (slot.entry_id, slot.target_id) or order["id"] in slot.exit_ids or (slot.entry_id and order.get("parentId") == slot.entry_id)
        if mine:
            await self._sync()
            await self._reconcile(slot)

    # ------------------------------------------------------------------ state
    async def _sync(self) -> None:
        """Re-reads the account from the exchange."""
        self.portfolio = await self.client.portfolio(self.competition_id)
        self.positions = {p["ticker"]: p for p in self.portfolio.get("positions", [])}
        orders = await self.client.open_orders(self.competition_id)
        self.open_orders = {o["id"]: o for o in orders}
        self.orders.update(self.open_orders)

    async def _reconcile(self, slot: _Slot) -> None:
        """Brings one symbol's state into line with what the exchange says is held and working."""
        held = self.positions.get(slot.symbol)
        quantity = int(held["quantity"]) if held else 0
        if slot.position is None:
            if slot.entry_id:
                if quantity != 0:
                    await self._opened(slot, quantity, float(held["averagePrice"]))
                elif slot.entry_id not in self.open_orders:
                    order = self.orders.get(slot.entry_id, {})
                    filled = int(order.get("filledQuantity") or 0)
                    if filled > 0:
                        # Filled and already taken out by its own bracket inside the minute.
                        await self._opened(slot, filled * (1 if order["side"] == "BUY" else -1), float(order["fillPrice"]))
                        await self._closed(slot)
                    else:
                        self._clear(slot)
            elif quantity != 0:
                # Not one of ours (left from an earlier run, or a stale order that slipped through).
                # Close it, and journal it so the session's P&L still adds up.
                self.journal.note(f"{slot.symbol}: found a position of {quantity} that {BOT_NAME} did not open; closing it.")
                price = float(held["averagePrice"])
                slot.position = Position(slot.symbol, LONG if quantity > 0 else -LONG, abs(quantity), price, price, price, self.clock)
                await self._exit(slot, "orphan")
        elif quantity == 0:
            await self._closed(slot)
        elif (quantity > 0) != (slot.position.side == LONG):
            self.journal.note(f"{slot.symbol}: position is on the wrong side ({quantity}); closing it.")
            slot.exit_reason = "flip"
            await self._market_out(slot, quantity, "flip")
        else:
            slot.position.quantity = abs(quantity)

    async def _opened(self, slot: _Slot, quantity: int, price: float) -> None:
        intent = slot.intent
        slot.position = Position(
            symbol=slot.symbol, side=LONG if quantity > 0 else -LONG, quantity=abs(quantity), entry_price=price,
            stop_price=intent.stop_price, target_price=intent.target_price, entry_time=self.clock, kind=intent.kind,
            initial_risk=abs(quantity) * abs(price - intent.stop_price),
        )
        if slot.entry_id in self.open_orders:
            # Partly filled: trade what was given rather than leave the rest working beside a position.
            await self.client.cancel_order(slot.entry_id)
        if not self._bracket_target(intent):
            await self._rest_target(slot, intent.target_price)

    async def _closed(self, slot: _Slot) -> None:
        """The position is gone: work out how it ended and journal the trade."""
        position = slot.position
        for order_id in slot.exit_ids & set(self.open_orders):
            # A take-profit of ours still resting would open a new position if it were left behind.
            await self.client.cancel_order(order_id)
        self.orders.update({o["id"]: o for o in await self.client.recent_orders(self.competition_id)})
        entry = self.orders.get(slot.entry_id or "", {})
        exits = [
            o for o in self.orders.values()
            if int(o.get("filledQuantity") or 0) > 0 and o["id"] != slot.entry_id
            and (o["id"] in slot.exit_ids or (slot.entry_id is not None and o.get("parentId") == slot.entry_id))
        ]
        shares = sum(int(o["filledQuantity"]) for o in exits)
        gross = sum(float(o.get("realizedPnl") or 0.0) for o in exits)
        fees = float(entry.get("commission") or 0.0) + sum(float(o.get("commission") or 0.0) for o in exits)
        exit_price = sum(float(o["fillPrice"]) * int(o["filledQuantity"]) for o in exits) / shares if shares else position.entry_price
        reason = slot.exit_reason
        if reason is None:
            kinds = {o["type"] for o in exits}
            reason = "stop" if kinds & {"STOP", "STOP_LIMIT"} else "target" if "LIMIT" in kinds else "closed"
        pnl = gross - fees
        r_multiple = pnl / position.initial_risk if position.initial_risk > 0 else 0.0
        self.risk.record_trade(r_multiple)
        self.journal.trade(
            symbol=slot.symbol, side=position.side, kind=position.kind, quantity=position.quantity,
            entry_time=pd.Timestamp(position.entry_time, unit="s", tz="UTC"), entry_price=position.entry_price,
            exit_time=pd.Timestamp(self.clock, unit="s", tz="UTC"), exit_price=round(exit_price, 4), exit_reason=reason,
            pnl=round(pnl, 4), gross=round(gross, 4), fees=round(fees, 4), r_multiple=r_multiple,
            notional=position.quantity * position.entry_price, bars_held=position.bars_held,
        )
        log.info("%s %s closed by %s: %+.2f", slot.symbol, "long" if position.side == LONG else "short", reason, pnl)
        self._clear(slot)

    @staticmethod
    def _clear(slot: _Slot) -> None:
        slot.entry_id = slot.intent = slot.position = slot.target_id = slot.exit_reason = None
        slot.exit_ids = set()
        slot.ttl = 0

    # ------------------------------------------------------------------ decisions
    def _bracket_target(self, intent: EntryIntent) -> bool:
        """True when the take-profit never moves, so the exchange can hold it as part of the bracket."""
        return intent.kind == CONTINUATION or not self.strategy_cfg.dynamic_target

    async def _decide(self, slot: _Slot) -> None:
        if len(slot.bars) < self.strategy_cfg.warmup_bars:
            return
        row = self.strategy.features(slot.bars.tail(self.window)).iloc[-1]
        if slot.position is not None:
            position = slot.position
            position.bars_held += 1
            decision = self.strategy.manage(row, position)
            touch = slot.bid if position.side == LONG else slot.ask
            if decision.exit_now or (decision.target_price - touch) * position.side <= 0:
                await self._exit(slot, decision.reason or "target_taker")
            elif not self._bracket_target(slot.intent) and decision.target_price != position.target_price:
                await self._rest_target(slot, decision.target_price)
            return
        if slot.entry_id is not None:
            slot.ttl -= 1
            if slot.ttl <= 0:
                cancelled = await self.client.cancel_order(slot.entry_id)
                if cancelled is not None and not int(cancelled.get("filledQuantity") or 0):
                    self._clear(slot)
            return
        if self.stopping or self.risk.halted or not (row["long_signal"] or row["short_signal"]):
            return
        if row["short_signal"] and not self.allow_shorting:
            return
        if sum(not s.idle for s in self.slots.values()) >= self.strategy_cfg.max_open_positions:
            return
        intent = self.strategy.entry(row, slot.bid, slot.ask)
        if intent is None:
            return
        cap = self.strategy_cfg.max_same_side
        if cap > 0 and sum(1 for s in self.slots.values() if not s.idle and s.intent is not None and s.intent.side == intent.side) >= cap:
            return
        quantity = self.risk.size(intent, float(self.portfolio.get("totalEquity", 0.0)), float(self.portfolio.get("buyingPower", 0.0)))
        if quantity <= 0:
            return
        try:
            order = await self.client.place_order(
                self.competition_id, slot.symbol, "BUY" if intent.side == LONG else "SELL", "LIMIT", quantity,
                limit_price=intent.limit_price, stop_loss=intent.stop_price,
                take_profit=intent.target_price if self._bracket_target(intent) else None,
            )
        except ExchangeError as err:
            if err.rejected:
                log.info("%s entry refused: %s", slot.symbol, err.message)
                return
            raise
        self.orders_sent += 1
        self.orders[order["id"]] = order
        slot.entry_id, slot.intent, slot.ttl = order["id"], intent, self.strategy_cfg.entry_ttl_bars
        if int(order.get("filledQuantity") or 0) > 0 or order.get("status") != "OPEN":
            await self._sync()
            await self._reconcile(slot)

    async def _rest_target(self, slot: _Slot, price: float) -> None:
        """Places (or moves) the bot's own take-profit limit for a position whose target follows the mean."""
        if not await self._pull_target(slot):
            return
        position = slot.position
        try:
            order = await self.client.place_order(self.competition_id, slot.symbol, "SELL" if position.side == LONG else "BUY", "LIMIT", position.quantity, limit_price=price)
        except ExchangeError as err:
            if not err.rejected:
                raise
            log.info("%s target refused: %s", slot.symbol, err.message)
            return
        self.orders_sent += 1
        self.orders[order["id"]] = order
        slot.target_id = order["id"]
        slot.exit_ids.add(order["id"])
        position.target_price = price

    async def _pull_target(self, slot: _Slot) -> bool:
        """Cancels the bot's resting take-profit. Returns False if the position turned out to be gone.

        If the cancel finds nothing to cancel, the target has just traded. Sending another exit
        on top of that would open a position the other way, so the account is re-read first.
        """
        if slot.target_id is None:
            return True
        already_gone = await self.client.cancel_order(slot.target_id) is None
        slot.target_id = None
        if already_gone:
            await self._sync()
            await self._reconcile(slot)
        return slot.position is not None

    async def _exit(self, slot: _Slot, reason: str) -> None:
        slot.exit_reason = reason
        if await self._pull_target(slot):
            await self._market_out(slot, slot.position.quantity * slot.position.side, reason)

    async def _market_out(self, slot: _Slot, signed_quantity: int, reason: str) -> None:
        """Closes `signed_quantity` shares at market. The exchange cancels a bracket's exits once the position is gone."""
        try:
            order = await self.client.place_order(self.competition_id, slot.symbol, "SELL" if signed_quantity > 0 else "BUY", "MARKET", abs(signed_quantity))
        except ExchangeError as err:
            log.warning("%s: could not close (%s): %s", slot.symbol, reason, err)
            return
        self.orders_sent += 1
        self.orders[order["id"]] = order
        slot.exit_ids.add(order["id"])
        if slot.position is not None:
            await self._sync()
            await self._reconcile(slot)

    async def _flatten_all(self, reason: str) -> None:
        for slot in self.slots.values():
            try:
                if slot.position is None and slot.entry_id:
                    await self.client.cancel_order(slot.entry_id)
            except ExchangeError as err:
                log.warning("%s: %s", slot.symbol, err)
        await self._sync()
        for slot in self.slots.values():
            try:
                await self._reconcile(slot)
                if slot.position is not None:
                    await self._exit(slot, reason)
            except ExchangeError as err:
                log.warning("%s: %s", slot.symbol, err)

    # ------------------------------------------------------------------ reporting
    def status(self) -> dict:
        return {
            "name": BOT_NAME,
            "playbook": self.strategy_cfg.playbook,
            "competition": self.competition_id,
            "clock": self.clock,
            "halted": self.risk.halted,
            "equity": self.portfolio.get("totalEquity"),
            "orders_sent": self.orders_sent,
            "trades": len(self.journal.trades),
            "positions": [
                {"symbol": s.symbol, "side": s.position.side, "quantity": s.position.quantity, "entry": s.position.entry_price,
                 "stop": s.position.stop_price, "target": s.position.target_price, "bars_held": s.position.bars_held}
                for s in self.slots.values() if s.position
            ],
            "working_entries": [s.symbol for s in self.slots.values() if s.entry_id and not s.position],
        }
