"""End to end: the real bot, over real HTTP and WebSocket, against the stand-in exchange."""
import asyncio
import json
from pathlib import Path

import pytest

from mrbot.backtest import Backtester
from mrbot.bot import BotConfig, Journal, TradingBot
from mrbot.config import StrategyConfig
from mrbot.data import load_dir
from mrbot.exchange import GomerClient
from mrbot.mock_exchange import MockExchange, serve
from mrbot.strategy import RegimeMeanReversion

ROOT = Path(__file__).resolve().parent.parent
SYMBOLS = ["TSLA", "NVDA", "AAPL", "COIN", "META", "JPM"]
# One whole exchange session as history, then most of the next one played live.
WARMUP, MINUTES = 390, 385
DATA = ROOT / "data/sessions_holdout"


async def settle(bot: TradingBot, ts_ms: int) -> None:
    """Waits until the bot has finished with everything the exchange has sent."""
    for _ in range(5000):
        if bot.clock >= ts_ms / 1000 and not bot.busy:
            await asyncio.sleep(0.004)
            if not bot.busy:
                return
        await asyncio.sleep(0.001)
    raise AssertionError("the bot never caught up with the exchange")


async def session(cfg: StrategyConfig, tmp_path, port: int, allow_shorting: bool = True):
    bars = {s: f.iloc[: WARMUP + MINUTES] for s, f in load_dir(DATA, SYMBOLS).items()}
    exchange = MockExchange(bars, warmup=WARMUP, allow_shorting=allow_shorting)
    server, task = await serve(exchange, port)
    client = GomerClient(f"http://127.0.0.1:{port}", exchange.trader_id, exchange.password)
    bot = TradingBot(client, cfg, BotConfig(symbols=SYMBOLS, history_bars=WARMUP), Journal(tmp_path / "run"))
    try:
        await bot.start()
        runner = asyncio.create_task(bot.run())
        await asyncio.sleep(0.1)
        for _ in range(MINUTES):
            await exchange.advance(after_tick=lambda ts: settle(bot, ts))
        await bot.shutdown()
        runner.cancel()
    finally:
        await client.close()
        server.should_exit = True
        await task
    return exchange, bot, bars


def chosen() -> StrategyConfig:
    path = ROOT / "configs/at.json"
    if not path.exists() or not DATA.exists():
        pytest.skip("run scripts/generate_history.ts and scripts/run_research.py first")
    return StrategyConfig(**json.loads(path.read_text()))


def check_books(exchange: MockExchange, bot: TradingBot) -> None:
    assert not exchange.positions, "the bot left a position open"
    assert not [o for o in exchange.orders.values() if o["status"] == "OPEN"], "the bot left an order working"
    journal_pnl = sum(t["pnl"] for t in bot.journal.trades)
    # What the bot says it made is what the exchange says it made, to the cent.
    assert journal_pnl == pytest.approx(exchange.realized, abs=0.01)
    assert exchange.cash - exchange.starting_cash == pytest.approx(exchange.realized, abs=0.01)


def test_continuation_session(tmp_path):
    cfg = chosen()
    exchange, bot, bars = asyncio.run(session(cfg, tmp_path, 47931))
    check_books(exchange, bot)
    stats = bot.journal.stats()
    assert stats["trades"] >= 12

    # The backtester over the same bars should tell the same story (not the same trades: the
    # mock plays four quotes a bar, the backtest uses each bar's extremes).
    strategy = RegimeMeanReversion(cfg)
    expected = Backtester(cfg, starting_equity=100_000.0, compound=False).run({s: strategy.features(b) for s, b in bars.items()}, start=bars[SYMBOLS[0]].index[WARMUP]).stats
    print(f"\nlive {stats['trades']} trades, win {stats['win_rate']:.1%}, {stats['expectancy_bps']:+.1f} bps | backtest {expected['trades']} trades, win {expected['win_rate']:.1%}, {expected['expectancy_bps']:+.1f} bps")
    # Per-trade profit is NOT compared: with four quotes a bar the mock's stops trade at the bar's
    # extreme, far beyond the stop price, where the real exchange checks twice a second.
    assert abs(stats["trades"] - expected["trades"]) <= max(3, 0.15 * expected["trades"])
    assert abs(stats["win_rate"] - expected["win_rate"]) < 0.08
    report = bot.journal.write_report(expected).read_text()
    assert "win rate" in report and "Against the backtest" in report


def test_reversion_session_moves_its_own_target(tmp_path):
    """The brief's playbook: the take-profit follows the mean, so the bot manages that order itself."""
    chosen()
    cfg = StrategyConfig(regime_tf_sec=60, regime_mode="off", bbw_max_pctile=1.0, use_absorption=False, entry_z=1.5, rsi_low=20.0, rsi_high=80.0, kelly_fraction=0.0)
    exchange, bot, _ = asyncio.run(session(cfg, tmp_path, 47932))
    check_books(exchange, bot)
    assert bot.journal.stats()["trades"] >= 5
    assert bot.orders_sent > 2 * len(bot.journal.trades)  # entries, exits, and targets being moved
    # "breaker" belongs here: fading this market loses steadily, and the daily-loss limit is meant to notice.
    assert {t["exit_reason"] for t in bot.journal.trades} <= {"stop", "target", "mean", "time", "target_taker", "session_end", "breaker"}


def test_no_shorts_where_shorting_is_off(tmp_path):
    exchange, bot, _ = asyncio.run(session(chosen(), tmp_path, 47933, allow_shorting=False))
    check_books(exchange, bot)
    assert bot.journal.trades and all(t["side"] == 1 for t in bot.journal.trades)
