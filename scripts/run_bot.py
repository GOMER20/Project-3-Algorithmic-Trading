"""Runs AT against a Gomer Trading server for one session, then writes its report.

    .venv/bin/python scripts/run_bot.py --minutes 60            # trade for an hour, starting now
    .venv/bin/python scripts/run_bot.py --us-session            # wait for the US open, trade to the close
    .venv/bin/python scripts/run_bot.py --us-session --api-port 8787

Connection settings come from the environment or a `.env` file beside this project:

    AT_BASE_URL=http://127.0.0.1:47831     the Gomer Trading app on this Mac
    AT_TRADER_ID=...                       AT's Trader ID
    AT_PASSWORD=...                        AT's password
    AT_COMPETITION=...                     optional: competition name or id
    AT_SYMBOLS=AAPL,TSLA,...               optional

The strategy is read from configs/at.json (written by scripts/run_research.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mrbot.bot import BOT_NAME, DEFAULT_SYMBOLS, BotConfig, Journal, TradingBot  # noqa: E402
from mrbot.config import StrategyConfig  # noqa: E402
from mrbot.exchange import ExchangeError, GomerClient  # noqa: E402

log = logging.getLogger("at.run")


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


async def wait_for_server(client: GomerClient) -> None:
    """The app may not be open yet. Keep trying quietly until it answers."""
    announced = False
    while True:
        try:
            await client.login()
            return
        except ExchangeError as err:
            if err.status in (400, 401, 403):
                raise SystemExit(f"{BOT_NAME} could not sign in: {err.message}")
            if not announced:
                log.info("Waiting for Gomer Trading at %s (is the app open?) ...", client.base_url)
                announced = True
            await asyncio.sleep(15)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=None, help="trade for this long, then flatten and report")
    parser.add_argument("--us-session", action="store_true", help="wait for the next US market open and trade until just before the close")
    parser.add_argument("--flatten-before-close", type=float, default=3.0, help="minutes before the close to stop and flatten")
    parser.add_argument("--config", default=str(ROOT / "configs/at.json"))
    parser.add_argument("--api-port", type=int, default=0, help="serve the local control panel on this port")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env(ROOT / ".env")

    trader_id, password = os.environ.get("AT_TRADER_ID", ""), os.environ.get("AT_PASSWORD", "")
    if not trader_id or not password:
        raise SystemExit("Set AT_TRADER_ID and AT_PASSWORD (in the environment or in .env). See the README.")
    strategy = StrategyConfig(**json.loads(Path(args.config).read_text()))
    cfg = BotConfig(
        base_url=os.environ.get("AT_BASE_URL", "http://127.0.0.1:47831"), trader_id=trader_id, password=password,
        competition=os.environ.get("AT_COMPETITION") or None,
        symbols=[s for s in os.environ.get("AT_SYMBOLS", "").split(",") if s] or list(DEFAULT_SYMBOLS),
        run_dir=str(ROOT / "runs"),
    )
    client = GomerClient(cfg.base_url, cfg.trader_id, cfg.password)
    await wait_for_server(client)

    until_wall = None
    if args.us_session:
        hours = await client.market_hours()
        if not hours["open"]:
            opens = epoch(hours["session"]["opensAt"])
            log.info("US market opens at %s local time. Waiting.", datetime.fromtimestamp(opens).strftime("%a %d %b %H:%M"))
            while time.time() < opens + 20:
                await asyncio.sleep(min(60.0, max(1.0, opens + 20 - time.time())))
            hours = await client.market_hours()
        until_wall = epoch(hours["session"]["closesAt"]) - 60 * args.flatten_before_close
    if args.minutes:
        limit = time.time() + 60 * args.minutes
        until_wall = min(until_wall, limit) if until_wall else limit

    journal = Journal(Path(cfg.run_dir) / time.strftime("%Y-%m-%d_%H%M"))
    bot = TradingBot(client, strategy, cfg, journal)
    await bot.start()

    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, name), lambda: setattr(bot, "stopping", True))
    server_task = None
    if args.api_port:
        import uvicorn

        from mrbot.api import create_app

        server = uvicorn.Server(uvicorn.Config(create_app(bot), host="127.0.0.1", port=args.api_port, log_level="warning"))
        server.install_signal_handlers = lambda: None  # the bot owns Ctrl-C
        server_task = asyncio.create_task(server.serve())
        log.info("Control panel: http://127.0.0.1:%d/status", args.api_port)

    try:
        await bot.run(until_wall=until_wall)
    finally:
        await bot.shutdown("stopped by hand" if bot.stopping and (until_wall is None or time.time() < until_wall) else "session_end")
        expected = None
        results = ROOT / "reports/backtest_results.json"
        if results.exists():
            expected = json.loads(results.read_text()).get("holdout")
        report = journal.write_report(expected)
        if server_task:
            server_task.cancel()
        await client.close()
        print(f"\n{report.read_text()}\nSaved to {report}")


if __name__ == "__main__":
    asyncio.run(main())
