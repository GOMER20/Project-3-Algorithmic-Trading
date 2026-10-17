"""Runs AT against the Gomer Trading app on this Mac for one session, then reports.

    .venv/bin/python scripts/run_at.py --minutes 30         # trade for half an hour, starting now
    .venv/bin/python scripts/run_at.py --us-session         # wait for the US open, trade to the close
    .venv/bin/python scripts/run_at.py --us-session --once  # ...and never the same session twice

AT signs in with the API key in .env (written by scripts/admit_at.py) and trades the
strategy in configs/at.json. When the session ends it closes everything and writes
runs/<date>/report.md, trades.csv and stats.json; runs/latest.md always holds the newest report.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mrbot.atbot import GatewayBot  # noqa: E402
from mrbot.bot import BOT_NAME, DEFAULT_SYMBOLS, Journal  # noqa: E402
from mrbot.config import StrategyConfig  # noqa: E402
from mrbot.data import to_frame  # noqa: E402
from mrbot.gateway import DesktopHistory, GatewayClient, GatewayError  # noqa: E402
from mrbot.review import record, review_session  # noqa: E402

log = logging.getLogger("at.run")
SESSIONS = ROOT / "runs/sessions.json"


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def notify(title: str, text: str) -> None:
    """A macOS notification. Best effort: a report is written whether or not this shows."""
    try:
        subprocess.run(["osascript", "-e", f'display notification {json.dumps(text)} with title {json.dumps(title)}'], timeout=10, check=False)
    except Exception:
        pass


async def wait_for_app(history: DesktopHistory, base_url: str) -> dict:
    """The app may not be open yet. Keep trying quietly until it answers."""
    announced = False
    while True:
        try:
            return await history.market_hours()
        except Exception:
            if not announced:
                log.info("Waiting for Gomer Trading at %s (is the app open?) ...", base_url)
                announced = True
            await asyncio.sleep(15)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=None, help="trade for this long, then flatten and report")
    parser.add_argument("--us-session", action="store_true", help="wait for the next US market open and trade until just before the close")
    parser.add_argument("--once", action="store_true", help="with --us-session: do nothing if that session has already been traded")
    parser.add_argument("--only-session", default=None, metavar="YYYY-MM-DD", help="with --us-session: trade this session and no other")
    parser.add_argument("--flatten-before-close", type=float, default=3.0, help="minutes before the close to stop and flatten")
    parser.add_argument("--config", default=str(ROOT / "configs/at.json"))
    parser.add_argument("--api-port", type=int, default=0, help="serve the local control panel on this port")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%d %b %H:%M:%S")
    # One AT at a time. Two copies on one key would each cancel the other's orders as strangers.
    (ROOT / "runs").mkdir(exist_ok=True)
    lock = open(ROOT / "runs/.at.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"{BOT_NAME} is already running (it starts by itself at each market open). Nothing to do.")
        return 0
    logging.getLogger("httpx").setLevel(logging.WARNING)
    load_env(ROOT / ".env")

    base_url, api_key = os.environ.get("AT_BASE_URL", "http://127.0.0.1:47831"), os.environ.get("AT_API_KEY", "")
    if not api_key:
        # Exit codes: 0 also covers "nothing can be done until a person steps in", so a
        # supervisor that restarts on failure does not spin on something a restart cannot fix.
        print("AT has no API key yet. Run scripts/admit_at.py first.")
        return 0
    strategy = StrategyConfig(**json.loads(Path(args.config).read_text()))
    symbols = [s for s in os.environ.get("AT_SYMBOLS", "").split(",") if s] or list(DEFAULT_SYMBOLS)
    history = DesktopHistory(base_url)
    hours = await wait_for_app(history, base_url)

    until_wall, session_date = None, time.strftime("%Y-%m-%d_%H%M")
    if args.us_session:
        done = json.loads(SESSIONS.read_text()) if SESSIONS.exists() else {}
        if args.only_session and hours["session"]["date"] != args.only_session:
            log.info("The next session is %s, not %s. Nothing to do.", hours["session"]["date"], args.only_session)
            return 0
        if not hours["open"]:
            opens = epoch(hours["session"]["opensAt"])
            if args.once and hours["session"]["date"] in done:
                log.info("The %s session has already been traded. Nothing to do.", hours["session"]["date"])
                return 0
            log.info("US market opens %s (local time). Waiting.", datetime.fromtimestamp(opens).strftime("%a %d %b at %H:%M"))
            while time.time() < opens + 20:
                await asyncio.sleep(min(60.0, max(1.0, opens + 20 - time.time())))
            hours = await wait_for_app(history, base_url)
        session_date = hours["session"]["date"]
        if args.once and session_date in done:
            log.info("The %s session has already been traded. Nothing to do.", session_date)
            return 0
        until_wall = epoch(hours["session"]["closesAt"]) - 60 * args.flatten_before_close
    if args.minutes:
        limit = time.time() + 60 * args.minutes
        until_wall = min(until_wall, limit) if until_wall else limit

    # One seat per competition AT trades in: its own sandbox, and any other competition it has a key for.
    # Every seat is the same AT account trading the same rules; each has its own portfolio and report.
    seats = [("", "AT_API_KEY", "", "latest.md", "reviews.jsonl")]
    if os.environ.get("AT_API_KEY_DEMO"):
        seats.append(("D", "AT_API_KEY_DEMO", "-demo", "latest-demo.md", "reviews-demo.jsonl"))
    folder = session_date if args.us_session else time.strftime("%Y-%m-%d_%H%M")
    bots: list[tuple[GatewayBot, tuple]] = []
    for seat in seats:
        tag, env_name, suffix, _, _ = seat
        bot = GatewayBot(GatewayClient(base_url, os.environ[env_name]), history, strategy, symbols, Journal(ROOT / "runs" / f"{folder}{suffix}"), tag=tag)
        while True:
            try:
                await bot.start()
                bots.append((bot, seat))
                break
            except GatewayError as err:
                if err.code == "BAD_CREDENTIALS":
                    print(f"The app refused the key in {env_name} ({err.text}). Run scripts/admit_at.py again.")
                    break
                raise
            except OSError:
                await asyncio.sleep(10)
    if not bots:
        return 0

    try:
        # Keep the Mac from idling to sleep for as long as this process is trading.
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError:
        pass
    loop = asyncio.get_running_loop()
    for name in ("SIGINT", "SIGTERM"):
        loop.add_signal_handler(getattr(signal, name), lambda: [setattr(bot, "stopping", True) for bot, _ in bots])
    server_task = None
    if args.api_port:
        import uvicorn

        from mrbot.api import create_app

        server = uvicorn.Server(uvicorn.Config(create_app(bots[0][0]), host="127.0.0.1", port=args.api_port, log_level="warning"))
        server.install_signal_handlers = lambda: None  # the bot owns Ctrl-C
        server_task = asyncio.create_task(server.serve())
        log.info("Control panel: http://127.0.0.1:%d/stats", args.api_port)

    by_hand = False
    try:
        await asyncio.gather(*(bot.run(until_wall=until_wall) for bot, _ in bots))
        by_hand = any(bot.stopping for bot, _ in bots)
    finally:
        import pandas as pd

        expected = None
        results = ROOT / "reports/backtest_results.json"
        if results.exists():
            expected = json.loads(results.read_text()).get("holdout")
        lines, reports = [], []
        for bot, (tag, env_name, suffix, latest, reviews) in bots:
            await bot.shutdown("stopped by hand" if by_hand else "session_end")
            where = bot.competition["name"]
            title = f"{BOT_NAME} — live session {session_date} in {where}" + (" (US market hours)" if args.us_session else "")
            report = bot.journal.write_report(expected, title)
            try:
                # What went wrong on the losing trades: replay the session's own minutes through the backtester.
                fetched = await asyncio.gather(*(history.bars(s, 1000) for s in bot.slots), return_exceptions=True)
                session_bars = {s: to_frame(rows[:-1]) for s, rows in zip(bot.slots, fetched) if not isinstance(rows, Exception) and rows}
                review = review_session(pd.DataFrame(bot.journal.trades, columns=Journal.FIELDS), session_bars, strategy, expected)
                with open(report, "a") as handle:
                    handle.write("\n" + review["text"])
                (report.parent / "loss_review.json").write_text(json.dumps({k: v for k, v in review.items() if k != "text"}, indent=2, default=float))
                record(review, session_date, ROOT / "runs" / reviews)
            except Exception as err:  # the report already exists; the review is an addition to it
                log.warning("loss review skipped: %s", err)
            if tag:
                # A competition with other traders in it: say where AT stands on its leaderboard.
                try:
                    board = await history.leaderboard(bot.competition["id"])
                    mine = next((e for e in board.get("entries", []) if e.get("traderId") == bot.client.session.get("traderId")), None)
                    standing = (f"On the '{where}' leaderboard AT is **#{mine['rank']} of {board.get('participants')}**, with ${mine['equity']:,.2f}."
                                if mine else f"On the '{where}' leaderboard AT is outside the top {len(board.get('entries', []))} of {board.get('participants')}.")
                    with open(report, "a") as handle:
                        handle.write(f"\n## Leaderboard\n\n{standing}\n")
                except Exception as err:
                    log.warning("leaderboard not read: %s", err)
            (ROOT / "runs" / latest).write_text(report.read_text())
            reports.append(report)
            stats = bot.journal.stats()
            if stats.get("trades"):
                wins = round(stats["win_rate"] * stats["trades"])
                lines.append(f"{where}: success rate {100 * stats['win_rate']:.1f}% ({wins} of {stats['trades']}), net ${stats['net_pnl']:+,.0f}.")
            else:
                lines.append(f"{where}: no trades.")
            await bot.client.close()
        if args.us_session and not by_hand:
            done = json.loads(SESSIONS.read_text()) if SESSIONS.exists() else {}
            done[session_date] = str(reports[0])
            SESSIONS.write_text(json.dumps(done, indent=2))
        notify(f"{BOT_NAME} finished its session", " ".join(lines))
        if server_task:
            server_task.cancel()
        await history.close()
        for report in reports:
            print(f"\n{report.read_text()}\nSaved to {report}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
