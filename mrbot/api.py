"""A small local control panel for the running bot (FastAPI).

    GET  /status   what AT is holding and working right now
    GET  /stats    the session's statistics so far (same numbers as the final report)
    GET  /trades   every closed trade
    POST /halt     stop opening new positions (existing ones keep their exits)
    POST /resume

It binds to 127.0.0.1 only. It has no authentication, so do not expose it to a network.
"""
from __future__ import annotations

from fastapi import FastAPI

from .bot import TradingBot


def create_app(bot: TradingBot) -> FastAPI:
    app = FastAPI(title="AT control")

    @app.get("/status")
    async def status() -> dict:
        return bot.status()

    @app.get("/stats")
    async def stats() -> dict:
        return {k: v for k, v in bot.journal.stats().items() if k != "exits"} | {"exits": bot.journal.stats().get("exits", {})}

    @app.get("/trades")
    async def trades() -> list[dict]:
        return [{k: (str(v) if k.endswith("_time") else v) for k, v in t.items()} for t in bot.journal.trades]

    @app.post("/halt")
    async def halt() -> dict:
        bot.risk.halted = True
        bot.journal.note("Halted by hand through the control panel.")
        return {"halted": True}

    @app.post("/resume")
    async def resume() -> dict:
        bot.risk.halted = False
        bot.journal.note("Resumed by hand through the control panel.")
        return {"halted": False}

    return app
