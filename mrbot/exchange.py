"""REST and WebSocket client for Gomer Trading's API.

The shapes here follow the server's own schemas (backend/src/middleware/risk.ts for orders,
backend/src/ws/gateway.ts for the stream). Nothing in the bot talks to the network except
through this class, so pointing the bot at another simulated exchange means writing one more
class with the same methods.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

import httpx
import websockets

log = logging.getLogger("at.exchange")


class ExchangeError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message

    @property
    def rejected(self) -> bool:
        """The exchange understood the order and refused it (a trading rule, not a fault)."""
        return self.status == 422 or self.code == "ORDER_REJECTED"


class GomerClient:
    def __init__(self, base_url: str, trader_id: str, password: str, timeout: float = 10.0, retries: int = 4):
        self.base_url = base_url.rstrip("/")
        self.trader_id, self._password = trader_id, password
        self.retries = retries
        self._http = httpx.AsyncClient(base_url=f"{self.base_url}/api", timeout=timeout)
        self._token: str | None = None
        self._login_lock = asyncio.Lock()

    @property
    def ws_url(self) -> str:
        return self.base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"

    async def close(self) -> None:
        await self._http.aclose()

    # --- session -----------------------------------------------------------------------
    async def login(self) -> dict:
        async with self._login_lock:
            response = await self._http.post("/auth/login", json={"traderId": self.trader_id, "password": self._password})
            body = self._parse(response)
            self._token = body["token"]
            return body["user"]

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """One API call. Signs in again if the session has expired, and waits out 429/503 (the
        server sheds load on purpose and says when to come back)."""
        if self._token is None:
            await self.login()
        delay = 0.5
        for attempt in range(self.retries + 1):
            try:
                response = await self._http.request(method, path, headers={"Authorization": f"Bearer {self._token}"}, **kwargs)
            except httpx.TransportError as err:
                if attempt == self.retries:
                    raise ExchangeError(0, "NETWORK", str(err)) from err
                await asyncio.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            if response.status_code == 401 and attempt < self.retries:
                await self.login()
                continue
            if response.status_code in (429, 503) and attempt < self.retries:
                wait = float(response.headers.get("Retry-After", delay))
                await asyncio.sleep(min(max(wait, 0.2), 10.0))
                delay = min(delay * 2, 8.0)
                continue
            return self._parse(response)
        raise ExchangeError(0, "RETRIES", f"{method} {path} did not succeed")

    @staticmethod
    def _parse(response: httpx.Response) -> dict:
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            error = body.get("error", {}) if isinstance(body, dict) else {}
            raise ExchangeError(response.status_code, error.get("code", "HTTP_ERROR"), error.get("message", response.text[:200]))
        return body

    # --- account -----------------------------------------------------------------------
    async def competitions(self) -> list[dict]:
        return (await self._request("GET", "/competitions"))["competitions"]

    async def portfolio(self, competition_id: str) -> dict:
        return (await self._request("GET", f"/portfolios/{competition_id}"))["portfolio"]

    async def open_orders(self, competition_id: str) -> list[dict]:
        return (await self._request("GET", f"/portfolios/{competition_id}/orders", params={"status": "OPEN", "limit": 500}))["orders"]

    async def recent_orders(self, competition_id: str, limit: int = 100) -> list[dict]:
        """The newest orders in any state, newest first."""
        return (await self._request("GET", f"/portfolios/{competition_id}/orders", params={"limit": limit}))["orders"]

    # --- market data -------------------------------------------------------------------
    async def bars(self, symbol: str, limit: int = 1000, timeframe: str = "1Min") -> list[dict]:
        return (await self._request("GET", f"/market/bars/{symbol}", params={"timeframe": timeframe, "limit": limit}))["bars"]

    async def market_hours(self) -> dict:
        return await self._request("GET", "/market/hours")

    # --- orders ------------------------------------------------------------------------
    async def place_order(
        self,
        competition_id: str,
        ticker: str,
        side: str,
        type: str,
        quantity: int,
        limit_price: float | None = None,
        stop_price: float | None = None,
        take_profit: float | None = None,
        stop_loss: float | None = None,
    ) -> dict:
        body: dict[str, Any] = {"competitionId": competition_id, "ticker": ticker, "side": side, "type": type, "quantity": int(quantity)}
        for key, value in (("limitPrice", limit_price), ("stopPrice", stop_price), ("takeProfit", take_profit), ("stopLoss", stop_loss)):
            if value is not None:
                body[key] = round(float(value), 2)  # the exchange accepts whole cents only
        return (await self._request("POST", "/orders", json=body))["order"]

    async def cancel_order(self, order_id: str) -> dict | None:
        """Cancels a working order. Returns None if it was already gone, which is not an error for the caller."""
        try:
            return (await self._request("DELETE", f"/orders/{order_id}"))["order"]
        except ExchangeError as err:
            if err.status in (400, 404, 409, 422):
                return None
            raise

    # --- stream ------------------------------------------------------------------------
    async def stream(self, symbols: list[str]) -> AsyncIterator[dict]:
        """Quote and order messages, forever. Reconnects with backoff and re-subscribes; yields
        {"type": "connected"} after each (re)connection so the caller can re-sync its state."""
        delay = 1.0
        while True:
            try:
                if self._token is None:
                    await self.login()
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=20, max_size=2**22) as socket:
                    await socket.send(json.dumps({"type": "auth", "token": self._token}))
                    async for raw in socket:
                        message = json.loads(raw)
                        kind = message.get("type")
                        if kind == "ready":
                            for start in range(0, len(symbols), 50):
                                await socket.send(json.dumps({"type": "subscribe", "symbols": symbols[start : start + 50]}))
                            delay = 1.0
                            yield {"type": "connected", "provider": message.get("provider")}
                        elif kind == "error" and message.get("code") == "UNAUTHORIZED":
                            self._token = None  # signed out elsewhere or expired: sign in again and reconnect
                            break
                        else:
                            yield message
            except asyncio.CancelledError:
                raise
            except Exception as err:  # connection refused, dropped, app restarting
                log.warning("stream disconnected (%s); reconnecting in %.0fs", type(err).__name__, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
