"""Client for Gomer Trading's bot gateway (JSON over WebSocket at /gateway).

This is the door the app provides for trading programs: sign in with an API key, subscribe
to market data, send orders, and receive an execution report for every step in the life of
every order. The message shapes follow the app's own docs/gateway/README.md.

The gateway has no price-history call, so `DesktopHistory` reads one-minute bars from the
app's chart API instead. It signs in the way the desktop app signs its owner in (a key the
app keeps on this Mac, accepted only from this Mac), and it is used for reading bars and
nothing else.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
from pathlib import Path

import httpx
import websockets

log = logging.getLogger("at.gateway")

APP_CONFIG = Path.home() / "Library/Application Support/Gomer Trading/config.json"


class GatewayError(Exception):
    """The exchange refused a request. `code` is one of the gateway's reject codes."""

    def __init__(self, code: str, text: str = ""):
        super().__init__(f"{code}: {text}")
        self.code, self.text = code, text


class GatewayClient:
    """One logged-on gateway session.

    Requests are awaited and return their reply (or raise GatewayError on a `reject`).
    Everything the exchange pushes on its own (`bbo`, `trade`, `exec`) lands in `events`, in
    the order it arrived, followed by `{"op": "disconnected"}` if the connection drops.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0):
        self.url = base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://") + "/gateway"
        self._api_key = api_key
        self.timeout = timeout
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self.session: dict = {}
        self._socket = None
        self._reader: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)

    @property
    def connected(self) -> bool:
        return self._reader is not None and not self._reader.done()

    async def connect(self) -> dict:
        """Opens the connection and logs on. Returns the logon acknowledgement (who we are, which sandbox)."""
        await self.close()
        self._socket = await websockets.connect(self.url, ping_interval=None, max_size=2**22, open_timeout=self.timeout)
        self._reader = asyncio.create_task(self._read(self._socket))
        self.session = await self.request("logon", apiKey=self._api_key)
        return self.session

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._socket is not None:
            await self._socket.close()
            self._socket = None

    async def _read(self, socket) -> None:
        try:
            async for raw in socket:
                message = json.loads(raw)
                future = self._pending.pop(message.get("id"), None) if isinstance(message.get("id"), int) else None
                if future is None:
                    self.events.put_nowait(message)
                elif future.done():
                    continue
                elif message.get("op") == "reject":
                    future.set_exception(GatewayError(message.get("code", "INTERNAL"), message.get("text", "")))
                else:
                    future.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            log.warning("gateway connection lost: %s", type(err).__name__)
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(GatewayError("DISCONNECTED", "The connection closed before the reply arrived."))
            self._pending.clear()
            self.events.put_nowait({"op": "disconnected"})

    async def request(self, op: str, **fields) -> dict:
        if self._socket is None:
            raise GatewayError("DISCONNECTED", "Not connected.")
        number = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[number] = future
        try:
            await self._socket.send(json.dumps({"op": op, "id": number, **fields}))
            return await asyncio.wait_for(future, self.timeout)
        except asyncio.TimeoutError:
            raise GatewayError("TIMEOUT", f"No reply to {op}.") from None
        except websockets.ConnectionClosed:
            raise GatewayError("DISCONNECTED", "The connection closed.") from None
        finally:
            self._pending.pop(number, None)

    # --- market data ---------------------------------------------------------------------
    async def subscribe(self, symbols: list[str], channels: tuple[str, ...] = ("bbo",)) -> dict:
        return await self.request("subscribe", channels=list(channels), symbols=symbols)

    # --- orders --------------------------------------------------------------------------
    async def new_order(self, cl_ord_id: str, symbol: str, side: str, type: str, qty: int,
                        price: float | None = None, stop_price: float | None = None, tif: str = "GTC") -> dict:
        fields: dict = {"clOrdId": cl_ord_id, "symbol": symbol, "side": side, "type": type, "tif": tif, "qty": int(qty)}
        if price is not None:
            fields["price"] = round(float(price), 2)
        if stop_price is not None:
            fields["stopPrice"] = round(float(stop_price), 2)
        return await self.request("order.new", **fields)

    async def cancel(self, cl_ord_id: str | None = None, order_id: str | None = None) -> bool:
        """Cancels a working order. False means it was already finished, which callers treat as normal."""
        try:
            await self.request("order.cancel", **({"clOrdId": cl_ord_id} if cl_ord_id else {"orderId": order_id}))
            return True
        except GatewayError as err:
            if err.code in ("TOO_LATE_TO_CANCEL", "UNKNOWN_ORDER"):
                return False
            raise

    async def open_orders(self) -> list[dict]:
        return (await self.request("orders.open"))["orders"]

    async def account(self) -> dict:
        return (await self.request("account"))["account"]


class DesktopHistory:
    """One-minute bar history from the app's chart API, for a bot running beside the desktop app."""

    def __init__(self, base_url: str, config_path: Path = APP_CONFIG):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/") + "/api", timeout=10.0)
        self._config_path = config_path
        self._token: str | None = None

    async def close(self) -> None:
        await self._http.aclose()

    async def _sign_in(self) -> None:
        key = json.loads(self._config_path.read_text())["desktopKey"]
        response = await self._http.post("/auth/desktop", headers={"x-desktop-key": key}, json={"account": "trader"})
        response.raise_for_status()
        self._token = response.json()["token"]

    async def bars(self, symbol: str, limit: int = 1000) -> list[dict]:
        for attempt in range(4):
            if self._token is None:
                await self._sign_in()
            response = await self._http.get(f"/market/bars/{symbol}", params={"timeframe": "1Min", "limit": limit}, headers={"Authorization": f"Bearer {self._token}"})
            if response.status_code == 401 and attempt < 3:
                self._token = None  # the session expired (they last twelve hours)
                continue
            if response.status_code == 503 and attempt < 3:
                # The app sheds load for a moment when it is busy (just after the Mac wakes, say) and says when to come back.
                await asyncio.sleep(min(5.0, float(response.headers.get("Retry-After", 1))))
                continue
            response.raise_for_status()
            return response.json()["bars"]
        return []

    async def leaderboard(self, competition_id: str) -> dict:
        """A competition's standings as the app shows them (the top ten), read as the app's owner."""
        if self._token is None:
            await self._sign_in()
        response = await self._http.get(f"/competitions/{competition_id}/leaderboard", headers={"Authorization": f"Bearer {self._token}"})
        response.raise_for_status()
        return response.json()

    async def market_hours(self) -> dict:
        response = await self._http.get("/market/hours")
        response.raise_for_status()
        return response.json()
