"""A stand-in for Gomer Trading's server, for testing the bot end to end.

It speaks the same REST and WebSocket API and applies the same execution rules (a taker pays
a cent of slippage; a resting limit fills at its own price when the opposite quote reaches
it; brackets become reduce-only exits that cancel each other). Prices are replayed from bar
history, and time only moves when the test calls `advance()`, so a whole session runs in
seconds and gives the same result every time.

It is a test double, not a copy of the real engine: one trader, one competition, no depth,
no partial fills, no margin calls.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

SLIPPAGE = 0.01
COMPETITION_ID = "00000000-0000-4000-8000-000000000001"


class MockExchange:
    def __init__(self, bars: dict[str, pd.DataFrame], warmup: int = 400, starting_cash: float = 100_000.0,
                 trader_id: str = "AT-BOT", password: str = "test-password", allow_shorting: bool = True):
        self.bars = bars
        self.index = next(iter(bars.values())).index
        self.cursor = warmup
        """Index of the bar currently forming. Bars before it are finished history."""
        self.trader_id, self.password = trader_id, password
        self.token = "mock-token"
        self.allow_shorting = allow_shorting
        self.starting_cash = self.cash = starting_cash
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.sequence = 0
        self.quotes: dict[str, dict] = {}
        self.sockets: list[tuple[WebSocket, set[str]]] = []
        self.realized = 0.0
        self.app = self._build()
        for symbol in bars:
            row = bars[symbol].iloc[self.cursor - 1]
            self._set_quote(symbol, row["bid_close"], row["ask_close"], self._ts(self.cursor - 1, 59.5))

    # --- time ----------------------------------------------------------------------------
    def _ts(self, position: int, offset: float) -> int:
        return int((self.index[position].timestamp() + offset) * 1000)

    def _set_quote(self, symbol: str, bid: float, ask: float, ts: int) -> None:
        self.quotes[symbol] = {"symbol": symbol, "bid": round(float(bid), 2), "ask": round(float(ask), 2), "bidSize": 500, "askSize": 500,
                               "last": round(float(bid), 2), "lastSize": 100, "prevClose": None, "ts": ts}

    async def advance(self, minutes: int = 1, after_tick=None) -> int:
        """Plays the next `minutes` bars as four quotes each. Returns the timestamp (ms) of the last quote sent.

        `after_tick(ts)` is awaited after every quote, which is how a test lets the bot react
        between quotes the way it would in real time."""
        ts = 0
        for _ in range(minutes):
            if self.cursor >= len(self.index):
                break
            position = self.cursor
            rows = {symbol: frame.iloc[position] for symbol, frame in self.bars.items()}
            # A plausible path through the bar: open, then the extremes in the order the close implies, then the close.
            for step, offset in enumerate((0.5, 20.0, 40.0, 59.5)):
                ts = self._ts(position, offset)
                events: list[dict] = []
                for symbol, row in rows.items():
                    up = row["close"] >= row["open"]
                    low, high = (row["bid_low"], row["ask_low"]), (row["bid_high"], row["ask_high"])
                    previous = self.bars[symbol].iloc[position - 1]
                    bid, ask = [(previous["bid_close"], previous["ask_close"]), low if up else high, high if up else low, (row["bid_close"], row["ask_close"])][step]
                    self._set_quote(symbol, bid, ask, ts)
                    events += self._match(symbol)
                await self._broadcast({"type": "quotes", "data": list(self.quotes.values())})
                for order in events:
                    await self._broadcast({"type": "order", "competitionId": COMPETITION_ID, "data": order})
                if after_tick is not None:
                    await after_tick(ts)
            self.cursor += 1
        return ts

    # --- matching ------------------------------------------------------------------------
    def _new_order(self, **fields: Any) -> dict:
        self.sequence += 1
        order = {
            "id": str(uuid.UUID(int=self.sequence)), "portfolioId": "p", "status": "OPEN", "limitPrice": None, "stopPrice": None,
            "triggeredAt": None, "takeProfit": None, "stopLoss": None, "parentId": None, "filledQuantity": 0, "commission": 0.0,
            "forcedBy": None, "fillPrice": None, "realizedPnl": None, "rejectReason": None, "createdAt": str(self.sequence), "filledAt": None,
            "_seq": self.sequence,
        }
        order.update(fields)
        self.orders[order["id"]] = order
        return order

    def _fill(self, order: dict, price: float) -> list[dict]:
        """Books a complete fill and returns every order whose state changed because of it."""
        symbol, quantity = order["ticker"], order["quantity"]
        signed = quantity if order["side"] == "BUY" else -quantity
        position = self.positions.get(symbol, {"quantity": 0, "averagePrice": 0.0})
        held, average = position["quantity"], position["averagePrice"]
        realized = 0.0
        if held == 0 or (held > 0) == (signed > 0):
            average = (average * abs(held) + price * quantity) / (abs(held) + quantity)
        else:
            closing = min(quantity, abs(held))
            realized = (price - average) * closing * (1 if held > 0 else -1)
            if quantity > abs(held):
                average = price
        self.cash -= signed * price
        self.realized += realized
        held += signed
        if held == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = {"quantity": held, "averagePrice": average}
        order.update(status="FILLED", filledQuantity=quantity, fillPrice=round(price, 4), realizedPnl=round(realized, 4), filledAt="now")
        changed = [order]
        if order["parentId"] is None and (order["takeProfit"] is not None or order["stopLoss"] is not None):
            exit_side = "SELL" if order["side"] == "BUY" else "BUY"
            if order["takeProfit"] is not None:
                changed.append(self._new_order(ticker=symbol, side=exit_side, type="LIMIT", quantity=quantity, limitPrice=order["takeProfit"], parentId=order["id"]))
            if order["stopLoss"] is not None:
                changed.append(self._new_order(ticker=symbol, side=exit_side, type="STOP", quantity=quantity, stopPrice=order["stopLoss"], parentId=order["id"]))
        if held == 0:
            # Closing the position by any route cancels the exits that were protecting it.
            for other in self.orders.values():
                if other["ticker"] == symbol and other["status"] == "OPEN" and other["parentId"] is not None:
                    other.update(status="CANCELLED", rejectReason="Position closed")
                    changed.append(other)
        return changed

    def _match(self, symbol: str) -> list[dict]:
        quote = self.quotes[symbol]
        changed: list[dict] = []
        for order in sorted((o for o in self.orders.values() if o["ticker"] == symbol and o["status"] == "OPEN"), key=lambda o: o["_seq"]):
            if order["status"] != "OPEN":
                continue  # cancelled by an earlier fill in this same pass
            buy = order["side"] == "BUY"
            if order["parentId"] is not None:
                held = self.positions.get(symbol, {}).get("quantity", 0)
                if held == 0 or (held > 0) == buy:
                    order.update(status="CANCELLED", rejectReason="Position already closed")
                    changed.append(order)
                    continue
                order["quantity"] = min(order["quantity"], abs(held))
            if order["type"] == "STOP":
                if (quote["ask"] >= order["stopPrice"]) if buy else (quote["bid"] <= order["stopPrice"]):
                    changed += self._fill(order, quote["ask"] + SLIPPAGE if buy else quote["bid"] - SLIPPAGE)
            elif order["type"] == "LIMIT":
                if (quote["ask"] <= order["limitPrice"]) if buy else (quote["bid"] >= order["limitPrice"]):
                    changed += self._fill(order, order["limitPrice"])
        return changed

    def _equity(self) -> float:
        value = self.cash
        for symbol, position in self.positions.items():
            quote = self.quotes[symbol]
            value += position["quantity"] * (quote["bid"] + quote["ask"]) / 2
        return value

    @staticmethod
    def _public(order: dict) -> dict:
        return {k: v for k, v in order.items() if not k.startswith("_")}

    async def _broadcast(self, message: dict) -> None:
        for socket, subscribed in list(self.sockets):
            payload = message
            if message["type"] == "quotes":
                payload = {"type": "quotes", "data": [q for q in message["data"] if q["symbol"] in subscribed]}
            elif message["type"] == "order":
                payload = {**message, "data": self._public(message["data"])}
            try:
                await socket.send_json(payload)
            except Exception:
                self.sockets = [(s, subs) for s, subs in self.sockets if s is not socket]

    # --- API -----------------------------------------------------------------------------
    def _build(self) -> FastAPI:
        app = FastAPI()
        exchange = self

        @app.exception_handler(HTTPException)
        async def _errors(_request: Request, err: HTTPException):
            code, _, message = str(err.detail).partition(": ")
            return JSONResponse({"error": {"code": code, "message": message}}, status_code=err.status_code)

        def authed(request: Request) -> None:
            if request.headers.get("authorization") != f"Bearer {exchange.token}":
                raise HTTPException(401, "UNAUTHORIZED: Invalid or expired session")

        @app.post("/api/auth/login")
        async def login(body: dict):
            if body.get("traderId") != exchange.trader_id or body.get("password") != exchange.password:
                raise HTTPException(401, "UNAUTHORIZED: Wrong Trader ID or password")
            return {"token": exchange.token, "expiresIn": 43200, "user": {"id": "u", "traderId": exchange.trader_id, "role": "TRADER", "displayName": "AT"}}

        @app.get("/api/competitions")
        async def competitions(request: Request):
            authed(request)
            return {"competitions": [{"id": COMPETITION_ID, "name": "Mock Cup", "status": "ACTIVE", "allowShorting": exchange.allow_shorting,
                                      "leverage": 1, "commissionPerShare": 0, "startingBalance": exchange.starting_cash, "enrolled": True}]}

        @app.get("/api/portfolios/{competition_id}")
        async def portfolio(competition_id: str, request: Request):
            authed(request)
            equity = exchange._equity()
            gross = sum(abs(p["quantity"]) * exchange.quotes[s]["bid"] for s, p in exchange.positions.items())
            positions = [{"ticker": s, "quantity": p["quantity"], "side": "LONG" if p["quantity"] > 0 else "SHORT", "averagePrice": p["averagePrice"]}
                         for s, p in exchange.positions.items()]
            return {"portfolio": {"id": "p", "competitionId": competition_id, "cash": exchange.cash, "totalEquity": equity,
                                  "buyingPower": max(0.0, equity - gross), "allowShorting": exchange.allow_shorting, "positions": positions}}

        @app.get("/api/portfolios/{competition_id}/orders")
        async def orders(competition_id: str, request: Request, status: str | None = None, limit: int = 100):
            authed(request)
            rows = sorted((o for o in exchange.orders.values() if status is None or o["status"] == status), key=lambda o: -o["_seq"])[:limit]
            return {"orders": [exchange._public(o) for o in rows], "total": len(rows), "limit": limit, "offset": 0}

        @app.get("/api/market/bars/{symbol}")
        async def bars(symbol: str, request: Request, limit: int = 300, timeframe: str = "1Min"):
            authed(request)
            frame = exchange.bars[symbol].iloc[max(0, exchange.cursor - limit) : exchange.cursor]
            rows = [{"time": int(ts.timestamp()), "open": r.open, "high": r.high, "low": r.low, "close": r.close, "volume": int(r.volume)} for ts, r in frame.iterrows()]
            # Like the real server, the newest bar returned is the one still forming.
            now = exchange.quotes[symbol]
            if exchange.cursor < len(exchange.index):
                rows.append({"time": int(exchange.index[exchange.cursor].timestamp()), "open": now["bid"], "high": now["ask"], "low": now["bid"], "close": now["bid"], "volume": 1})
            return {"symbol": symbol, "timeframe": timeframe, "bars": rows[-limit:]}

        @app.get("/api/market/hours")
        async def hours():
            return {"open": True, "session": {"date": "mock", "opensAt": None, "closesAt": None, "earlyClose": False}, "holiday": None, "upcoming": []}

        @app.post("/api/orders", status_code=201)
        async def place(body: dict, request: Request):
            authed(request)
            symbol, side, kind, quantity = body["ticker"], body["side"], body["type"], int(body["quantity"])
            quote = exchange.quotes.get(symbol)
            if quote is None:
                raise HTTPException(422, f"ORDER_REJECTED: {symbol} is not a tradable symbol")
            for key in ("limitPrice", "stopPrice", "takeProfit", "stopLoss"):
                value = body.get(key)
                if value is not None and abs(value * 100 - round(value * 100)) > 1e-6:
                    raise HTTPException(400, f"VALIDATION: {key} must be in whole cents")
            held = exchange.positions.get(symbol, {}).get("quantity", 0)
            opening = held == 0 or (held > 0) == (side == "BUY")
            if side == "SELL" and not exchange.allow_shorting and quantity > max(held, 0):
                raise HTTPException(422, "ORDER_REJECTED: Short selling is disabled in this competition")
            if not opening and (body.get("takeProfit") is not None or body.get("stopLoss") is not None):
                raise HTTPException(422, "ORDER_REJECTED: A bracket belongs on an order that opens or adds to a position")
            order = exchange._new_order(ticker=symbol, side=side, type=kind, quantity=quantity, limitPrice=body.get("limitPrice"),
                                        stopPrice=body.get("stopPrice"), takeProfit=body.get("takeProfit"), stopLoss=body.get("stopLoss"))
            buy = side == "BUY"
            changed: list[dict] = []
            if kind == "MARKET":
                changed = exchange._fill(order, quote["ask"] + SLIPPAGE if buy else quote["bid"] - SLIPPAGE)
            elif kind == "LIMIT" and ((order["limitPrice"] >= quote["ask"]) if buy else (order["limitPrice"] <= quote["bid"])):
                taker = quote["ask"] + SLIPPAGE if buy else quote["bid"] - SLIPPAGE
                changed = exchange._fill(order, min(taker, order["limitPrice"]) if buy else max(taker, order["limitPrice"]))
            for other in changed:
                if other is not order:
                    await exchange._broadcast({"type": "order", "competitionId": COMPETITION_ID, "data": other})
            return {"order": exchange._public(order)}

        @app.delete("/api/orders/{order_id}")
        async def cancel(order_id: str, request: Request):
            authed(request)
            order = exchange.orders.get(order_id)
            if order is None or order["status"] != "OPEN":
                raise HTTPException(404, "NOT_FOUND: No working order with that id")
            order.update(status="CANCELLED")
            return {"order": exchange._public(order)}

        @app.websocket("/ws")
        async def stream(socket: WebSocket):
            await socket.accept()
            subscribed: set[str] = set()
            entry = (socket, subscribed)
            try:
                first = await socket.receive_json()
                if first.get("type") != "auth" or first.get("token") != exchange.token:
                    await socket.send_json({"type": "error", "code": "UNAUTHORIZED", "message": "Invalid or expired session"})
                    return
                await socket.send_json({"type": "ready", "traderId": exchange.trader_id, "provider": "mock", "feed": {"state": "streaming"}})
                exchange.sockets.append(entry)
                while True:
                    message = await socket.receive_json()
                    if message.get("type") == "subscribe":
                        for symbol in message["symbols"]:
                            if symbol in exchange.quotes:
                                subscribed.add(symbol)
                                await socket.send_json({"type": "quote", "data": exchange.quotes[symbol]})
                    elif message.get("type") == "ping":
                        await socket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                pass
            finally:
                exchange.sockets = [(s, subs) for s, subs in exchange.sockets if s is not socket]

        return app


async def serve(exchange: MockExchange, port: int):
    """Runs the mock on 127.0.0.1:port inside the current event loop. Returns (server, task)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(exchange.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task
