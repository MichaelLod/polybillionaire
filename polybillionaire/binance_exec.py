"""Authenticated Binance USDM futures client.

Reads BINANCE_KEY and BINANCE_SECRET from env. Uses raw httpx + HMAC
so we don't depend on the monolithic python-binance package.

Scope is intentionally tight — the things the futures loop actually
uses: balance, positions, set leverage, open market order, place stop,
close position. If you find yourself adding something exotic here,
question whether the trading loop needs it.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from math import floor
from urllib.parse import urlencode

import httpx

FAPI_BASE = "https://fapi.binance.com"


class BinanceError(Exception):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


@dataclass
class SymbolSpec:
    symbol: str
    price_precision: int
    qty_precision: int
    step_size: float
    min_qty: float
    tick_size: float
    min_notional: float


@dataclass
class OpenPosition:
    symbol: str
    side: str           # "LONG" | "SHORT"
    qty: float          # absolute, base-asset units
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int


class BinanceFutures:
    def __init__(self, key: str, secret: str, *, timeout: float = 15.0) -> None:
        self.key = key
        self.secret = secret
        self._http = httpx.Client(timeout=timeout)
        self._spec_cache: dict[str, SymbolSpec] = {}

    def close(self) -> None:
        self._http.close()

    # ── HTTP primitives ────────────────────────────

    def _sign(self, params: dict) -> str:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        qs = urlencode(params)
        sig = hmac.new(
            self.secret.encode(), qs.encode(), hashlib.sha256,
        ).hexdigest()
        return f"{qs}&signature={sig}"

    def _request(self, method: str, path: str, params: dict | None = None) -> dict | list:
        qs = self._sign(params or {})
        url = f"{FAPI_BASE}{path}?{qs}"
        r = self._http.request(method, url, headers={"X-MBX-APIKEY": self.key})
        if r.status_code >= 400:
            try:
                body = r.json()
                raise BinanceError(int(body.get("code", r.status_code)), str(body.get("msg", r.text)))
            except (ValueError, KeyError):
                raise BinanceError(r.status_code, r.text[:200])
        return r.json()

    def _get(self, path: str, params: dict | None = None):
        return self._request("GET", path, params)

    def _post(self, path: str, params: dict | None = None):
        return self._request("POST", path, params)

    def _delete(self, path: str, params: dict | None = None):
        return self._request("DELETE", path, params)

    # ── Account ────────────────────────────────────

    def get_available_usdt(self) -> float:
        balances = self._get("/fapi/v2/balance")
        for a in balances:
            if a.get("asset") == "USDT":
                return float(a.get("availableBalance", 0))
        return 0.0

    def get_available_collateral(self) -> float:
        """Return aggregate spendable futures collateral in USD terms.

        Uses ``/fapi/v2/account`` so it covers any collateral mode:
        plain USDT, multi-assets mode (USDC/BUSD), or Credits Trading
        Mode (BNFCR/BFUSD). Binance expresses ``availableBalance`` in
        USD regardless of the underlying asset mix.
        """
        acct = self._get("/fapi/v2/account")
        return float(acct.get("availableBalance") or 0)

    def get_positions(self) -> list[OpenPosition]:
        acct = self._get("/fapi/v2/account")
        out: list[OpenPosition] = []
        for p in acct.get("positions", []):
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                continue
            entry = float(p.get("entryPrice", 0))
            mark = float(p.get("markPrice", 0) or 0)
            if mark <= 0:
                mark = self.mark_price(p["symbol"])
            out.append(OpenPosition(
                symbol=p["symbol"],
                side="LONG" if amt > 0 else "SHORT",
                qty=abs(amt),
                entry_price=entry,
                mark_price=mark,
                unrealized_pnl=float(p.get("unrealizedProfit", 0)),
                leverage=int(p.get("leverage", 1)),
            ))
        return out

    def mark_price(self, symbol: str) -> float:
        r = self._http.get(
            f"{FAPI_BASE}/fapi/v1/premiumIndex",
            params={"symbol": symbol}, timeout=10.0,
        )
        r.raise_for_status()
        return float(r.json().get("markPrice", 0))

    # ── Symbol info (public, no auth) ─────────────

    def load_specs(self, symbols: list[str]) -> None:
        """Populate spec cache for the symbols we care about."""
        r = self._http.get(f"{FAPI_BASE}/fapi/v1/exchangeInfo", timeout=15.0)
        r.raise_for_status()
        for s in r.json().get("symbols", []):
            sym = s.get("symbol")
            if sym not in symbols:
                continue
            filters = {f["filterType"]: f for f in s.get("filters", [])}
            lot = filters.get("LOT_SIZE", {})
            price = filters.get("PRICE_FILTER", {})
            notional = filters.get("MIN_NOTIONAL", {})
            self._spec_cache[sym] = SymbolSpec(
                symbol=sym,
                price_precision=int(s.get("pricePrecision", 2)),
                qty_precision=int(s.get("quantityPrecision", 3)),
                step_size=float(lot.get("stepSize", 0.001)),
                min_qty=float(lot.get("minQty", 0.001)),
                tick_size=float(price.get("tickSize", 0.01)),
                min_notional=float(notional.get("notional", 0)),
            )

    def spec(self, symbol: str) -> SymbolSpec:
        if symbol not in self._spec_cache:
            self.load_specs([symbol])
        return self._spec_cache[symbol]

    # ── Configuration ─────────────────────────────

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._post("/fapi/v1/leverage", {
            "symbol": symbol, "leverage": int(leverage),
        })

    def set_isolated_margin(self, symbol: str) -> None:
        """Best-effort switch to ISOLATED margin. Swallows:
        - ``-4046`` (no change needed, already isolated)
        - ``-4175`` (account in credit state — can't toggle; stays CROSS)

        In CROSS mode the whole futures balance backs every position —
        less safe than isolated, but trading still works fine.
        """
        try:
            self._post("/fapi/v1/marginType", {
                "symbol": symbol, "marginType": "ISOLATED",
            })
        except BinanceError as e:
            if e.code not in (-4046, -4175):
                raise

    # ── Orders ─────────────────────────────────────

    def open_market(
        self,
        symbol: str,
        side: str,   # "BUY" = long entry or short close, "SELL" = short entry or long close
        qty: float,
        *,
        reduce_only: bool = False,
    ) -> dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self._format_qty(symbol, qty),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return self._post("/fapi/v1/order", params)

    def place_stop_market(
        self,
        symbol: str,
        position_side: str,   # "LONG" | "SHORT"
        stop_price: float,
    ) -> dict:
        """STOP_MARKET with closePosition=true — fires full position
        out at market when stop_price is crossed. qty is ignored by the
        exchange when closePosition=true.
        """
        exit_side = "SELL" if position_side == "LONG" else "BUY"
        return self._post("/fapi/v1/order", {
            "symbol": symbol,
            "side": exit_side,
            "type": "STOP_MARKET",
            "stopPrice": self._format_price(symbol, stop_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "timeInForce": "GTE_GTC",
        })

    def close_position(self, symbol: str) -> dict | None:
        """Flatten any existing position on ``symbol`` via MARKET
        reduce-only."""
        for p in self.get_positions():
            if p.symbol != symbol:
                continue
            exit_side = "SELL" if p.side == "LONG" else "BUY"
            return self.open_market(symbol, exit_side, p.qty, reduce_only=True)
        return None

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    # ── Quantity / price formatting ────────────────

    def _format_qty(self, symbol: str, qty: float) -> str:
        s = self.spec(symbol)
        step = s.step_size
        qty_rounded = floor(qty / step) * step
        return f"{qty_rounded:.{s.qty_precision}f}"

    def _format_price(self, symbol: str, price: float) -> str:
        s = self.spec(symbol)
        tick = s.tick_size
        price_rounded = floor(price / tick) * tick
        return f"{price_rounded:.{s.price_precision}f}"

    def compute_qty_for_notional(
        self, symbol: str, notional_usdt: float, reference_price: float,
    ) -> float:
        """Round ``notional_usdt / reference_price`` down to step_size.
        Returns 0 if the result is below ``min_qty`` or ``min_notional``.
        """
        s = self.spec(symbol)
        if reference_price <= 0:
            return 0.0
        raw = notional_usdt / reference_price
        step = s.step_size
        qty = floor(raw / step) * step
        if qty < s.min_qty:
            return 0.0
        if qty * reference_price < s.min_notional:
            return 0.0
        return qty

    @classmethod
    def from_env(cls) -> BinanceFutures:
        import os
        key = os.environ.get("BINANCE_KEY", "")
        secret = os.environ.get("BINANCE_SECRET", "")
        if not key or not secret:
            raise RuntimeError(
                "BINANCE_KEY / BINANCE_SECRET missing from environment"
            )
        return cls(key, secret)
