"""Polymarket API client for market data and trading."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

DEFAULT_TIMEOUT = 15.0


@dataclass
class Market:
    condition_id: str
    question: str
    slug: str
    active: bool
    closed: bool
    tokens: list[dict[str, Any]]
    volume: float
    liquidity: float
    end_date: str
    description: str = ""
    category: str = ""
    outcomes: list[str] = field(default_factory=list)
    best_bid: float = 0.0
    best_ask: float = 1.0
    last_trade_price: float = 0.0
    spread: float = 0.0
    one_day_change: float = 0.0
    one_week_change: float = 0.0
    volume_24h: float = 0.0
    outcome_prices: list[float] = field(default_factory=list)

    @property
    def yes_token_id(self) -> str | None:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "yes":
                return t.get("token_id")
        return self.tokens[0]["token_id"] if self.tokens else None

    @property
    def no_token_id(self) -> str | None:
        for t in self.tokens:
            if t.get("outcome", "").lower() == "no":
                return t.get("token_id")
        return self.tokens[1]["token_id"] if len(self.tokens) > 1 else None


@dataclass
class OrderBook:
    token_id: str
    bids: list[dict[str, float]]
    asks: list[dict[str, float]]
    spread: float
    midpoint: float
    best_bid: float
    best_ask: float


@dataclass
class PricePoint:
    timestamp: int
    price: float


class PolymarketClient:
    """Unified client for Polymarket's public APIs."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._http = httpx.Client(timeout=timeout)
        self._rate_limit_until = 0.0

    def _wait_rate_limit(self) -> None:
        now = time.time()
        if now < self._rate_limit_until:
            time.sleep(self._rate_limit_until - now)

    def _get(self, base: str, path: str, params: dict | None = None) -> Any:
        self._wait_rate_limit()
        resp = self._http.get(f"{base}{path}", params=params or {})
        if resp.status_code == 429:
            self._rate_limit_until = time.time() + 2.0
            time.sleep(2.0)
            return self._get(base, path, params)
        resp.raise_for_status()
        return resp.json()

    # ── Gamma API (market metadata) ──────────────────────────────

    def get_markets(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
        end_date_max: str | None = None,
    ) -> list[Market]:
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if end_date_max:
            params["end_date_max"] = end_date_max
        data = self._get(GAMMA_API, "/markets", params)
        return [self._parse_market(m) for m in data if m.get("enableOrderBook")]

    def get_daily_markets(self, *, hours: int = 24, limit: int = 200) -> list[Market]:
        """Fetch active markets resolving within the next `hours` hours."""
        now = datetime.now(timezone.utc)
        end_max = now.replace(hour=23, minute=59, second=59) if hours <= 24 else None

        # Fetch a large batch and filter client-side by end_date
        markets = self.get_markets(limit=limit, order="volume24hr")
        daily: list[Market] = []
        for m in markets:
            if not m.end_date:
                continue
            try:
                end = datetime.fromisoformat(m.end_date.replace("Z", "+00:00"))
            except ValueError:
                continue
            delta_hours = (end - now).total_seconds() / 3600
            if 0 < delta_hours <= hours:
                daily.append(m)
        return daily

    def search_markets(self, query: str, *, limit: int = 20) -> list[Market]:
        """Search markets by keyword (client-side filter over active markets)."""
        query_lower = query.lower()
        data = self._get(GAMMA_API, "/markets", {
            "limit": 200,
            "active": "true",
            "closed": "false",
            "order": "volume24hr",
            "ascending": "false",
        })
        matches = [
            m for m in data
            if m.get("enableOrderBook")
            and query_lower in (m.get("question", "") + " " + m.get("description", "")).lower()
        ]
        return [self._parse_market(m) for m in matches[:limit]]

    def get_events(self, *, limit: int = 50, active: bool = True) -> list[dict]:
        params = {"limit": limit, "active": str(active).lower()}
        return self._get(GAMMA_API, "/events", params)

    # ── CLOB API (orderbook / pricing) ───────────────────────────

    def get_orderbook(self, token_id: str) -> OrderBook:
        data = self._get(CLOB_API, f"/book", params={"token_id": token_id})
        bids = sorted(
            [{"price": float(o["price"]), "size": float(o["size"])} for o in data.get("bids", [])],
            key=lambda x: x["price"],
            reverse=True,  # Best (highest) bid first
        )
        asks = sorted(
            [{"price": float(o["price"]), "size": float(o["size"])} for o in data.get("asks", [])],
            key=lambda x: x["price"],
        )  # Best (lowest) ask first
        best_bid = bids[0]["price"] if bids else 0.0
        best_ask = asks[0]["price"] if asks else 1.0
        return OrderBook(
            token_id=token_id,
            bids=bids,
            asks=asks,
            spread=round(best_ask - best_bid, 4),
            midpoint=round((best_bid + best_ask) / 2, 4),
            best_bid=best_bid,
            best_ask=best_ask,
        )

    def get_price(self, token_id: str) -> dict[str, float]:
        return self._get(CLOB_API, "/price", params={"token_id": token_id})

    def get_midpoint(self, token_id: str) -> float:
        data = self._get(CLOB_API, f"/midpoint", params={"token_id": token_id})
        return float(data.get("mid", 0.0))

    def get_spread(self, token_id: str) -> float:
        data = self._get(CLOB_API, f"/spread", params={"token_id": token_id})
        return float(data.get("spread", 0.0))

    def get_price_history(
        self, token_id: str, *, fidelity: int = 60
    ) -> list[PricePoint]:
        data = self._get(
            CLOB_API,
            f"/prices-history",
            params={"market": token_id, "interval": "max", "fidelity": fidelity},
        )
        return [
            PricePoint(timestamp=int(p["t"]), price=float(p["p"]))
            for p in data.get("history", [])
        ]

    def get_last_trade(self, token_id: str) -> dict:
        return self._get(CLOB_API, f"/last-trade-price", params={"token_id": token_id})

    def get_trades(self, token_id: str, *, limit: int = 50) -> list[dict]:
        data = self._get(
            CLOB_API, f"/trades", params={"market": token_id, "limit": limit}
        )
        return data if isinstance(data, list) else data.get("trades", data.get("data", []))

    # ── Data API (positions, activity) ───────────────────────────

    def get_positions(self, address: str) -> list[dict]:
        return self._get(DATA_API, f"/positions", params={"user": address})

    def get_profit_loss(self, address: str) -> dict:
        return self._get(DATA_API, f"/pnl", params={"user": address})

    # ── Market resolution ────────────────────────────────────────

    def is_market_resolved(self, token_id: str) -> tuple[bool, float | None]:
        """Check if a market has resolved by probing its orderbook.

        Returns (resolved, last_known_price).
        A resolved market has no orderbook — the CLOB API returns 404
        or an error for midpoint/book queries.
        """
        self._wait_rate_limit()
        try:
            resp = self._http.get(
                f"{CLOB_API}/midpoint",
                params={"token_id": token_id},
            )
            if resp.status_code == 404:
                return True, None  # No orderbook = resolved
            if resp.status_code == 429:
                self._rate_limit_until = time.time() + 2.0
                return False, None  # Rate limited — don't assume resolved
            resp.raise_for_status()
            mid = float(resp.json().get("mid", 0))
            if mid > 0:
                return False, mid  # Still active
            # mid == 0 is ambiguous — could be resolved or just illiquid
            return False, None
        except httpx.HTTPStatusError:
            return True, None  # Server says no — likely resolved
        except Exception:
            return False, None  # Network error — don't assume resolved

    # ── Helpers ───────────────────────────────────────────────────

    def _parse_market(self, raw: dict) -> Market:
        import json as _json

        tokens = raw.get("clobTokenIds", [])
        if isinstance(tokens, str):
            try:
                tokens = _json.loads(tokens)
            except (ValueError, TypeError):
                tokens = []

        outcomes = raw.get("outcomes", [])
        if isinstance(outcomes, str):
            try:
                outcomes = _json.loads(outcomes)
            except (ValueError, TypeError):
                outcomes = []

        token_list = []
        for i, tid in enumerate(tokens if isinstance(tokens, list) else []):
            token_list.append({
                "token_id": tid,
                "outcome": outcomes[i] if i < len(outcomes) else f"outcome_{i}",
            })

        outcome_prices_raw = raw.get("outcomePrices", [])
        if isinstance(outcome_prices_raw, str):
            try:
                outcome_prices_raw = _json.loads(outcome_prices_raw)
            except (ValueError, TypeError):
                outcome_prices_raw = []
        outcome_prices = [float(p) for p in outcome_prices_raw if p is not None]

        return Market(
            condition_id=raw.get("conditionId", raw.get("condition_id", "")),
            question=raw.get("question", ""),
            slug=raw.get("slug", ""),
            active=raw.get("active", False),
            closed=raw.get("closed", False),
            tokens=token_list,
            volume=float(raw.get("volume", 0) or 0),
            liquidity=float(raw.get("liquidity", 0) or 0),
            end_date=raw.get("endDate", raw.get("end_date_iso", "")),
            description=raw.get("description", ""),
            category=raw.get("category", ""),
            outcomes=outcomes if isinstance(outcomes, list) else [],
            best_bid=float(raw.get("bestBid") or 0),
            best_ask=float(raw.get("bestAsk") or 1),
            last_trade_price=float(raw.get("lastTradePrice") or 0),
            spread=float(raw.get("spread") or 0),
            one_day_change=float(raw.get("oneDayPriceChange") or 0),
            one_week_change=float(raw.get("oneWeekPriceChange") or 0),
            volume_24h=float(raw.get("volume24hr") or 0),
            outcome_prices=outcome_prices,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> PolymarketClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
