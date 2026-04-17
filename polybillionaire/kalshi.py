"""Kalshi Trading API v2 client — public market-data endpoints only.

Kalshi is a CFTC-regulated prediction market (US-incorporated, ~140
countries now). Public market data (`/events`, `/markets`,
`/markets/{ticker}/orderbook`) does **not** require authentication.
Order placement does — it uses RSA-PSS signed requests — but v1 of
this client is read-only for cross-venue price comparison against
Polymarket.

Base URL: ``https://api.elections.kalshi.com/trade-api/v2`` (despite
the name, this serves all Kalshi markets, not just elections).

Fee math (2026-04, per kalshi.com/fee-schedule):
    taker_fee_cents = ceil(0.07 * contracts * P * (1 - P))
At P=0.50 that's 1.75c per contract = 3.50% round-trip. At P=0.10 or
P=0.90 it's ~0.63c per contract = 1.26% round-trip. Cheaper at the
tails than we assumed in the original research.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx

BASE = "https://api.elections.kalshi.com/trade-api/v2"

#: Minimum 24h volume (in contracts) for a market to be considered
#: "liquid enough" for arb. Kalshi has ~700k active markets at any
#: time, but the overwhelming majority are MVE parlay markets
#: (KXMVESPORTSMULTIGAMEEXTENDED, KXMVECROSSCATEGORY) with zero volume.
#: 100 contracts / 24h is a cheap filter that drops those; real
#: tradeable events clear this by a wide margin.
LIQUID_MIN_VOLUME_24H = 100.0

#: Curated whitelist of Kalshi series that consistently carry volume
#: (observed 2026-04-17 — top series by 24h volume). Used as the fast
#: path in ``fetch_liquid_markets``: per-series queries return <200
#: markets each, while an unfiltered walk pages through ~50k MVE
#: parlay markets first. Update periodically as Kalshi's product mix
#: shifts.
LIQUID_SERIES_DEFAULT: tuple[str, ...] = (
    # Sports (highest turnover)
    "KXPGATOUR", "KXMLBGAME", "KXMLBTOTAL", "KXMLBSPREAD", "KXMLBHR", "KXMLBKS",
    "KXNBAGAME", "KXNBASERIES", "KXNBACHAMP", "KXNHLGAME", "KXNHLSERIES",
    "KXNFLGAME", "KXIPLGAME", "KXSOCCERGAME",
    # Crypto (daily + intraday)
    "KXBTC", "KXBTCD", "KXBTCMAX", "KXETHD", "KXSOL", "KXHYPE",
    # Macro / economy
    "KXWTI", "KXCPIYOY", "KXFED", "KXRECESSION", "KXUNEMP",
    "KXSP500", "KXNASDAQ",
    # Politics / news
    "KXTRUMPMENTION", "KXTRUMPSAY", "KXPRES", "KXMAYORNYC",
    "KXSENATE", "KXHOUSE", "KXGOV", "KXCONGRESS",
    # Entertainment
    "KXOSCARS", "KXEMMY", "KXSUPERBOWL",
)


@dataclass
class KalshiMarket:
    """A single Kalshi binary YES/NO market.

    ``yes_bid`` / ``yes_ask`` are in dollars (0.0–1.0). If the top of
    book is missing on a side, the corresponding value is 0 or 1 as
    appropriate (see ``_from_api``).
    """

    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str
    status: str              # "unopened" | "open" | "closed" | "settled"
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    last_price: float
    volume_24h: float
    open_time: datetime
    close_time: datetime
    rules_primary: str
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def mid(self) -> float:
        """Best estimate of YES mid price."""
        if self.yes_ask > 0 and self.yes_bid > 0:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last_price

    @property
    def seconds_until_close(self) -> float:
        return (self.close_time - datetime.now(timezone.utc)).total_seconds()


@dataclass
class KalshiOrderBook:
    """Level-2 book. Kalshi only returns bids — YES asks are derived
    from NO bids (``yes_ask = 1 - no_best_bid``)."""

    ticker: str
    yes_bids: list[tuple[float, float]]  # (price_usd, qty) descending
    no_bids: list[tuple[float, float]]

    @property
    def yes_best_bid(self) -> float:
        return self.yes_bids[0][0] if self.yes_bids else 0.0

    @property
    def yes_best_ask(self) -> float:
        """Derived from the best NO bid. If no NO bids, asks are at 1.0."""
        return 1.0 - self.no_bids[0][0] if self.no_bids else 1.0

    @property
    def mid(self) -> float:
        b, a = self.yes_best_bid, self.yes_best_ask
        if b > 0 and a < 1.0:
            return (b + a) / 2
        return b if b > 0 else (a if a < 1.0 else 0.5)


class KalshiClient:
    """Read-only public-data Kalshi client.

    Parameters
    ----------
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(self, timeout: float = 10.0) -> None:
        self._http = httpx.Client(
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    # ── Raw HTTP ─────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict:
        # Kalshi's public API tier rate-limits aggressively; back off on 429.
        delay = 0.5
        for _ in range(5):
            r = self._http.get(f"{BASE}{path}", params=params)
            if r.status_code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 8.0)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    # ── Events ───────────────────────────────────────

    def get_events(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        with_nested_markets: bool = False,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict:
        """Fetch a page of events. Returns the raw response containing
        ``events`` and ``cursor`` (for pagination).
        """
        params: dict = {
            "status": status,
            "limit": min(limit, 200),
            "with_nested_markets": str(with_nested_markets).lower(),
        }
        if series_ticker:
            params["series_ticker"] = series_ticker
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if cursor:
            params["cursor"] = cursor
        return self._get("/events", params)

    def iter_events(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        with_nested_markets: bool = False,
        page_size: int = 200,
    ):
        """Generator that paginates through all events matching the filters."""
        cursor: str | None = None
        while True:
            resp = self.get_events(
                status=status,
                series_ticker=series_ticker,
                min_close_ts=min_close_ts,
                max_close_ts=max_close_ts,
                with_nested_markets=with_nested_markets,
                limit=page_size,
                cursor=cursor,
            )
            for ev in resp.get("events", []):
                yield ev
            cursor = resp.get("cursor")
            if not cursor:
                return

    # ── Markets ──────────────────────────────────────

    def get_markets(
        self,
        *,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str = "open",
        tickers: list[str] | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> list[KalshiMarket]:
        params: dict = {"status": status, "limit": min(limit, 1000)}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if tickers:
            params["tickers"] = ",".join(tickers)
        if min_close_ts:
            params["min_close_ts"] = min_close_ts
        if max_close_ts:
            params["max_close_ts"] = max_close_ts
        if cursor:
            params["cursor"] = cursor
        resp = self._get("/markets", params)
        return [_market_from_api(m) for m in resp.get("markets", [])]

    def get_market(self, ticker: str) -> KalshiMarket:
        resp = self._get(f"/markets/{ticker}")
        return _market_from_api(resp["market"])

    # ── Orderbook ────────────────────────────────────

    def iter_markets(
        self,
        *,
        status: str = "open",
        series_ticker: str | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        page_size: int = 1000,
    ):
        """Generator paginating ``/markets`` via cursor until exhausted."""
        cursor: str | None = None
        while True:
            params: dict = {"status": status, "limit": min(page_size, 1000)}
            if series_ticker:
                params["series_ticker"] = series_ticker
            if min_close_ts:
                params["min_close_ts"] = min_close_ts
            if max_close_ts:
                params["max_close_ts"] = max_close_ts
            if cursor:
                params["cursor"] = cursor
            resp = self._get("/markets", params)
            for m in resp.get("markets", []):
                yield _market_from_api(m)
            cursor = resp.get("cursor")
            if not cursor:
                return

    def fetch_liquid_markets(
        self,
        *,
        min_volume_24h: float = LIQUID_MIN_VOLUME_24H,
        series_tickers: tuple[str, ...] | list[str] | None = LIQUID_SERIES_DEFAULT,
        max_close_ts: int | None = None,
        min_close_ts: int | None = None,
    ) -> list[KalshiMarket]:
        """Return open markets above a volume floor.

        Defaults to the ``LIQUID_SERIES_DEFAULT`` whitelist — each
        series is small (<200 markets) and returns fast. Pass
        ``series_tickers=None`` to walk the entire open-market set via
        cursor pagination; this is ~75s on 2026-04 because the unfiltered
        /markets page is dominated by zero-volume MVE parlay markets.

        ``max_close_ts`` is an epoch-second cap; use it with a broad
        walk to narrow by close-time and skip past-expiry scanning.
        """
        out: list[KalshiMarket] = []
        if series_tickers:
            for st in series_tickers:
                for m in self.iter_markets(
                    status="open",
                    series_ticker=st,
                    min_close_ts=min_close_ts,
                    max_close_ts=max_close_ts,
                ):
                    if m.volume_24h >= min_volume_24h:
                        out.append(m)
            return out

        for m in self.iter_markets(
            status="open",
            min_close_ts=min_close_ts,
            max_close_ts=max_close_ts,
        ):
            if m.volume_24h >= min_volume_24h:
                out.append(m)
        return out

    def get_orderbook(self, ticker: str, *, depth: int = 0) -> KalshiOrderBook:
        """Fetch the order book for a single market.

        Kalshi returns prices as integer cents by default. We normalise to
        dollars (0.0–1.0) for consistency with Polymarket.
        """
        params = {}
        if depth > 0:
            params["depth"] = depth
        resp = self._get(f"/markets/{ticker}/orderbook", params or None)
        ob = resp.get("orderbook", {}) or {}
        yes_dollars = ob.get("yes") or ob.get("yes_dollars") or []
        no_dollars = ob.get("no") or ob.get("no_dollars") or []
        return KalshiOrderBook(
            ticker=ticker,
            yes_bids=[_normalise_level(lvl) for lvl in yes_dollars],
            no_bids=[_normalise_level(lvl) for lvl in no_dollars],
        )


# ── Fee math ─────────────────────────────────────────

def kalshi_taker_fee_cents(price: float, contracts: int = 1) -> int:
    """Fee per-side in cents (per kalshi.com/fee-schedule 2026-04).

    ``ceil(0.07 * contracts * P * (1 - P))``. Note: same formula for
    YES and NO sides; P is the price of the contract being taken.
    """
    return max(1, math.ceil(0.07 * contracts * price * (1 - price)))


def kalshi_round_trip_cost_pct(entry_price: float, exit_price: float) -> float:
    """Fraction of $1 notional consumed by fees for a round-trip.

    Useful for arb spread filtering: ``gross_edge_pct > kalshi_rt + poly_rt``.
    """
    entry_fee = kalshi_taker_fee_cents(entry_price) / 100.0
    exit_fee = kalshi_taker_fee_cents(exit_price) / 100.0
    return entry_fee + exit_fee


# ── Internal helpers ─────────────────────────────────

def _market_from_api(m: dict) -> KalshiMarket:
    def _dollars(v) -> float:
        """Kalshi now returns prices in dollars directly (``*_dollars`` fields).
        Fallback to cents for legacy keys (``*_bid`` etc.)."""
        if v is None:
            return 0.0
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return f / 100.0 if f > 1.0 else f

    def _fp(v) -> float:
        """``_fp``-suffixed fields are decimal strings (contract counts
        or notional), *not* fixed-point integers — verified against
        KXHIGHNY-26APR17-B77.5 on 2026-04-17 (volume_24h_fp='8854.61'
        matched the trading UI directly, no scaling needed)."""
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    return KalshiMarket(
        ticker=m.get("ticker", ""),
        event_ticker=m.get("event_ticker", ""),
        series_ticker=(m.get("series_ticker")
                       or m.get("event_ticker", "").rsplit("-", 1)[0]),
        title=m.get("title", ""),
        subtitle=(m.get("yes_sub_title")
                  or m.get("subtitle")
                  or m.get("sub_title", "")),
        status=m.get("status", ""),
        yes_bid=_dollars(m.get("yes_bid_dollars", m.get("yes_bid"))),
        yes_ask=_dollars(m.get("yes_ask_dollars", m.get("yes_ask"))),
        no_bid=_dollars(m.get("no_bid_dollars", m.get("no_bid"))),
        no_ask=_dollars(m.get("no_ask_dollars", m.get("no_ask"))),
        last_price=_dollars(m.get("last_price_dollars", m.get("last_price"))),
        volume_24h=_fp(m.get("volume_24h_fp")) or float(m.get("volume_24h") or 0),
        open_time=_parse_ts(m.get("open_time")),
        close_time=_parse_ts(m.get("close_time")),
        rules_primary=m.get("rules_primary", ""),
        raw=m,
    )


def _normalise_level(lvl) -> tuple[float, float]:
    """Kalshi book levels come as ``[price_cents, qty]``. Return
    (price_dollars, qty)."""
    if not lvl or len(lvl) < 2:
        return (0.0, 0.0)
    price, qty = lvl[0], lvl[1]
    try:
        price_f = float(price)
        if price_f > 1.0:  # cents
            price_f /= 100.0
        return (price_f, float(qty))
    except (TypeError, ValueError):
        return (0.0, 0.0)


def _parse_ts(s: str | None) -> datetime:
    if not s:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
