"""Polymarket Gamma API discovery for crypto up/down markets.

The Gamma REST API (https://gamma-api.polymarket.com) publishes market
metadata. Up-or-down crypto markets are one-market-per-event, so we
query ``/events`` (tag_id=21 = Crypto) and pull the nested market for
each.

Settlement: Binance USDT-pair spot close vs open on the matching short
candle (5m / 15m / 1h). Close >= open → Up wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .crossvenue import PolymarketMarketMin

GAMMA_API = "https://gamma-api.polymarket.com"
CRYPTO_TAG_ID = 21

#: Hard cap on the number of markets returned by fetch_all_open_markets.
#: Polymarket has ~10-20k open markets at any time; scanning all is both
#: slow and useless — almost none will close in the near term. Caller
#: should set ``max_seconds_until_end`` to bound the scan window.
_MAX_MARKETS_SCAN = 5000

#: Polymarket slug token → Binance USDT-pair symbol. HYPE is excluded —
#: no Binance USDT pair at this writing.
_SYMBOL_MAP: dict[str, str] = {
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "eth": "ETHUSDT",
    "solana": "SOLUSDT",
    "sol": "SOLUSDT",
    "xrp": "XRPUSDT",
    "bnb": "BNBUSDT",
    "dogecoin": "DOGEUSDT",
    "doge": "DOGEUSDT",
}


@dataclass
class UpDownMarket:
    """A short-duration crypto directional market on Polymarket."""

    condition_id: str
    slug: str
    question: str
    symbol: str
    up_token_id: str
    down_token_id: str
    up_price: float
    end_time: datetime
    duration_s: float
    volume: float
    liquidity: float

    @property
    def seconds_until_end(self) -> float:
        return (self.end_time - datetime.now(timezone.utc)).total_seconds()

    @property
    def duration_label(self) -> str:
        s = self.duration_s
        if s <= 7 * 60:
            return "5m"
        if s <= 18 * 60:
            return "15m"
        if s <= 70 * 60:
            return "1h"
        return f"{int(s / 60)}m"


def fetch_updown_markets(
    http: httpx.Client,
    *,
    max_seconds_until_end: int = 3600,
    min_seconds_until_end: int = 300,
) -> list[UpDownMarket]:
    """Fetch live up-or-down crypto markets ending within the window.

    ``min_seconds_until_end`` defaults to 300s — Polymarket halts
    trading ~1–5 min before candle close on short markets, so there's
    no point returning markets inside that halt window.
    """
    params = {
        "active": "true",
        "closed": "false",
        "tag_id": CRYPTO_TAG_ID,
        "order": "endDate",
        "ascending": "true",
        "limit": 500,
    }
    resp = http.get(f"{GAMMA_API}/events", params=params, timeout=15.0)
    resp.raise_for_status()
    events = resp.json()

    now = datetime.now(timezone.utc)
    out: list[UpDownMarket] = []
    for ev in events:
        slug = (ev.get("slug") or "").lower()
        if "updown" not in slug and "up-or-down" not in slug:
            continue

        symbol = _symbol_from_slug(slug)
        if symbol is None:
            continue

        end = _parse_iso(ev.get("endDate") or "")
        if end is None:
            continue
        sec_remaining = (end - now).total_seconds()
        if sec_remaining <= min_seconds_until_end:
            continue
        if sec_remaining > max_seconds_until_end:
            continue

        duration_s = _duration_from_slug(slug)
        if duration_s <= 0 or duration_s > 3600:
            continue  # unrecognized or longer than 1h — out of scope

        # Each up-or-down event has exactly one market (the binary bet).
        markets = ev.get("markets") or []
        if not markets:
            continue
        m = markets[0]

        tokens = _coerce_json_list(m.get("clobTokenIds"))
        outcomes = _coerce_json_list(m.get("outcomes"))
        if len(tokens) < 2 or len(outcomes) < 2:
            continue

        up_tid, down_tid = _align_up_down(tokens, outcomes)
        if up_tid is None or down_tid is None:
            continue

        up_price = _coerce_up_price(m, outcomes)

        out.append(UpDownMarket(
            condition_id=m.get("conditionId") or m.get("condition_id") or "",
            slug=slug,
            question=ev.get("title") or m.get("question", ""),
            symbol=symbol,
            up_token_id=up_tid,
            down_token_id=down_tid,
            up_price=up_price,
            end_time=end,
            duration_s=duration_s,
            volume=float(m.get("volume") or ev.get("volume") or 0),
            liquidity=float(m.get("liquidity") or ev.get("liquidity") or 0),
        ))
    return out


def fetch_all_open_markets(
    http: httpx.Client,
    *,
    max_seconds_until_end: int = 7 * 24 * 3600,
    min_seconds_until_end: int = 300,
    page_size: int = 500,
    max_total: int = _MAX_MARKETS_SCAN,
) -> list[PolymarketMarketMin]:
    """Fetch all active binary YES/NO Polymarket markets closing within
    the time window.

    Unlike ``fetch_updown_markets`` (crypto-only, tag 21), this scans
    every category so we can cross-match against Kalshi's full surface.

    Pagination: Gamma ``/markets`` returns up to 500 per page; we
    paginate via ``offset`` until the end-date window is exhausted or we
    hit ``max_total``.

    Returns ``PolymarketMarketMin`` (crossvenue's lightweight interface),
    not ``UpDownMarket`` — cross-venue matching doesn't need the up/down
    token IDs.
    """
    now = datetime.now(timezone.utc)
    out: list[PolymarketMarketMin] = []
    offset = 0
    max_off = max_total
    while offset < max_off:
        params = {
            "active": "true",
            "closed": "false",
            "order": "endDate",
            "ascending": "true",
            "limit": page_size,
            "offset": offset,
        }
        r = http.get(f"{GAMMA_API}/markets", params=params, timeout=20.0)
        r.raise_for_status()
        page = r.json() or []
        if not page:
            break

        page_hit_window_end = False
        for m in page:
            end = _parse_iso(m.get("endDate") or m.get("endDateIso") or "")
            if end is None:
                continue
            sec = (end - now).total_seconds()
            if sec <= min_seconds_until_end:
                continue
            if sec > max_seconds_until_end:
                # Endings are sorted ascending, so once we overshoot the
                # window we can stop scanning this page and all future pages.
                page_hit_window_end = True
                break

            if not m.get("enableOrderBook"):
                continue

            outcomes = _coerce_json_list(m.get("outcomes"))
            prices = _coerce_json_list(m.get("outcomePrices"))
            yes_price = _binary_yes_price(outcomes, prices)
            if yes_price is None:
                continue

            out.append(PolymarketMarketMin(
                condition_id=m.get("conditionId") or "",
                slug=(m.get("slug") or "").lower(),
                question=m.get("question") or "",
                yes_price=yes_price,
                end_time=end,
                volume=float(m.get("volumeNum") or m.get("volume") or 0),
                raw=m,
            ))
            if len(out) >= max_total:
                return out

        if page_hit_window_end:
            break
        if len(page) < page_size:
            break
        offset += page_size
    return out


def _binary_yes_price(outcomes: list, prices: list) -> float | None:
    """Return the YES (or Up) price if this is a binary market, else None.

    Polymarket multi-outcome events come as separate binary markets per
    candidate, so most ``/markets`` rows are already binary. We still
    guard against degenerate 1- or 3-outcome rows.
    """
    if len(outcomes) != 2 or len(prices) != 2:
        return None
    labels = [str(o).strip().lower() for o in outcomes]
    # Expect {"yes","no"} or {"up","down"}.
    if "yes" in labels:
        idx = labels.index("yes")
    elif "up" in labels:
        idx = labels.index("up")
    else:
        return None
    try:
        p = float(prices[idx])
    except (TypeError, ValueError):
        return None
    if p <= 0.0 or p >= 1.0:
        return None
    return p


def _symbol_from_slug(slug: str) -> str | None:
    for prefix, sym in _SYMBOL_MAP.items():
        if slug.startswith(prefix + "-"):
            return sym
    return None


def _duration_from_slug(slug: str) -> float:
    """Polymarket encodes the settlement candle length in the slug.

    Returns -1.0 for unrecognized patterns — caller should skip those.
    """
    if "-5m-" in slug or slug.endswith("-5m"):
        return 5 * 60
    if "-15m-" in slug or slug.endswith("-15m"):
        return 15 * 60
    if "-1h-" in slug or slug.endswith("-1h"):
        return 60 * 60
    if "-4h-" in slug or slug.endswith("-4h"):
        return 4 * 3600
    # human-readable hourly slugs: "...april-16-2026-3pm-et"
    if "pm-et" in slug or "am-et" in slug:
        return 60 * 60
    return -1.0


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_json_list(v: object) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return []
    return []


def _align_up_down(
    tokens: list, outcomes: list,
) -> tuple[str | None, str | None]:
    """Return (up_token_id, down_token_id) by matching outcome labels."""
    up_tid: str | None = None
    down_tid: str | None = None
    for tid, label in zip(tokens, outcomes):
        lab = str(label).strip().lower()
        if lab in {"up", "yes"}:
            up_tid = str(tid)
        elif lab in {"down", "no"}:
            down_tid = str(tid)
    return up_tid, down_tid


def _coerce_up_price(m: dict, outcomes: list) -> float:
    prices = _coerce_json_list(m.get("outcomePrices"))
    for price, label in zip(prices, outcomes):
        if str(label).strip().lower() in {"up", "yes"}:
            try:
                return float(price)
            except (ValueError, TypeError):
                break
    try:
        return float(m.get("lastTradePrice") or 0.5)
    except (ValueError, TypeError):
        return 0.5
