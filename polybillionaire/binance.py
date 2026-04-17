"""Public Binance REST client for spot klines + funding rate.

No auth required for market data. All we consume here is read-only:
- Spot candles (open/close/high/low/volume) per interval
- Perpetual funding rate (sentiment / carry)

Polymarket's crypto up-or-down markets settle against Binance's spot
USDT-pair close vs open on the matching candle, so using Binance as
the feature source keeps features aligned with the settlement source.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

SPOT_API = "https://api.binance.com"
FUTURES_API = "https://fapi.binance.com"


@dataclass
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int

    @property
    def return_pct(self) -> float:
        if self.open <= 0:
            return 0.0
        return (self.close - self.open) / self.open


@dataclass
class Snapshot:
    """Per-symbol market snapshot used by signals.py.

    All *_klines are oldest→newest. ``last_price`` is the close of the
    most recent 1m candle.
    """

    symbol: str
    last_price: float
    klines_1m: list[Kline]
    klines_5m: list[Kline]
    klines_15m: list[Kline]
    klines_1h: list[Kline]
    funding_rate: float


def fetch_snapshot(http: httpx.Client, symbol: str) -> Snapshot:
    """Fetch klines (four intervals) + funding rate for one symbol."""
    k1m = _fetch_klines(http, symbol, "1m", 60)
    k5m = _fetch_klines(http, symbol, "5m", 48)
    k15m = _fetch_klines(http, symbol, "15m", 32)
    k1h = _fetch_klines(http, symbol, "1h", 48)
    funding = _fetch_funding(http, symbol)
    last = k1m[-1].close if k1m else 0.0
    return Snapshot(
        symbol=symbol,
        last_price=last,
        klines_1m=k1m,
        klines_5m=k5m,
        klines_15m=k15m,
        klines_1h=k1h,
        funding_rate=funding,
    )


def _fetch_klines(
    http: httpx.Client, symbol: str, interval: str, limit: int,
) -> list[Kline]:
    try:
        resp = http.get(
            f"{SPOT_API}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10.0,
        )
        resp.raise_for_status()
        rows = resp.json()
    except (httpx.HTTPError, ValueError):
        return []

    out: list[Kline] = []
    for r in rows:
        try:
            out.append(Kline(
                open_time=int(r[0]),
                open=float(r[1]),
                high=float(r[2]),
                low=float(r[3]),
                close=float(r[4]),
                volume=float(r[5]),
                close_time=int(r[6]),
            ))
        except (ValueError, IndexError, TypeError):
            continue
    return out


def _fetch_funding(http: httpx.Client, symbol: str) -> float:
    try:
        resp = http.get(
            f"{FUTURES_API}/fapi/v1/premiumIndex",
            params={"symbol": symbol},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return float(data.get("lastFundingRate") or 0)
    except (httpx.HTTPError, ValueError, TypeError):
        return 0.0
