"""Probability model for crypto up-or-down candle resolution.

Settlement rule: the Polymarket market wins ``Up`` if the Binance
USDT-pair close is ``>= open`` on the matching candle. So the model
computes P(final_close >= candle_open | current market state).

Core math — normal approximation:
    remaining_return ~ N(drift * frac, sigma_bar * sqrt(frac))
    P(final_return >= 0) = P(remaining_return >= -current_return)
                         = Phi((current_return + drift_term) / sigma_remaining)

This is the "fair" baseline a well-run market should approximately
price. Our only source of edge is (a) tighter sigma estimate from
recent candles, (b) a small drift prior from longer-horizon momentum
and funding, and (c) reacting to current_price shifts faster than the
order book does. Expect tiny raw edge — the hourly loop then requires
a threshold to account for fees + slippage before trading.

All drift coefficients here are priors, not backtested. Tune after
collecting live data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import stdev

from .binance import Kline, Snapshot


@dataclass
class Prediction:
    p_up: float
    current_return: float
    sigma_bar: float
    sigma_remaining: float
    drift: float
    bar_interval_s: int


def predict_up_probability(
    snapshot: Snapshot,
    *,
    duration_s: float,
    seconds_remaining: float,
) -> Prediction | None:
    """Return P(settlement candle close >= open).

    ``None`` if inputs are too degenerate to model (missing klines,
    zero vol, market already resolved, etc.) — caller should skip.
    """
    bars, bar_s = _pick_settlement_series(snapshot, duration_s)
    if not bars:
        return None

    sigma_bar = _realized_vol(bars, n_recent=24)
    # Vol floor: during quiet recent periods, a tiny stdev would make
    # us catastrophically overconfident. The floor scales with bar
    # length so it matches typical BTC realized vol at any horizon.
    sigma_floor = _sigma_floor_for(bar_s)
    sigma_bar = max(sigma_bar, sigma_floor)
    if sigma_bar <= 0:
        return None

    # Time math: the settlement candle is [end - bar_s, end]. If now is
    # earlier than the candle's open, current-price has no bearing on
    # the eventual open-to-close return.
    full_bar_drift = _momentum_drift(snapshot, bar_s)

    if seconds_remaining > bar_s:
        # Pre-candle: prior is ~50% + tiny drift over the full candle.
        z = full_bar_drift / sigma_bar
        p = max(min(_phi(z), 0.999), 0.001)
        return Prediction(p, 0.0, sigma_bar, sigma_bar, full_bar_drift, bar_s)

    if seconds_remaining <= 0:
        # Past the close — degenerate, shouldn't be traded.
        current_bar = bars[-1]
        if current_bar.open <= 0:
            return None
        current_price = snapshot.last_price or current_bar.close
        current_return = (current_price - current_bar.open) / current_bar.open
        p = 1.0 if current_return >= 0 else 0.0
        return Prediction(p, current_return, sigma_bar, 0.0, 0.0, bar_s)

    current_bar = bars[-1]
    if current_bar.open <= 0:
        return None
    current_price = snapshot.last_price or current_bar.close
    current_return = (current_price - current_bar.open) / current_bar.open

    frac_remaining = seconds_remaining / bar_s
    sigma_remaining = sigma_bar * math.sqrt(frac_remaining)
    drift = full_bar_drift * frac_remaining

    if sigma_remaining <= 1e-9:
        p = 1.0 if (current_return + drift) >= 0 else 0.0
    else:
        z = (current_return + drift) / sigma_remaining
        p = _phi(z)

    # Clamp to sane range (numerical safety, no other reason)
    p = max(min(p, 0.999), 0.001)

    return Prediction(
        p_up=p,
        current_return=current_return,
        sigma_bar=sigma_bar,
        sigma_remaining=sigma_remaining,
        drift=drift,
        bar_interval_s=bar_s,
    )


def _pick_settlement_series(
    snapshot: Snapshot, duration_s: float,
) -> tuple[list[Kline], int]:
    if duration_s <= 7 * 60:
        return snapshot.klines_5m, 5 * 60
    if duration_s <= 18 * 60:
        return snapshot.klines_15m, 15 * 60
    return snapshot.klines_1h, 60 * 60


def _sigma_floor_for(bar_seconds: int) -> float:
    """Per-bar sigma floor. Roughly matches typical BTC realized vol
    sqrt-scaled from a daily ~3% baseline: sigma_bar ≈ 3% × sqrt(bar/day)."""
    day = 86400.0
    return 0.03 * (bar_seconds / day) ** 0.5


def _realized_vol(klines: list[Kline], *, n_recent: int) -> float:
    returns = [k.return_pct for k in klines[-n_recent:] if k.open > 0]
    # stdev needs at least 2 points; return a conservative default
    # when history is too thin.
    if len(returns) < 3:
        return 0.01
    try:
        return stdev(returns)
    except Exception:
        return 0.01


def _momentum_drift(snapshot: Snapshot, bar_seconds: float) -> float:
    """Weak directional prior — expected drift over one bar of length
    ``bar_seconds``, in absolute-return units.

    Combines two rate signals (both heavily shrunk toward zero):

    1. 6-hour net return — a trend-continuation prior. Crypto has
       weak positive autocorrelation at the 4–24h horizon. We believe
       only 10% of the observed trend carries forward.

    2. Funding rate — a contrarian prior (8h funding). Extreme positive
       funding is a weak short signal.

    Both coefficients are priors, not fitted.
    """
    momentum_rate = 0.0
    hours = snapshot.klines_1h
    if len(hours) >= 6 and hours[-6].open > 0:
        net_6h = (hours[-1].close - hours[-6].open) / hours[-6].open
        net_6h = max(min(net_6h, 0.05), -0.05)
        momentum_rate = 0.1 * net_6h / (6 * 3600)

    funding = max(min(snapshot.funding_rate, 0.001), -0.001)
    funding_rate = -0.5 * funding / (8 * 3600)

    return (momentum_rate + funding_rate) * bar_seconds


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
