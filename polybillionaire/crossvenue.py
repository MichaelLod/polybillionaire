"""Cross-venue arbitrage detection: Kalshi ↔ Polymarket.

v1 scope: fetch open markets on both venues, fuzzy-match events that
appear on both, compute post-fee spread, alert on ≥ 7% opportunities.

Research basis (2026-04): Polymarket's dynamic taker fee (≈3.15% at
P=0.50) + Kalshi's taker fee (~1.75c per contract = 3.5% at P=0.50)
means the combined round-trip floor is ~6.65%. The 7% filter gates
for a real edge after fees. See ``HANDOVER.md`` research notes and
``kalshi.py`` fee helpers for the math.

EU caveat: execution on Kalshi from the EU is in a regulatory gray
zone. A saner v1 uses Kalshi **as a reference price feed only** —
execute on Polymarket when Polymarket is the mispriced leg, skip the
opportunity when Kalshi is. This module tags every opportunity with
which leg is mispriced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from .kalshi import KalshiClient, KalshiMarket, kalshi_round_trip_cost_pct
from .llm_match import score_pairs as llm_score_pairs

#: Common English words that carry no matching signal. We require at
#: least one *other* shared token between titles — without this, fuzzy
#: ratio alone pairs "Will Bitcoin be above $62,000" with "Will CPI
#: inflation be above 3.6%" because both match the "Will ... be above
#: [number]" template. Observed in the wild on 2026-04-17.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "not", "is", "be", "will", "does", "do",
    "on", "in", "at", "to", "of", "for", "with", "by", "from", "as",
    "this", "that", "it", "any", "all", "above", "below", "over", "under",
    "than", "more", "less", "greater", "higher", "lower",
    "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "january", "february", "march",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "am", "pm", "et", "utc", "ist",
    "yes", "no", "up", "down",
    "2024", "2025", "2026", "2027", "2028",
})
_TOKEN_RE = re.compile(r"[a-zA-Z]{3,}")

#: Polymarket's dynamic taker fee approximation (2026). Fee is
#: price-dependent; this is the P=0.5 worst case. See
#: https://docs.polymarket.com/fees for exact formula.
POLYMARKET_ROUND_TRIP_COST_PCT = 0.032

#: Gate for real edge after fees on both venues.
MIN_EDGE_AFTER_FEES = 0.07

#: Candidate pairs must close within this window of each other.
CLOSE_TIME_BUCKET = timedelta(minutes=10)

#: Fuzzy match threshold on question text (0-1 from difflib).
#: 0.62 is empirically enough to catch "Will X win" variants while
#: rejecting unrelated markets — tune after real pairs are seen.
TITLE_SIMILARITY_THRESHOLD = 0.62


@dataclass
class PolymarketMarketMin:
    """Minimal shape required from a Polymarket market for matching.

    The existing ``gamma.UpDownMarket`` covers only up/down crypto
    markets. For cross-venue arb we need broader coverage. Rather than
    couple this module to any specific Polymarket fetcher, we take a
    lightweight interface — the caller adapts whatever source fits
    (broad ``/events`` query, saved watchlist, etc.)
    """

    condition_id: str
    slug: str
    question: str
    yes_price: float            # 0.0–1.0
    end_time: datetime
    volume: float = 0.0
    raw: dict = field(repr=False, default_factory=dict)


@dataclass
class ArbOpportunity:
    poly: PolymarketMarketMin
    kalshi: KalshiMarket
    similarity: float
    #: Signed post-fee edge. +0.09 means buying YES on the cheap venue
    #: and (conceptually) selling YES on the expensive venue captures
    #: 9% of notional after fees.
    edge_after_fees: float
    cheap_venue: str            # "polymarket" | "kalshi"
    expensive_venue: str        # opposite
    notes: str = ""


def scan_opportunities(
    polymarket_markets: list[PolymarketMarketMin],
    *,
    kalshi: KalshiClient | None = None,
    min_edge: float = MIN_EDGE_AFTER_FEES,
    similarity: float = TITLE_SIMILARITY_THRESHOLD,
    bucket: timedelta = CLOSE_TIME_BUCKET,
    min_shared_tokens: int = 1,
) -> list[ArbOpportunity]:
    """Pure function: given a list of Polymarket markets, scan Kalshi
    open markets and return qualifying arb opportunities.

    Does not place orders. Returns results sorted by edge descending.

    Parameters
    ----------
    polymarket_markets : list[PolymarketMarketMin]
        Caller's responsibility to fetch these. Can be the full open
        set or a filtered watchlist.
    kalshi : KalshiClient | None
        Injected for testability. If None, creates one.
    min_edge : float
        Minimum post-fee edge. Defaults to 7%.
    similarity : float
        Title-fuzzy-match threshold.
    bucket : timedelta
        Candidate pairs must close within ±bucket of each other.
    """
    own = kalshi is None
    if kalshi is None:
        kalshi = KalshiClient()
    try:
        kalshi_markets = kalshi.get_markets(status="open", limit=1000)
    finally:
        if own:
            kalshi.close()

    # Precompute Kalshi token sets — reused across the Polymarket scan.
    km_tokens = [_significant_tokens(f"{m.title} {m.subtitle}") for m in kalshi_markets]

    opportunities: list[ArbOpportunity] = []
    for pm in polymarket_markets:
        pm_tokens = _significant_tokens(pm.question)
        for km, km_tok in zip(kalshi_markets, km_tokens):
            # Close-time bucket — cheapest filter, so it runs first.
            if abs((pm.end_time - km.close_time).total_seconds()) > bucket.total_seconds():
                continue

            # Require shared non-stopword tokens. Kills "Bitcoin above X"
            # vs "CPI above Y" false positives from pure template overlap.
            if len(pm_tokens & km_tok) < min_shared_tokens:
                continue

            sim = _title_similarity(pm.question, f"{km.title} {km.subtitle}".strip())
            if sim < similarity:
                continue

            opp = _score_pair(pm, km, sim)
            if opp is not None and opp.edge_after_fees >= min_edge:
                opportunities.append(opp)

    opportunities.sort(key=lambda o: o.edge_after_fees, reverse=True)
    return opportunities


def _significant_tokens(text: str) -> set[str]:
    """Return lowercase ≥3-letter word tokens excluding stopwords."""
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS}


def scan_opportunities_llm(
    polymarket_markets: list[PolymarketMarketMin],
    kalshi_markets: list[KalshiMarket],
    *,
    min_edge: float = MIN_EDGE_AFTER_FEES,
    min_llm_equiv: float = 0.7,
    bucket: timedelta = timedelta(hours=12),
    progress_fn=None,
    **llm_kwargs,
) -> list[ArbOpportunity]:
    """Semantic variant of ``scan_opportunities`` that uses an LLM to
    decide whether two markets resolve on the same event.

    Pipeline:
      1. Time-bucket + shared-token filter (cheap, kills 99%+ of pairs)
      2. Batch-score survivors via ``llm_match.score_pairs``
      3. Keep pairs with ``equiv >= min_llm_equiv``
      4. Compute edge, filter by ``min_edge``, sort descending

    ``progress_fn`` is an optional ``(stage: str, n: int) -> None``
    callback so the caller can show progress for the LLM step which
    can take minutes on large scans.
    """
    km_tokens = [_significant_tokens(f"{m.title} {m.subtitle}") for m in kalshi_markets]

    candidates: list[tuple[PolymarketMarketMin, KalshiMarket]] = []
    for pm in polymarket_markets:
        pm_tokens = _significant_tokens(pm.question)
        if not pm_tokens:
            continue
        for km, km_tok in zip(kalshi_markets, km_tokens):
            if abs((pm.end_time - km.close_time).total_seconds()) > bucket.total_seconds():
                continue
            if not (pm_tokens & km_tok):
                continue
            candidates.append((pm, km))

    if progress_fn:
        progress_fn("candidates", len(candidates))
    if not candidates:
        return []

    pairs = [(pm.question, f"{km.title} {km.subtitle}".strip()) for pm, km in candidates]
    scores = llm_score_pairs(pairs, **llm_kwargs)

    opportunities: list[ArbOpportunity] = []
    for (pm, km), score in zip(candidates, scores):
        if score.equiv < min_llm_equiv:
            continue
        opp = _score_pair(pm, km, score.equiv)
        if opp is None:
            continue
        opp.notes = (opp.notes + f" llm={score.equiv:.2f}: {score.reason}").strip()
        if opp.edge_after_fees >= min_edge:
            opportunities.append(opp)

    if progress_fn:
        progress_fn("final", len(opportunities))
    opportunities.sort(key=lambda o: o.edge_after_fees, reverse=True)
    return opportunities


def _score_pair(
    pm: PolymarketMarketMin, km: KalshiMarket, similarity: float,
) -> ArbOpportunity | None:
    """Compute the post-fee edge for a candidate pair.

    We compare YES mid prices. Both markets are binary YES/NO with
    YES paying $1 on resolution. If both sides are liquid and their
    YES prices diverge by more than combined fees, there's arb.
    """
    pm_yes = pm.yes_price
    km_yes = km.mid
    if pm_yes <= 0 or km_yes <= 0 or pm_yes >= 1 or km_yes >= 1:
        return None  # degenerate

    gross_edge = abs(pm_yes - km_yes)

    # Fees on the two legs: buy cheap YES, need to either hold to
    # resolution (no further fee) or sell on the expensive side.
    # Conservative: price in a round trip on both venues since we may
    # exit early.
    fees = (
        POLYMARKET_ROUND_TRIP_COST_PCT
        + kalshi_round_trip_cost_pct(km_yes, km_yes)
    )
    edge = gross_edge - fees

    cheap = "polymarket" if pm_yes < km_yes else "kalshi"
    expensive = "kalshi" if cheap == "polymarket" else "polymarket"

    notes = ""
    if km.volume_24h < 100:
        notes += "kalshi-low-volume; "
    if pm.volume < 100:
        notes += "polymarket-low-volume; "

    return ArbOpportunity(
        poly=pm,
        kalshi=km,
        similarity=similarity,
        edge_after_fees=edge,
        cheap_venue=cheap,
        expensive_venue=expensive,
        notes=notes.strip(),
    )


def _title_similarity(a: str, b: str) -> float:
    """Normalised ratio in [0,1]. Uses difflib (stdlib) — swap to
    rapidfuzz token_set_ratio later if match quality is insufficient.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def format_opportunity(o: ArbOpportunity) -> str:
    """Human-readable one-liner for logging / alerting."""
    edge_pct = o.edge_after_fees * 100
    return (
        f"ARB {edge_pct:+.1f}% | buy YES on {o.cheap_venue} "
        f"@{_pick_price(o, o.cheap_venue):.3f}, sell on {o.expensive_venue} "
        f"@{_pick_price(o, o.expensive_venue):.3f} | "
        f"sim={o.similarity:.2f} | "
        f"poly: {o.poly.question[:60]!r} | "
        f"kalshi: {o.kalshi.title[:60]!r}"
        + (f" | ⚠ {o.notes}" if o.notes else "")
    )


def _pick_price(o: ArbOpportunity, venue: str) -> float:
    if venue == "polymarket":
        return o.poly.yes_price
    return o.kalshi.mid
