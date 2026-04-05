"""Market scanner — finds and ranks trading opportunities on Polymarket."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .client import Market, PolymarketClient


@dataclass
class Opportunity:
    market: Market
    price: float
    side: str
    outcome: str
    payout_multiple: float
    spread: float
    liquidity: float
    volume: float
    volume_24h: float
    kelly_fraction: float
    recommended_bet: float
    score: float
    reason: str


def kelly_criterion(prob: float, odds: float) -> float:
    """Optimal Kelly bet fraction. prob=true probability, odds=decimal odds."""
    if odds <= 0 or prob <= 0 or prob >= 1:
        return 0.0
    f = (prob * odds - (1 - prob)) / odds
    return max(0.0, f)


class MarketScanner:
    """Scans Polymarket for trading opportunities using Gamma API data."""

    def __init__(self, client: PolymarketClient, bankroll: float = 5.0) -> None:
        self.client = client
        self.bankroll = bankroll

    def scan_top_markets(self, *, limit: int = 50) -> list[Market]:
        return self.client.get_markets(limit=limit, order="volume24hr")

    def scan_daily_markets(self, *, hours: int = 24) -> list[Market]:
        return self.client.get_daily_markets(hours=hours)

    def scan_liquid_markets(self, *, min_liquidity: float = 5000, limit: int = 100) -> list[Market]:
        markets = self.client.get_markets(limit=limit, order="liquidity", ascending=False)
        return [m for m in markets if m.liquidity >= min_liquidity]

    def analyze_market(self, market: Market) -> dict:
        """Deep analysis of a single market using CLOB orderbook data."""
        result = {
            "question": market.question,
            "volume": market.volume,
            "volume_24h": market.volume_24h,
            "liquidity": market.liquidity,
            "end_date": market.end_date,
            "best_bid": market.best_bid,
            "best_ask": market.best_ask,
            "last_trade": market.last_trade_price,
            "day_change": market.one_day_change,
            "tokens": [],
        }

        for token in market.tokens:
            tid = token["token_id"]
            outcome = token.get("outcome", "?")
            try:
                book = self.client.get_orderbook(tid)
                history = self.client.get_price_history(tid, fidelity=30)
                prices = [p.price for p in history]

                momentum = 0.0
                volatility = 0.0
                if len(prices) >= 5:
                    recent = prices[-5:]
                    older = prices[-10:-5] if len(prices) >= 10 else prices[:5]
                    momentum = sum(recent) / len(recent) - sum(older) / len(older)
                if len(prices) >= 2:
                    returns = [
                        prices[i] / prices[i - 1] - 1
                        for i in range(1, len(prices))
                        if prices[i - 1] > 0
                    ]
                    if returns:
                        mean_r = sum(returns) / len(returns)
                        variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
                        volatility = math.sqrt(variance)

                result["tokens"].append({
                    "outcome": outcome,
                    "token_id": tid,
                    "midpoint": book.midpoint,
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "spread": book.spread,
                    "spread_pct": round(book.spread / book.midpoint * 100, 2) if book.midpoint > 0 else 0,
                    "bid_depth": sum(b["size"] for b in book.bids[:5]),
                    "ask_depth": sum(a["size"] for a in book.asks[:5]),
                    "momentum": round(momentum, 4),
                    "volatility": round(volatility, 4),
                    "price_history_len": len(prices),
                })
            except Exception as e:
                result["tokens"].append({
                    "outcome": outcome,
                    "token_id": tid,
                    "error": str(e),
                })

        return result

    def find_opportunities(
        self,
        markets: list[Market],
        *,
        min_volume: float = 1000,
        max_spread: float = 0.05,
    ) -> list[Opportunity]:
        """Score markets and find actionable opportunities.

        Strategy categories:
        1. LONG SHOTS: YES price $0.01-$0.15 — high payout multiples (10x-100x)
        2. MOMENTUM: $0.15-$0.85 with strong day/week momentum
        3. VALUE: Mid-range with tight spreads and deep liquidity
        """
        opportunities: list[Opportunity] = []

        for market in markets:
            if market.volume < min_volume or not market.tokens:
                continue

            yes_price = market.outcome_prices[0] if market.outcome_prices else 0
            no_price = market.outcome_prices[1] if len(market.outcome_prices) > 1 else (1 - yes_price)

            if yes_price <= 0:
                continue

            spread = market.spread
            if spread > max_spread:
                continue

            # Evaluate both sides of the market
            for side, price, outcome in [("YES", yes_price, "Yes"), ("NO", no_price, "No")]:
                if price <= 0.001 or price >= 0.999:
                    continue

                payout = 1.0 / price
                score = 0.0
                reasons: list[str] = []

                # ── LONG SHOT (asymmetric upside) ────────────────
                if price < 0.15:
                    score += 3.0 + (0.15 - price) * 20  # Lower price = higher score
                    reasons.append(f"{payout:.0f}x payout if {outcome}")
                    if market.volume_24h > 50_000:
                        score += 2.0
                        reasons.append("active trading")
                    if abs(market.one_day_change) > 0.005:
                        direction = "rising" if (market.one_day_change > 0 and side == "YES") or (market.one_day_change < 0 and side == "NO") else "falling"
                        if direction == "rising":
                            score += 2.0
                            reasons.append(f"price rising ({market.one_day_change:+.1%})")
                        else:
                            score += 0.5
                            reasons.append(f"contrarian play")

                # ── MOMENTUM PLAY ($0.15-$0.85) ──────────────────
                elif 0.15 <= price <= 0.85:
                    # Only take momentum side
                    is_momentum_side = (
                        (side == "YES" and market.one_day_change > 0) or
                        (side == "NO" and market.one_day_change < 0)
                    )
                    if not is_momentum_side:
                        continue

                    if abs(market.one_day_change) > 0.03:
                        score += 4.0
                        reasons.append(f"strong momentum ({market.one_day_change:+.1%} today)")
                    elif abs(market.one_day_change) > 0.01:
                        score += 2.0
                        reasons.append(f"momentum ({market.one_day_change:+.1%} today)")
                    else:
                        continue  # Skip weak momentum in mid-range

                    if abs(market.one_week_change) > 0.05:
                        score += 1.5
                        reasons.append(f"week trend {market.one_week_change:+.1%}")

                    reasons.append(f"{payout:.1f}x payout")

                # ── NEAR CERTAINTY — skip (bad risk/reward) ──────
                else:
                    continue

                # ── Common signals ────────────────────────────────
                # Tight spread = liquid, tradeable
                if spread <= 0.001:
                    score += 1.5
                    reasons.append("razor-thin spread")
                elif spread <= 0.005:
                    score += 0.5

                # High volume = more emotional retail money = more mispricing
                if market.volume_24h > 500_000:
                    score += 3.0
                    reasons.append(f"massive vol ${market.volume_24h:,.0f}")
                elif market.volume_24h > 100_000:
                    score += 2.0
                    reasons.append(f"high vol ${market.volume_24h:,.0f}")
                elif market.volume_24h > 10_000:
                    score += 0.5

                # Liquidity = can enter/exit without slippage
                if market.liquidity > 500_000:
                    score += 1.0

                if score < 3.0 or not reasons:
                    continue

                # ── Bet sizing (half Kelly) ───────────────────────
                # Default sizing uses score-based edge estimate.
                # Runner overrides with agent's real edge when available.
                odds = payout - 1
                est_edge = 0.05 if score > 8 else 0.03 if score > 5 else 0.02
                est_prob = min(0.95, price + est_edge)
                kf = kelly_criterion(est_prob, odds) * 0.5

                recommended = round(self.bankroll * max(kf, 0.01), 2)
                recommended = max(0.02, min(recommended, self.bankroll * 0.10))

                opportunities.append(Opportunity(
                    market=market,
                    price=price,
                    side=side,
                    outcome=outcome,
                    payout_multiple=round(payout, 1),
                    spread=spread,
                    liquidity=market.liquidity,
                    volume=market.volume,
                    volume_24h=market.volume_24h,
                    kelly_fraction=round(kf, 4),
                    recommended_bet=recommended,
                    score=round(score, 2),
                    reason=" | ".join(reasons),
                ))

        opportunities.sort(key=lambda o: o.score, reverse=True)
        return opportunities

    def resize_with_edge(
        self, market_price: float, agent_prob: float,
    ) -> float:
        """Recalculate bet size using the agent's probability estimate.

        Returns the recommended bet in dollars. Uses half Kelly.
        """
        if agent_prob <= market_price or agent_prob <= 0 or agent_prob >= 1:
            return 0.0
        odds = (1.0 / market_price) - 1
        kf = kelly_criterion(agent_prob, odds) * 0.5
        bet = round(self.bankroll * max(kf, 0.01), 2)
        return max(0.02, min(bet, self.bankroll * 0.10))
