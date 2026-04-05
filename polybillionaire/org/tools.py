"""ToolKit — Python-side data fetching and trade execution.

Agents use Claude Code for reasoning.  This module provides the
functions that actually talk to Polymarket APIs and manage portfolio state.
"""

from __future__ import annotations

from ..client import PolymarketClient
from ..risk import PortfolioRiskTracker
from ..scanner import MarketScanner
from ..trader import PaperTrader


class ToolKit:
    def __init__(self, client: PolymarketClient, trader: PaperTrader) -> None:
        self.client = client
        self.trader = trader
        self.scanner = MarketScanner(client, bankroll=trader.bankroll)
        self.risk_tracker = PortfolioRiskTracker(bankroll=trader.bankroll)

    def get_portfolio(self) -> dict:
        positions = self.trader.positions
        return {
            "bankroll": round(self.trader.bankroll, 4),
            "positions": len(positions),
            "total_value": round(self.trader.total_value, 4),
            "deployed": round(sum(p.cost for p in positions), 4),
            "open_pnl": round(sum(p.pnl for p in positions), 4),
            "details": [
                {
                    "market": p.market_question[:60],
                    "outcome": p.outcome,
                    "entry": p.entry_price,
                    "current": p.current_price,
                    "pnl": round(p.pnl, 4),
                }
                for p in positions
            ],
        }

    def scan_markets(self, limit: int = 30, daily_only: bool = False) -> list[dict]:
        if daily_only:
            markets = self.scanner.scan_daily_markets()
            if not markets:
                markets = self.scanner.scan_daily_markets(hours=48)
        else:
            markets = self.scanner.scan_top_markets(limit=limit)
        opps = self.scanner.find_opportunities(markets)
        return [
            {
                "market": o.market.question[:80],
                "side": o.side,
                "outcome": o.outcome,
                "price": o.price,
                "payout_multiple": o.payout_multiple,
                "score": o.score,
                "recommended_bet": o.recommended_bet,
                "volume_24h": o.volume_24h,
                "reason": o.reason,
                "token_id": next(
                    (
                        t["token_id"]
                        for t in o.market.tokens
                        if t.get("outcome", "").lower() == o.outcome.lower()
                    ),
                    o.market.tokens[0]["token_id"] if o.market.tokens else "",
                ),
                "end_date": o.market.end_date,
            }
            for o in opps
        ]

    def check_risk(self, token_id: str, cost: float, question: str) -> dict:
        can, reason = self.trader.risk.can_trade(
            cost, len(self.trader.positions)
        )
        if not can:
            return {"approved": False, "reason": reason}

        category = self.risk_tracker._categorize(question)
        can, reason = self.risk_tracker.can_open_position(
            cost, self.trader.bankroll, self.trader.positions, category,
        )
        return {
            "approved": can,
            "reason": reason if not can else "OK",
            "category": category,
        }

    def execute_buy(
        self, token_id: str, market: str, outcome: str, size: float,
        end_date: str = "",
    ) -> dict:
        ok, msg = self.trader.buy(
            token_id=token_id,
            market_question=market,
            outcome=outcome,
            size=size,
            end_date=end_date,
        )
        return {"success": ok, "message": msg}

    def execute_sell(self, token_id: str) -> dict:
        ok, msg = self.trader.sell(token_id=token_id)
        return {"success": ok, "message": msg}
