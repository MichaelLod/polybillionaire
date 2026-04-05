"""Portfolio-level risk tracking framework for 100-position strategy."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .trader import Position, Trade


@dataclass
class CategoryExposure:
    category: str
    position_count: int
    total_cost: float
    total_value: float
    pnl: float
    pct_of_bankroll: float


@dataclass
class RebalanceAlert:
    token_id: str
    market_question: str
    outcome: str
    freed_capital: float
    reason: str


@dataclass
class CorrelationFlag:
    category: str
    position_count: int
    total_exposure: float
    pct_of_bankroll: float
    message: str


@dataclass
class PortfolioSnapshot:
    timestamp: float
    bankroll: float
    total_deployed: float
    total_value: float
    available_capital: float
    capital_at_risk: float
    open_positions: int
    target_positions: int
    utilization_pct: float
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    best_performer: dict | None
    worst_performer: dict | None
    category_breakdown: list[CategoryExposure]
    correlation_flags: list[CorrelationFlag]
    rebalance_alerts: list[RebalanceAlert]
    stop_loss_alerts: list[str]


class PortfolioRiskTracker:
    """Portfolio-level risk monitoring for high-frequency micro-betting.

    Tracks exposure, correlation, P&L, and generates rebalancing alerts
    across up to 100 concurrent positions.
    """

    def __init__(
        self,
        bankroll: float = 5.0,
        target_positions: int = 20,
        max_category_pct: float = 0.25,
        max_single_bet_pct: float = 0.10,
        stop_loss_pct: float = 0.50,
        max_portfolio_drawdown_pct: float = 0.30,
    ) -> None:
        self.initial_bankroll = bankroll
        self.target_positions = target_positions
        self.max_category_pct = max_category_pct
        self.max_single_bet_pct = max_single_bet_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_portfolio_drawdown_pct = max_portfolio_drawdown_pct

    def generate_snapshot(
        self,
        bankroll: float,
        positions: list[Position],
        trades: list[Trade],
    ) -> PortfolioSnapshot:
        total_deployed = sum(p.cost for p in positions)
        total_value = sum(p.value for p in positions)
        available_capital = bankroll
        capital_at_risk = total_deployed
        total_pnl = sum(p.pnl for p in positions)

        # Win/loss from closed trades (sells)
        sells = [t for t in trades if t.side == "SELL"]
        buys_by_token = {}
        for t in trades:
            if t.side == "BUY":
                buys_by_token[t.token_id] = t.price

        win_count = 0
        loss_count = 0
        for s in sells:
            entry = buys_by_token.get(s.token_id, 0)
            if s.price > entry:
                win_count += 1
            elif s.price < entry:
                loss_count += 1

        total_closed = win_count + loss_count
        win_rate = win_count / total_closed if total_closed > 0 else 0.0

        # Best/worst open positions
        best = None
        worst = None
        if positions:
            sorted_by_pnl = sorted(positions, key=lambda p: p.pnl, reverse=True)
            bp = sorted_by_pnl[0]
            wp = sorted_by_pnl[-1]
            best = {
                "market": bp.market_question,
                "outcome": bp.outcome,
                "pnl": bp.pnl,
                "entry": bp.entry_price,
                "current": bp.current_price,
            }
            worst = {
                "market": wp.market_question,
                "outcome": wp.outcome,
                "pnl": wp.pnl,
                "entry": wp.entry_price,
                "current": wp.current_price,
            }

        total_capital = bankroll + total_deployed
        utilization = (len(positions) / self.target_positions * 100) if self.target_positions > 0 else 0

        category_breakdown = self._category_breakdown(positions, total_capital)
        correlation_flags = self._check_correlations(positions, total_capital)
        rebalance_alerts = self._check_rebalance(positions, bankroll)
        stop_loss_alerts = self._check_stop_losses(positions)

        return PortfolioSnapshot(
            timestamp=time.time(),
            bankroll=bankroll,
            total_deployed=total_deployed,
            total_value=total_value,
            available_capital=available_capital,
            capital_at_risk=capital_at_risk,
            open_positions=len(positions),
            target_positions=self.target_positions,
            utilization_pct=utilization,
            total_pnl=total_pnl,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            best_performer=best,
            worst_performer=worst,
            category_breakdown=category_breakdown,
            correlation_flags=correlation_flags,
            rebalance_alerts=rebalance_alerts,
            stop_loss_alerts=stop_loss_alerts,
        )

    def _category_breakdown(
        self, positions: list[Position], total_capital: float
    ) -> list[CategoryExposure]:
        by_cat: dict[str, list[Position]] = defaultdict(list)
        for p in positions:
            cat = self._categorize(p.market_question)
            by_cat[cat].append(p)

        result = []
        for cat, pos_list in sorted(by_cat.items()):
            total_cost = sum(p.cost for p in pos_list)
            total_val = sum(p.value for p in pos_list)
            pnl = sum(p.pnl for p in pos_list)
            pct = total_cost / total_capital if total_capital > 0 else 0
            result.append(CategoryExposure(
                category=cat,
                position_count=len(pos_list),
                total_cost=total_cost,
                total_value=total_val,
                pnl=pnl,
                pct_of_bankroll=pct,
            ))
        return result

    def _check_correlations(
        self, positions: list[Position], total_capital: float
    ) -> list[CorrelationFlag]:
        flags = []
        by_cat: dict[str, list[Position]] = defaultdict(list)
        for p in positions:
            cat = self._categorize(p.market_question)
            by_cat[cat].append(p)

        for cat, pos_list in by_cat.items():
            exposure = sum(p.cost for p in pos_list)
            pct = exposure / total_capital if total_capital > 0 else 0
            if pct > self.max_category_pct:
                flags.append(CorrelationFlag(
                    category=cat,
                    position_count=len(pos_list),
                    total_exposure=exposure,
                    pct_of_bankroll=pct,
                    message=f"Category '{cat}' at {pct:.0%} of capital "
                            f"({len(pos_list)} positions, ${exposure:.2f}) — "
                            f"exceeds {self.max_category_pct:.0%} limit",
                ))

        # Also flag duplicate market questions (same event, multiple bets)
        by_question: dict[str, list[Position]] = defaultdict(list)
        for p in positions:
            by_question[p.market_question].append(p)
        for q, pos_list in by_question.items():
            if len(pos_list) > 1:
                exposure = sum(p.cost for p in pos_list)
                pct = exposure / total_capital if total_capital > 0 else 0
                flags.append(CorrelationFlag(
                    category=f"duplicate:{q[:50]}",
                    position_count=len(pos_list),
                    total_exposure=exposure,
                    pct_of_bankroll=pct,
                    message=f"Multiple positions on same market: '{q[:60]}' "
                            f"({len(pos_list)} positions, ${exposure:.2f})",
                ))

        return flags

    def _check_rebalance(
        self, positions: list[Position], bankroll: float
    ) -> list[RebalanceAlert]:
        alerts = []
        for p in positions:
            # Position resolved (price near 0 or 1)
            if p.current_price >= 0.95:
                alerts.append(RebalanceAlert(
                    token_id=p.token_id,
                    market_question=p.market_question,
                    outcome=p.outcome,
                    freed_capital=p.value,
                    reason=f"Near resolution (price={p.current_price:.3f}). "
                           f"Sell to free ~${p.value:.2f}",
                ))
            elif p.current_price <= 0.005 and p.entry_price > 0.01:
                alerts.append(RebalanceAlert(
                    token_id=p.token_id,
                    market_question=p.market_question,
                    outcome=p.outcome,
                    freed_capital=0.0,
                    reason=f"Near-zero (price={p.current_price:.4f}). "
                           f"Consider closing to free position slot.",
                ))

        # Alert if well below target and capital available
        slots_open = self.target_positions - len(positions)
        if slots_open > 10 and bankroll > 0.10:
            avg_bet = bankroll * self.max_single_bet_pct
            deployable = int(bankroll / avg_bet) if avg_bet > 0 else 0
            alerts.append(RebalanceAlert(
                token_id="",
                market_question="",
                outcome="",
                freed_capital=bankroll,
                reason=f"{slots_open} position slots open. "
                       f"${bankroll:.2f} available, ~{min(deployable, slots_open)} "
                       f"new positions deployable at ${avg_bet:.2f}/each",
            ))

        return alerts

    def _check_stop_losses(self, positions: list[Position]) -> list[str]:
        alerts = []
        for p in positions:
            if p.entry_price <= 0:
                continue
            loss_pct = (p.entry_price - p.current_price) / p.entry_price
            if loss_pct >= self.stop_loss_pct:
                alerts.append(
                    f"STOP LOSS: {p.outcome} on '{p.market_question[:50]}' "
                    f"down {loss_pct:.0%} (entry=${p.entry_price:.3f}, "
                    f"now=${p.current_price:.4f})"
                )
        return alerts

    def check_portfolio_drawdown(
        self, bankroll: float, positions: list[Position]
    ) -> tuple[bool, float]:
        total_capital = bankroll + sum(p.value for p in positions)
        drawdown = 1 - (total_capital / self.initial_bankroll) if self.initial_bankroll > 0 else 0
        breached = drawdown >= self.max_portfolio_drawdown_pct
        return breached, drawdown

    def can_open_position(
        self,
        cost: float,
        bankroll: float,
        positions: list[Position],
        category: str = "",
    ) -> tuple[bool, str]:
        total_capital = bankroll + sum(p.cost for p in positions)

        if len(positions) >= self.target_positions:
            return False, f"At max positions ({self.target_positions})"

        max_bet = total_capital * self.max_single_bet_pct
        if cost > max_bet:
            return False, f"Cost ${cost:.2f} exceeds max single bet ${max_bet:.2f} ({self.max_single_bet_pct:.0%})"

        if cost > bankroll:
            return False, f"Cost ${cost:.2f} exceeds available capital ${bankroll:.2f}"

        if category:
            cat_exposure = sum(
                p.cost for p in positions
                if self._categorize(p.market_question) == category
            )
            cat_limit = total_capital * self.max_category_pct
            if cat_exposure + cost > cat_limit:
                return False, (
                    f"Category '{category}' would reach ${cat_exposure + cost:.2f} "
                    f"(limit: ${cat_limit:.2f}, {self.max_category_pct:.0%} of capital)"
                )

        breached, dd = self.check_portfolio_drawdown(bankroll, positions)
        if breached:
            return False, f"Portfolio drawdown {dd:.0%} exceeds limit {self.max_portfolio_drawdown_pct:.0%}"

        return True, "OK"

    def format_risk_report(self, snapshot: PortfolioSnapshot) -> str:
        lines = [
            "## Portfolio Risk Report",
            "",
            "### Capital",
            f"- **Bankroll (available):** ${snapshot.bankroll:.2f}",
            f"- **Deployed:** ${snapshot.total_deployed:.2f}",
            f"- **Portfolio value:** ${snapshot.total_value:.2f}",
            f"- **Capital at risk:** ${snapshot.capital_at_risk:.2f}",
            "",
            "### Positions",
            f"- **Open:** {snapshot.open_positions} / {snapshot.target_positions} target",
            f"- **Utilization:** {snapshot.utilization_pct:.0f}%",
            "",
            "### P&L",
            f"- **Total P&L:** ${snapshot.total_pnl:+.4f}",
            f"- **Win/Loss:** {snapshot.win_count}W / {snapshot.loss_count}L "
            f"({snapshot.win_rate:.0%} win rate)",
        ]

        if snapshot.best_performer:
            bp = snapshot.best_performer
            lines.append(f"- **Best:** {bp['outcome']} on '{bp['market'][:40]}' (${bp['pnl']:+.4f})")
        if snapshot.worst_performer:
            wp = snapshot.worst_performer
            lines.append(f"- **Worst:** {wp['outcome']} on '{wp['market'][:40]}' (${wp['pnl']:+.4f})")

        if snapshot.category_breakdown:
            lines += ["", "### Category Exposure"]
            for cat in snapshot.category_breakdown:
                lines.append(
                    f"- **{cat.category}:** {cat.position_count} positions, "
                    f"${cat.total_cost:.2f} deployed ({cat.pct_of_bankroll:.0%}), "
                    f"P&L ${cat.pnl:+.4f}"
                )

        if snapshot.correlation_flags:
            lines += ["", "### Correlation Alerts"]
            for cf in snapshot.correlation_flags:
                lines.append(f"- {cf.message}")

        if snapshot.stop_loss_alerts:
            lines += ["", "### Stop Loss Alerts"]
            for sl in snapshot.stop_loss_alerts:
                lines.append(f"- {sl}")

        if snapshot.rebalance_alerts:
            lines += ["", "### Rebalancing"]
            for ra in snapshot.rebalance_alerts:
                lines.append(f"- {ra.reason}")

        return "\n".join(lines)

    @staticmethod
    def _categorize(question: str) -> str:
        q = question.lower()
        categories = {
            "crypto": ["bitcoin", "btc", "eth", "ethereum", "crypto", "solana", "sol", "token", "defi"],
            "us_politics": ["trump", "biden", "congress", "senate", "republican", "democrat", "election", "gop", "dnc", "potus", "president"],
            "geopolitics": ["iran", "russia", "ukraine", "china", "taiwan", "nato", "ceasefire", "war", "sanctions", "missile"],
            "economics": ["fed", "interest rate", "inflation", "gdp", "recession", "tariff", "trade war", "unemployment"],
            "sports": ["nba", "nfl", "mlb", "ufc", "premier league", "champions league", "world cup", "super bowl", "march madness"],
            "tech": ["ai", "openai", "google", "apple", "meta", "microsoft", "spacex", "tesla"],
            "entertainment": ["oscar", "grammy", "emmy", "box office", "netflix", "movie", "album"],
            "weather": ["hurricane", "earthquake", "temperature", "climate", "storm"],
        }
        for cat, keywords in categories.items():
            if any(kw in q for kw in keywords):
                return cat
        return "other"

    @classmethod
    def from_portfolio_file(
        cls, path: str = "paper_portfolio.json"
    ) -> tuple[PortfolioRiskTracker, float, list[Position], list[Trade]]:
        from .trader import Position as Pos, Trade as Tr

        p = Path(path)
        if not p.exists():
            tracker = cls()
            return tracker, 5.0, [], []

        state = json.loads(p.read_text())
        bankroll = state.get("bankroll", 5.0)
        positions = [Pos(**d) for d in state.get("positions", [])]
        trades = [Tr(**d) for d in state.get("trades", [])]
        tracker = cls(bankroll=bankroll)
        return tracker, bankroll, positions, trades
