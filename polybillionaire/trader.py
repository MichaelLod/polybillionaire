"""Trading engine with risk management and paper trading."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .client import PolymarketClient


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass
class Position:
    token_id: str
    market_question: str
    outcome: str
    side: str
    entry_price: float
    size: float
    cost: float
    current_price: float = 0.0
    pnl: float = 0.0
    opened_at: float = field(default_factory=time.time)
    end_date: str = ""

    @property
    def value(self) -> float:
        return self.size * self.current_price

    def update_pnl(self, current_price: float) -> None:
        self.current_price = current_price
        self.pnl = (current_price - self.entry_price) * self.size


@dataclass
class Trade:
    token_id: str
    market_question: str
    outcome: str
    side: str
    price: float
    size: float
    cost: float
    timestamp: float = field(default_factory=time.time)
    paper: bool = True


class RiskManager:
    """Enforces position limits and risk rules."""

    def __init__(
        self,
        bankroll: float = 5.0,
        max_bet_fraction: float = 0.10,
        max_positions: int = 20,
        stop_loss_pct: float = 0.50,
        max_daily_loss: float | None = None,
    ) -> None:
        self.bankroll = bankroll
        self.max_bet_fraction = max_bet_fraction
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.max_daily_loss = max_daily_loss or bankroll * 0.30
        self.daily_loss = 0.0

    @property
    def max_bet(self) -> float:
        return self.bankroll * self.max_bet_fraction

    def can_trade(self, cost: float, num_positions: int) -> tuple[bool, str]:
        if cost > self.max_bet:
            return False, f"Cost ${cost:.2f} exceeds max bet ${self.max_bet:.2f}"
        if cost > self.bankroll:
            return False, f"Cost ${cost:.2f} exceeds bankroll ${self.bankroll:.2f}"
        if num_positions >= self.max_positions:
            return False, f"Already at max positions ({self.max_positions})"
        if self.daily_loss >= self.max_daily_loss:
            return False, f"Daily loss limit reached (${self.daily_loss:.2f}/${self.max_daily_loss:.2f})"
        return True, "OK"

    def should_stop_loss(self, position: Position) -> bool:
        if position.entry_price <= 0:
            return False
        loss_pct = (position.entry_price - position.current_price) / position.entry_price
        return loss_pct >= self.stop_loss_pct

    def update_bankroll(self, pnl: float) -> None:
        self.bankroll += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)


class PaperTrader:
    """Simulated trading for testing strategies without real money."""

    def __init__(
        self,
        client: PolymarketClient,
        bankroll: float = 5.0,
        state_file: str = "paper_portfolio.json",
    ) -> None:
        self.client = client
        self.risk = RiskManager(bankroll=bankroll)
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.state_file = Path(state_file)
        self._load_state()

    @property
    def bankroll(self) -> float:
        return self.risk.bankroll

    @property
    def total_value(self) -> float:
        return self.bankroll + sum(p.value for p in self.positions)

    @property
    def total_pnl(self) -> float:
        return sum(p.pnl for p in self.positions) + sum(
            t.cost * (-1 if t.side == "SELL" else 1) for t in self.trades
        )

    def buy(
        self,
        token_id: str,
        market_question: str,
        outcome: str,
        size: float,
        end_date: str = "",
    ) -> tuple[bool, str]:
        """Buy shares at current ask price."""
        try:
            book = self.client.get_orderbook(token_id)
        except Exception as e:
            return False, f"Failed to get price: {e}"

        price = book.best_ask
        if price <= 0 or price >= 1:
            return False, f"Invalid price: {price}"

        cost = price * size
        can, reason = self.risk.can_trade(cost, len(self.positions))
        if not can:
            return False, reason

        self.risk.update_bankroll(-cost)
        existing = next((p for p in self.positions if p.token_id == token_id), None)
        if existing:
            total_size = existing.size + size
            existing.entry_price = (existing.entry_price * existing.size + price * size) / total_size
            existing.size = total_size
            existing.cost += cost
            existing.current_price = price
        else:
            position = Position(
                token_id=token_id,
                market_question=market_question,
                outcome=outcome,
                side="BUY",
                entry_price=price,
                size=size,
                cost=cost,
                current_price=price,
                end_date=end_date,
            )
            self.positions.append(position)
        self.trades.append(Trade(
            token_id=token_id,
            market_question=market_question,
            outcome=outcome,
            side="BUY",
            price=price,
            size=size,
            cost=cost,
        ))
        self._save_state()
        return True, f"Bought {size:.1f} {outcome} shares @ ${price:.3f} (cost: ${cost:.2f})"

    def sell(self, token_id: str, size: float | None = None) -> tuple[bool, str]:
        """Sell shares at current bid price."""
        pos = next((p for p in self.positions if p.token_id == token_id), None)
        if not pos:
            return False, "No position found for this token"

        try:
            book = self.client.get_orderbook(token_id)
        except Exception as e:
            return False, f"Failed to get price: {e}"

        sell_size = size or pos.size
        if sell_size > pos.size:
            return False, f"Can only sell up to {pos.size:.1f} shares"

        price = book.best_bid
        proceeds = price * sell_size
        pnl = (price - pos.entry_price) * sell_size

        self.risk.update_bankroll(proceeds)
        self.trades.append(Trade(
            token_id=token_id,
            market_question=pos.market_question,
            outcome=pos.outcome,
            side="SELL",
            price=price,
            size=sell_size,
            cost=proceeds,
        ))

        if sell_size >= pos.size:
            self.positions.remove(pos)
        else:
            pos.size -= sell_size
            pos.cost = pos.entry_price * pos.size

        self._save_state()
        return True, (
            f"Sold {sell_size:.1f} shares @ ${price:.3f} "
            f"(proceeds: ${proceeds:.2f}, PnL: ${pnl:+.2f})"
        )

    def settle_resolved(self) -> list[dict]:
        """Check all positions for resolved markets and collect winnings.

        Detection: a resolved market has no orderbook (midpoint errors, book
        empty). Payout is determined by the last known price — near 1.0 means
        our outcome won, near 0.0 means it lost.
        """
        settled: list[dict] = []
        to_remove: list[Position] = []

        for pos in self.positions:
            resolved, last_price = self.client.is_market_resolved(pos.token_id)
            if not resolved:
                # Still active — update price if we got one
                if last_price is not None:
                    pos.update_pnl(last_price)
                continue

            # Extra guard: only settle if end_date has passed (when available)
            if pos.end_date:
                try:
                    from datetime import datetime, timezone
                    end = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                    if datetime.now(timezone.utc) < end:
                        continue  # Market hasn't ended yet — orderbook may just be thin
                except (ValueError, TypeError):
                    pass

            # Market is resolved. Use last known current_price to decide payout.
            # Resolved winning tokens converge to ~$1.00, losers to ~$0.00.
            last = pos.current_price
            won = last >= 0.5
            payout_price = 1.0 if won else 0.0

            proceeds = payout_price * pos.size
            pnl = proceeds - pos.cost
            self.risk.update_bankroll(proceeds)
            to_remove.append(pos)

            self.trades.append(Trade(
                token_id=pos.token_id,
                market_question=pos.market_question,
                outcome=pos.outcome,
                side="SETTLE",
                price=payout_price,
                size=pos.size,
                cost=proceeds,
            ))

            settled.append({
                "position": pos,
                "won": won,
                "payout_price": payout_price,
                "proceeds": proceeds,
                "pnl": pnl,
                "message": (
                    f"{'WON' if won else 'LOST'}: {pos.outcome} \"{pos.market_question[:50]}\" "
                    f"→ ${proceeds:.4f} (PnL: ${pnl:+.4f})"
                ),
            })

        for pos in to_remove:
            self.positions.remove(pos)
        if settled:
            self._save_state()
        return settled

    def update_positions(self, auto_stop: bool = True) -> list[dict]:
        """Refresh all position prices, execute stop-losses, return alerts."""
        alerts = []
        to_stop: list[Position] = []
        for pos in self.positions:
            try:
                mid = self.client.get_midpoint(pos.token_id)
                pos.update_pnl(mid)
                if self.risk.should_stop_loss(pos):
                    to_stop.append(pos)
            except Exception:
                pass
        self._save_state()

        if auto_stop:
            for pos in to_stop:
                ok, msg = self.sell(pos.token_id)
                alerts.append({
                    "type": "stop_loss",
                    "position": pos,
                    "sold": ok,
                    "message": f"STOP LOSS: {pos.outcome} ({pos.market_question[:50]}) "
                               f"down {(1 - pos.current_price / pos.entry_price) * 100:.1f}% — "
                               + (msg if ok else f"sell failed: {msg}"),
                })
        else:
            for pos in to_stop:
                alerts.append({
                    "type": "stop_loss",
                    "position": pos,
                    "sold": False,
                    "message": f"STOP LOSS: {pos.outcome} ({pos.market_question[:50]}) "
                               f"down {(1 - pos.current_price / pos.entry_price) * 100:.1f}%",
                })
        return alerts

    def _save_state(self) -> None:
        state = {
            "bankroll": self.risk.bankroll,
            "positions": [asdict(p) for p in self.positions],
            "trades": [asdict(t) for t in self.trades[-100:]],
        }
        self.state_file.write_text(json.dumps(state, indent=2, default=str))

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            state = json.loads(self.state_file.read_text())
            self.risk.bankroll = state.get("bankroll", self.risk.bankroll)
            self.positions = [
                Position(**p) for p in state.get("positions", [])
            ]
            self.trades = [
                Trade(**t) for t in state.get("trades", [])
            ]
        except (json.JSONDecodeError, TypeError):
            pass


class LiveTrader:
    """Real trading via py-clob-client. Requires private key + API creds.

    Implements the same interface as PaperTrader so the swarm can use
    either one interchangeably:
      .bankroll, .positions, .total_value, .risk
      .buy(token_id, market_question, outcome, size, end_date)
      .sell(token_id, size)
      .settle_resolved() -> list[dict]
      .update_positions(auto_stop) -> list[dict]
    """

    def __init__(
        self,
        private_key: str,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        proxy_address: str = "",
        bankroll: float = 5.0,
    ) -> None:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import (
            ApiCreds, AssetType, BalanceAllowanceParams,
            MarketOrderArgs, OrderArgs,
        )

        self.MarketOrderArgs = MarketOrderArgs
        self.OrderArgs = OrderArgs
        self.AssetType = AssetType
        self.BalanceAllowanceParams = BalanceAllowanceParams

        self.clob = ClobClient(
            host="https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
            signature_type=2,
            funder=proxy_address if proxy_address else None,
        )
        self.clob.set_api_creds(ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        ))
        self.risk = RiskManager(bankroll=bankroll)
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self._client: PolymarketClient | None = None

    @property
    def bankroll(self) -> float:
        return self.risk.bankroll

    @property
    def total_value(self) -> float:
        return self.bankroll + sum(p.value for p in self.positions)

    def set_client(self, client: PolymarketClient) -> None:
        """Set the PolymarketClient for price lookups."""
        self._client = client

    def buy(
        self,
        token_id: str,
        market_question: str,
        outcome: str,
        size: float,
        end_date: str = "",
    ) -> tuple[bool, str]:
        """Buy shares at market price via CLOB API."""
        from py_clob_client.order_builder.constants import BUY

        # Estimate cost from orderbook
        price = 0.0
        if self._client:
            try:
                book = self._client.get_orderbook(token_id)
                price = book.best_ask
            except Exception:
                pass

        cost = price * size if price > 0 else size * 0.5
        can, reason = self.risk.can_trade(cost, len(self.positions))
        if not can:
            return False, reason

        try:
            signed = self.clob.create_market_order(
                self.MarketOrderArgs(
                    token_id=token_id,
                    amount=cost,
                    side=BUY,
                )
            )
            result = self.clob.post_order(signed)
            if "error" in result:
                return False, str(result["error"])
        except Exception as e:
            return False, f"Order failed: {e}"

        self.risk.update_bankroll(-cost)
        position = Position(
            token_id=token_id,
            market_question=market_question,
            outcome=outcome,
            side="BUY",
            entry_price=price,
            size=size,
            cost=cost,
            current_price=price,
            end_date=end_date,
        )
        self.positions.append(position)
        self.trades.append(Trade(
            token_id=token_id,
            market_question=market_question,
            outcome=outcome,
            side="BUY",
            price=price,
            size=size,
            cost=cost,
            paper=False,
        ))
        return True, f"LIVE bought {size:.1f} {outcome} @ ${price:.3f} (cost: ${cost:.2f})"

    def sell(self, token_id: str, size: float | None = None) -> tuple[bool, str]:
        """Sell shares at market price via CLOB API."""
        from py_clob_client.order_builder.constants import SELL

        pos = next((p for p in self.positions if p.token_id == token_id), None)
        if not pos:
            return False, "No position found"

        sell_size = size or pos.size
        price = pos.current_price

        try:
            signed = self.clob.create_market_order(
                self.MarketOrderArgs(
                    token_id=token_id,
                    amount=price * sell_size,
                    side=SELL,
                )
            )
            result = self.clob.post_order(signed)
            if "error" in result:
                return False, str(result["error"])
        except Exception as e:
            return False, f"Sell failed: {e}"

        proceeds = price * sell_size
        pnl = (price - pos.entry_price) * sell_size
        self.risk.update_bankroll(proceeds)
        self.trades.append(Trade(
            token_id=token_id,
            market_question=pos.market_question,
            outcome=pos.outcome,
            side="SELL",
            price=price,
            size=sell_size,
            cost=proceeds,
            paper=False,
        ))

        if sell_size >= pos.size:
            self.positions.remove(pos)
        else:
            pos.size -= sell_size
            pos.cost = pos.entry_price * pos.size

        return True, f"LIVE sold {sell_size:.1f} @ ${price:.3f} (PnL: ${pnl:+.2f})"

    def settle_resolved(self) -> list[dict]:
        """Check positions for resolved markets. Returns list of settlements."""
        if not self._client:
            return []
        settled: list[dict] = []
        to_remove: list[Position] = []

        for pos in self.positions:
            resolved, last_price = self._client.is_market_resolved(pos.token_id)
            if not resolved:
                if last_price is not None:
                    pos.update_pnl(last_price)
                continue

            won = (last_price or pos.current_price) >= 0.5
            payout_price = 1.0 if won else 0.0
            proceeds = payout_price * pos.size
            pnl = proceeds - pos.cost
            self.risk.update_bankroll(proceeds)
            to_remove.append(pos)
            settled.append({
                "position": pos,
                "won": won,
                "payout_price": payout_price,
                "proceeds": proceeds,
                "pnl": pnl,
                "message": (
                    f"{'WON' if won else 'LOST'}: {pos.outcome} "
                    f"\"{pos.market_question[:50]}\" → ${proceeds:.4f} (PnL: ${pnl:+.4f})"
                ),
            })

        for pos in to_remove:
            self.positions.remove(pos)
        return settled

    def update_positions(self, auto_stop: bool = True) -> list[dict]:
        """Refresh prices and execute stop-losses."""
        if not self._client:
            return []
        alerts: list[dict] = []
        to_stop: list[Position] = []

        for pos in self.positions:
            try:
                mid = self._client.get_midpoint(pos.token_id)
                pos.update_pnl(mid)
                if self.risk.should_stop_loss(pos):
                    to_stop.append(pos)
            except Exception:
                pass

        if auto_stop:
            for pos in to_stop:
                ok, msg = self.sell(pos.token_id)
                alerts.append({
                    "type": "stop_loss",
                    "position": pos,
                    "sold": ok,
                    "message": f"STOP LOSS: {msg}",
                })
        return alerts

    def recover_positions(self) -> int:
        """Pull existing positions from Polymarket API on startup.

        Returns number of positions recovered.
        """
        if not self._client:
            return 0
        # Positions live on the proxy address, not the main address
        address = os.environ.get("POLY_PROXY_ADDRESS", "") or os.environ.get("POLY_ADDRESS", "")
        if not address:
            return 0
        try:
            raw_positions = self._client.get_positions(address)
        except Exception:
            return 0

        recovered = 0
        for rp in raw_positions:
            size = float(rp.get("size", 0))
            if size <= 0:
                continue
            # Skip resolved/redeemable positions
            if rp.get("redeemable", False):
                continue
            token_id = rp.get("asset", "")
            if not token_id:
                continue
            # Skip if already tracked
            if any(p.token_id == token_id for p in self.positions):
                continue
            avg_price = float(rp.get("avgPrice", 0))
            cur_price = float(rp.get("curPrice", 0)) or avg_price
            current_value = float(rp.get("currentValue", 0))
            initial_value = float(rp.get("initialValue", 0))
            market = rp.get("title", "") or "recovered"
            outcome = rp.get("outcome", "YES")
            end_date = rp.get("endDate", "")
            pos = Position(
                token_id=token_id,
                market_question=market[:200],
                outcome=outcome,
                side="BUY",
                entry_price=avg_price,
                size=size,
                cost=initial_value,
                current_price=cur_price,
                end_date=end_date,
            )
            pos.update_pnl(cur_price)
            self.positions.append(pos)
            recovered += 1
        return recovered

    @classmethod
    def from_env(cls) -> LiveTrader:
        """Create a LiveTrader from .env configuration."""
        trader = cls(
            private_key=os.environ["POLY_PRIVATE_KEY"],
            api_key=os.environ["POLY_API_KEY"],
            api_secret=os.environ["POLY_API_SECRET"],
            api_passphrase=os.environ["POLY_API_PASSPHRASE"],
            proxy_address=os.environ.get("POLY_PROXY_ADDRESS", ""),
            bankroll=float(os.environ.get("BANKROLL", "5.0")),
        )
        trader.risk.max_bet_fraction = float(os.environ.get("MAX_BET_FRACTION", "0.10"))
        return trader

    def get_balance(self) -> dict:
        """Check USDC balance."""
        try:
            params = self.BalanceAllowanceParams(
                asset_type=self.AssetType.COLLATERAL,
                signature_type=2,
            )
            raw = self.clob.get_balance_allowance(params)
            balance_raw = int(raw.get("balance", "0"))
            return {
                "usdc": balance_raw / 1_000_000,
                "usdc_raw": balance_raw,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_open_orders(self) -> list:
        """Get all open orders."""
        try:
            return self.clob.get_orders()
        except Exception as e:
            return [{"error": str(e)}]

    def place_market_buy(
        self,
        token_id: str,
        amount: float,
    ) -> dict:
        """Buy shares at market price (best available ask).

        Args:
            token_id: The token to buy.
            amount: Dollar amount to spend.
        """
        from py_clob_client.order_builder.constants import BUY

        can, reason = self.risk.can_trade(amount, 0)
        if not can:
            return {"error": reason}

        try:
            signed = self.clob.create_market_order(
                self.MarketOrderArgs(
                    token_id=token_id,
                    amount=amount,
                    side=BUY,
                )
            )
            result = self.clob.post_order(signed)
            self.risk.update_bankroll(-amount)
            return result
        except Exception as e:
            return {"error": str(e)}

    def place_limit_order(
        self,
        token_id: str,
        price: float,
        size: float,
        side: Side = Side.BUY,
    ) -> dict:
        """Place a limit order at a specific price.

        Args:
            token_id: The token to trade.
            price: Limit price (0-1).
            size: Number of shares.
            side: BUY or SELL.
        """
        from py_clob_client.order_builder.constants import BUY, SELL

        cost = price * size if side == Side.BUY else 0
        can, reason = self.risk.can_trade(cost, 0)
        if not can:
            return {"error": reason}

        try:
            signed = self.clob.create_order(
                self.OrderArgs(
                    token_id=token_id,
                    price=price,
                    size=size,
                    side=BUY if side == Side.BUY else SELL,
                )
            )
            result = self.clob.post_order(signed)
            if side == Side.BUY:
                self.risk.update_bankroll(-cost)
            return result
        except Exception as e:
            return {"error": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        try:
            return self.clob.cancel(order_id)
        except Exception as e:
            return {"error": str(e)}

    def cancel_all(self) -> dict:
        try:
            return self.clob.cancel_all()
        except Exception as e:
            return {"error": str(e)}
