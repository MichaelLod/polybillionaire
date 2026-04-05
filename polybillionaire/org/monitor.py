"""Monitor thread — settlement, stop-losses, active position management.

Runs independently every N seconds. Zero LLM cost.
Extracted from runner.py so it can run alongside the swarm.

Active management rules for exponential compounding:
  1. Redeem resolved positions (dead capital = 0% return)
  2. Take profit when token price >= 0.85 (85%+ gain captured)
  3. Capital velocity — sell when daily return from holding < recycling
  4. Cut losers — sell when price dropped 70%+ from entry
  5. Concentration limit — no position > 30% of portfolio
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..trader import PaperTrader
    from .bus import Bus
    from .db import OrgDB
    from .display import Display

# Active management thresholds
TAKE_PROFIT_PRICE = 0.85       # sell when token price >= this
CUT_LOSS_DROP = 0.70           # sell when price dropped this % from entry
CONCENTRATION_MAX = 0.30       # max single position as fraction of portfolio
MIN_DAILY_RETURN_PCT = 0.01    # 1% — sell if holding returns less per day
VELOCITY_LOOKBACK_DAYS = 3     # don't apply velocity rule to fresh positions


class MonitorThread:
    """Independent loop: settlement, stop-loss, position sync, lead cleanup,
    and active position management for compounding velocity.

    Runs every ``interval`` seconds regardless of agent activity.
    Posts events to the bus when positions settle, stop out, or get actively managed.
    """

    def __init__(
        self,
        trader: PaperTrader,
        db: OrgDB,
        bus: Bus,
        display: Display,
        interval: int = 60,
    ) -> None:
        self.trader = trader
        self.db = db
        self.bus = bus
        self.display = display
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="monitor", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def tick(self) -> dict:
        """Run one monitoring pass. Returns stats dict."""
        n_settled = self._settle_resolved()
        n_stopped = self._enforce_stop_losses()
        n_managed = self._manage_positions()
        self._sync_positions()
        self.db.expire_stale_leads()
        self.db.expire_old_trails()
        return {"settled": n_settled, "stopped": n_stopped, "managed": n_managed}

    # ── Private ────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.tick()
                activity = []
                if result["settled"]:
                    activity.append(f"{result['settled']} settled")
                if result["stopped"]:
                    activity.append(f"{result['stopped']} stopped")
                if result["managed"]:
                    activity.append(f"{result['managed']} managed")
                if activity:
                    self.bus.post(
                        "Monitor", "all", "info",
                        f"Monitor: {', '.join(activity)}",
                    )
            except Exception as e:
                self.bus.post(
                    "Monitor", "all", "alert",
                    f"Monitor error: {str(e)[:100]}",
                )
            self._stop.wait(self.interval)

    def _push_portfolio(self) -> None:
        p = self.trader.positions
        self.display.update_portfolio(
            bankroll=self.trader.bankroll,
            deployed=sum(pos.cost for pos in p),
            value=self.trader.total_value,
            pnl=sum(pos.pnl for pos in p),
            positions=len(p),
        )

    def _settle_resolved(self) -> int:
        """Collect winnings from resolved markets."""
        settled = self.trader.settle_resolved()
        for s in settled:
            kind = "trade" if s["won"] else "alert"
            self.bus.post("Trader", "all", kind, s["message"])
            pos = s["position"]
            db_pos = self.db._fetchone(
                "SELECT hypothesis_id FROM positions WHERE token_id = ?",
                (pos.token_id,),
            )
            if db_pos and db_pos["hypothesis_id"]:
                status = "won" if s["won"] else "lost"
                self.db.update_hypothesis(
                    db_pos["hypothesis_id"], status=status,
                )
                self.db.cascade_lead_status(db_pos["hypothesis_id"], status)
        if settled:
            self._push_portfolio()
        return len(settled)

    def _enforce_stop_losses(self) -> int:
        """Refresh prices and auto-sell positions past stop-loss."""
        alerts = self.trader.update_positions(auto_stop=True)
        stopped = 0
        for a in alerts:
            self.bus.post(
                "Trader", "all",
                "trade" if a["sold"] else "alert",
                a["message"],
            )
            if a["sold"]:
                stopped += 1
                pos = a["position"]
                db_pos = self.db._fetchone(
                    "SELECT hypothesis_id FROM positions WHERE token_id = ?",
                    (pos.token_id,),
                )
                if db_pos and db_pos["hypothesis_id"]:
                    self.db.update_hypothesis(
                        db_pos["hypothesis_id"], status="stopped_out",
                    )
                    self.db.cascade_lead_status(
                        db_pos["hypothesis_id"], "stopped_out",
                    )
        if alerts:
            self._push_portfolio()
        return stopped

    def _manage_positions(self) -> int:
        """Active position management for compounding velocity.

        Returns number of positions sold.
        """
        positions = list(self.trader.positions)  # copy, we may modify
        if not positions:
            return 0

        total_value = self.trader.total_value
        if total_value <= 0:
            return 0

        now = time.time()
        sold = 0

        for pos in positions:
            reason = self._should_exit(pos, total_value, now)
            if not reason:
                continue

            ok, msg = self.trader.sell(pos.token_id)
            if ok:
                sold += 1
                self.bus.post(
                    "Monitor", "all", "trade",
                    f"[MANAGE] Sold {pos.outcome} \"{pos.market_question[:40]}\" "
                    f"@ ${pos.current_price:.4f} — {reason}",
                )
            else:
                self.bus.post(
                    "Monitor", "all", "alert",
                    f"[MANAGE] Failed to sell \"{pos.market_question[:40]}\": {msg}",
                )

        if sold:
            self._push_portfolio()
        return sold

    def _should_exit(self, pos, total_value: float, now: float) -> str | None:
        """Check if a position should be actively exited.

        Returns reason string if yes, None if hold.
        """
        price = pos.current_price
        entry = pos.entry_price

        # Rule 1: Take profit — most of the gain is captured
        if price >= TAKE_PROFIT_PRICE:
            return f"take-profit (price ${price:.2f} >= ${TAKE_PROFIT_PRICE})"

        # Rule 2: Cut losers — thesis is dead
        if entry > 0 and price < entry * (1 - CUT_LOSS_DROP):
            return f"cut-loss (dropped {(1 - price/entry)*100:.0f}% from entry)"

        # Rule 3: Concentration — too much in one position
        if total_value > 0 and pos.value / total_value > CONCENTRATION_MAX:
            pct = pos.value / total_value * 100
            return f"concentration ({pct:.0f}% of portfolio > {CONCENTRATION_MAX*100:.0f}% max)"

        # Rule 4: Stale long-shot — bought cheap, hasn't moved, thesis dead
        age_days = (now - pos.opened_at) / 86400
        if entry > 0 and entry < 0.05 and age_days > 14 and price <= entry * 1.5:
            return (
                f"stale long-shot (entry ${entry:.4f}, "
                f"still ${price:.4f} after {age_days:.0f}d)"
            )

        # Rule 5: Far future — capital locked too long for small position
        if pos.end_date:
            try:
                end_dt = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                days_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400
            except (ValueError, TypeError):
                days_left = 0

            if days_left > 90 and pos.value < 1.0:
                return (
                    f"far future (${pos.value:.2f} locked for "
                    f"{days_left:.0f}d — recycle)"
                )

        # Rule 6: Capital velocity — high-value position near expiry
        # where remaining upside per day is tiny
        if pos.end_date and pos.value >= 1.0:
            try:
                end_dt = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                days_left = max(1, (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400)
            except (ValueError, TypeError):
                days_left = 30

            if price > 0.50 and days_left > 0:
                remaining_upside_pct = (1.0 - price) / price
                daily_return = remaining_upside_pct / days_left
                if daily_return < MIN_DAILY_RETURN_PCT:
                    return (
                        f"velocity (${pos.value:.2f} earning "
                        f"{daily_return*100:.2f}%/day < {MIN_DAILY_RETURN_PCT*100:.0f}% "
                        f"threshold, {days_left:.0f}d left)"
                    )

        return None

    def _sync_positions(self) -> None:
        """Sync positions from trader into DB."""
        pos_data = [
            {
                "token_id": p.token_id,
                "market_question": p.market_question,
                "outcome": p.outcome,
                "side": p.side,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "size": p.size,
                "cost": p.cost,
                "pnl": p.pnl,
                "end_date": p.end_date,
            }
            for p in self.trader.positions
        ]
        closed = self.db.sync_positions(pos_data)
        for pos in closed:
            pnl = pos["pnl"] or 0
            mkt = pos["market_question"] or "unknown"
            self.bus.post(
                "Trader", "all", "info",
                f"Position closed: \"{mkt[:50]}\" | PnL: ${pnl:+.4f}",
            )
            if pos.get("hypothesis_id"):
                hyp = self.db._fetchone(
                    "SELECT status FROM hypotheses WHERE id = ?",
                    (pos["hypothesis_id"],),
                )
                if hyp and hyp["status"] not in ("won", "lost", "stopped_out"):
                    self.db.update_hypothesis(
                        pos["hypothesis_id"], status="resolved",
                    )
                    self.db.cascade_lead_status(
                        pos["hypothesis_id"], "resolved",
                    )
