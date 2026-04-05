"""Event-driven agent swarm — replaces the cycle-based pipeline.

Research agents run continuously in their own threads, pushing findings
to a queue. The Reasoning agent monitors the queue, evaluates findings
against live markets, and proposes trades. The TradeExecutor handles
risk checks and execution. All zero-LLM monitoring runs in MonitorThread.
"""

from __future__ import annotations

import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Any

from .agent_config import AgentConfig, load_configs, save_configs, next_agent_name
from .backend import AgentBackend, AgentResponse, create_backend
from .bus import EventBus
from .inspiration import InspirationEngine
from .monitor import MonitorThread
from .runner import (
    _parse_findings,
    _parse_trade_probability,
    _parse_trade_hypothesis_ref,
    _find_hypothesis_by_title,
)

if TYPE_CHECKING:
    from ..trader import PaperTrader
    from .db import OrgDB
    from .display import Display
    from .tools import ToolKit

# Prompt registry — keys match AgentConfig.system_prompt_key
_PROMPT_REGISTRY: dict[str, str] = {}


def _load_prompts() -> None:
    """Lazy-load prompts from agents.py into registry."""
    if _PROMPT_REGISTRY:
        return
    from . import agents as _a
    for name in dir(_a):
        if name.endswith("_PROMPT") and isinstance(getattr(_a, name), str):
            _PROMPT_REGISTRY[name] = getattr(_a, name)


def _get_prompt(key: str) -> str:
    _load_prompts()
    return _PROMPT_REGISTRY.get(key, "")


# ── Agent Worker ───────────────────────────────────────────────

class AgentWorker:
    """Wraps a backend + config, runs in its own daemon thread."""

    def __init__(
        self,
        config: AgentConfig,
        backend: AgentBackend,
        bus: EventBus,
        db: OrgDB,
        display: Display,
        toolkit: ToolKit | None = None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.bus = bus
        self.db = db
        self.display = display
        self.toolkit = toolkit
        self.system_prompt = _get_prompt(config.system_prompt_key)
        self._shared_findings_q: queue.Queue | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._session_id: str | None = None
        self._cost_this_hour: float = 0.0
        self._hour_start: float = time.time()
        self._last_call: float = 0.0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost: float = 0.0
        # 4th dimension — exploration depth
        self._trail_id: int | None = None
        self._depth: int = 0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"agent-{self.config.name}", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> str:
        if not self.alive:
            return "stopped"
        if not self._can_call():
            return "cooldown"
        return "idle"

    # ── Throttling ─────────────────────────────────────────────

    def _can_call(self) -> bool:
        now = time.time()
        # Reset hourly cost
        if now - self._hour_start > 3600:
            self._cost_this_hour = 0.0
            self._hour_start = now
        # Power mode — skip interval throttling
        power = getattr(self.display, "power_mode", False)
        # Min interval (skipped in power mode)
        if not power and now - self._last_call < self.config.min_interval_s:
            return False
        # Cost cap (0 = unlimited, for free local models)
        if self.config.max_cost_per_hour > 0 and self._cost_this_hour >= self.config.max_cost_per_hour:
            return False
        return True

    def _track(self, resp: AgentResponse) -> None:
        self._last_call = time.time()
        self._cost_this_hour += resp.cost_usd
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost += resp.cost_usd
        if resp.session_id:
            self._session_id = resp.session_id
        self.display.update_tokens(
            self.config.name, resp.input_tokens, resp.output_tokens,
        )
        self.db.log_agent_event(
            self.config.name, "call",
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cost_usd=resp.cost_usd,
            model=resp.model or self.config.model,
            duration_s=resp.duration_s,
        )

    # ── Main loop ──────────────────────────────────────────────

    def _run(self) -> None:
        retries = 0
        max_retries = 5
        while not self._stop.is_set() and retries < max_retries:
            try:
                if self.config.role == "research":
                    self._research_loop()
                elif self.config.role == "reasoning":
                    self._reasoning_loop()
                break  # clean exit
            except Exception as e:
                retries += 1
                self.bus.post(
                    self.config.name, "all", "alert",
                    f"Agent error ({retries}/{max_retries}): {str(e)[:80]}",
                )
                if retries < max_retries:
                    self._stop.wait(10 * retries)  # backoff: 10s, 20s, 30s...
        if retries >= max_retries:
            self.bus.post(
                self.config.name, "all", "alert",
                f"Agent gave up after {max_retries} retries",
            )

    # ── Research loop (4th dimension) ────────────────────────────

    def _maybe_pick_up_trail(self, latest_direction: str) -> str | None:
        """Check for abandoned trails from other agents. Returns trail
        context to prepend to prompt, or None if starting fresh."""
        abandoned = self.db.get_abandoned_trails(
            exclude_agent=self.config.name, limit=3,
        )
        if not abandoned:
            return None

        # Pick up the deepest trail — it has the most work invested
        trail = abandoned[0]
        picked = self.db.pick_up_trail(trail["id"], self.config.name)
        if not picked:
            return None

        import json
        crumbs = json.loads(picked["breadcrumbs"] or "[]")
        # The new continuation trail is now our active trail
        active = self.db.get_agent_active_trail(self.config.name)
        if active:
            self._trail_id = active["id"]
            self._depth = active["depth"]

        self.bus.post(
            self.config.name, "all", "info",
            f"Picked up trail from {picked['agent']}: "
            f"\"{picked['topic'][:60]}\" (depth {picked['depth']})",
        )

        trail_context = (
            f"== PICKING UP TRAIL (from {picked['agent']}, depth {picked['depth']}) ==\n"
            f"Topic: {picked['topic']}\n"
            f"Breadcrumbs so far:\n"
        )
        for i, crumb in enumerate(crumbs[-5:], 1):  # last 5 breadcrumbs
            trail_context += f"  {i}. {crumb}\n"
        trail_context += (
            "\nContinue this exploration. Go DEEPER — the previous agent "
            "left off here. Build on what they found, don't repeat it."
        )
        return trail_context

    def _start_fresh_trail(self, direction: str) -> None:
        """Abandon current trail (if any) and start a new one."""
        if self._trail_id:
            self.db.abandon_trail(self._trail_id)
        topic = direction[:200] if direction else "general exploration"
        self._trail_id = self.db.start_trail(
            self.config.name, topic,
        )
        self._depth = 1

    def _research_loop(self) -> None:
        direction_q = self.bus.subscribe("direction_update")
        latest_direction = ""

        while not self._stop.is_set():
            # Drain direction updates
            while not direction_q.empty():
                try:
                    msg = direction_q.get_nowait()
                    # Pick per-agent assignment if available, else general
                    assignments = msg.data.get("assignments", {})
                    new_dir = assignments.get(
                        self.config.name, msg.data.get("direction", ""),
                    )
                    if new_dir and new_dir != latest_direction:
                        latest_direction = new_dir
                        # Direction changed — abandon current trail, start fresh
                        self._start_fresh_trail(new_dir)
                except queue.Empty:
                    break

            if not self._can_call():
                self.display.agent_done(self.config.name, "cooldown")
                self._stop.wait(10)
                continue

            self.display.agent_thinking(self.config.name)

            # 4th dimension: check for trails to pick up, or go deeper
            trail_context = None
            if not self._trail_id:
                # No active trail — try picking up an abandoned one
                trail_context = self._maybe_pick_up_trail(latest_direction)
                if not trail_context:
                    self._start_fresh_trail(latest_direction)

            # Build prompt
            research_context = self.db.build_research_context()
            prompt_parts = []
            if latest_direction:
                prompt_parts.append(f"== DIRECTION ==\n{latest_direction}")
            if trail_context:
                prompt_parts.append(trail_context)
            elif self._depth > 1:
                # Going deeper on own trail — tell agent its depth
                import json
                active = self.db.get_agent_active_trail(self.config.name)
                if active:
                    crumbs = json.loads(active["breadcrumbs"] or "[]")
                    depth_ctx = (
                        f"== DEPTH {self._depth} — GOING DEEPER ==\n"
                        f"You are {self._depth} steps deep exploring: "
                        f"{active['topic'][:100]}\n"
                        f"Your trail so far:\n"
                    )
                    for i, c in enumerate(crumbs[-3:], 1):
                        depth_ctx += f"  {i}. {c}\n"
                    depth_ctx += (
                        "\nDig DEEPER. Follow the most promising thread. "
                        "If you find hard numbers with >= 5% edge, SURFACE IT."
                    )
                    prompt_parts.append(depth_ctx)
            if research_context:
                prompt_parts.append(research_context)
            prompt_parts.append(
                "Search the web for signals that prediction markets "
                "haven't priced in yet. Report findings as:\n"
                "  FINDING N: [title]\n"
                "  Source: [where]\n"
                "  Signal: [what it means]\n"
                "  Confidence: [high/medium/low]\n"
                "  Hypothesis: [which hypothesis this relates to, or 'new']"
            )
            prompt = "\n\n".join(prompt_parts)

            resp = self.backend.send(
                prompt,
                system_prompt=self.system_prompt,
                session_id=self._session_id,
            )
            self._track(resp)

            # Parse findings and push to queue
            findings = _parse_findings(resp.text)
            has_high_confidence = False
            for f in findings:
                hyp_id = None
                if f["hypothesis_ref"]:
                    hyp = _find_hypothesis_by_title(self.db, f["hypothesis_ref"])
                    if hyp:
                        hyp_id = hyp["id"]

                lead_id = self.db.add_lead(
                    title=f["title"], source=f["source"],
                    signal=f["signal"], confidence=f["confidence"],
                    hypothesis_id=hyp_id, found_by=self.config.name,
                )

                if f["confidence"] == "high":
                    has_high_confidence = True
                    # SURFACE — come forth from the 4th dimension
                    if self._trail_id:
                        self.db.surface_trail(self._trail_id, lead_id=lead_id)
                    self.bus.emit(
                        "finding", self.config.name,
                        {
                            "lead_id": lead_id,
                            "finding": f,
                            "agent": self.config.name,
                            "depth": self._depth,
                            "surfaced": True,
                        },
                        content=f"[SURFACED d={self._depth}] {f['title'][:80]}",
                    )
                else:
                    self.bus.emit(
                        "finding", self.config.name,
                        {
                            "lead_id": lead_id,
                            "finding": f,
                            "agent": self.config.name,
                            "depth": self._depth,
                            "surfaced": False,
                        },
                        content=f["title"][:100],
                    )

            # Leave breadcrumb on the trail
            if self._trail_id and resp.text:
                summary = resp.text.strip()[:200]
                breadcrumb = (
                    f"[{self.config.name} d={self._depth}] "
                    f"{len(findings)} findings. {summary}"
                )
                self._depth = self.db.deepen_trail(self._trail_id, breadcrumb)

            # If we surfaced something, start a new trail next cycle
            if has_high_confidence:
                self._trail_id = None
                self._depth = 0

            n = len(findings)
            depth_tag = f" (d={self._depth})" if self._depth > 1 else ""
            surfaced_tag = " SURFACED" if has_high_confidence else ""
            self.display.agent_done(
                self.config.name,
                f"{n} finding{'s' if n != 1 else ''}{depth_tag}{surfaced_tag}"
                if n else f"no findings{depth_tag}",
            )
            self.bus.post(
                self.config.name, "Reasoning", "info",
                resp.text.strip()[:500] if resp.text else "(empty)",
            )

            # Wait before next research cycle
            self._stop.wait(self.config.min_interval_s)

    # ── Reasoning loop ─────────────────────────────────────────

    def _reasoning_loop(self) -> None:
        findings_q = self._shared_findings_q or self.bus.subscribe("finding")
        self.bus.post(
            self.config.name, "all", "info",
            "Awaiting findings...",
        )

        while not self._stop.is_set():
            # Collect a batch of findings (faster in power mode)
            power = getattr(self.display, "power_mode", False)
            batch = self._collect_batch(
                findings_q,
                max_wait=30 if power else 120,
                min_items=1 if power else 2,
            )
            if not batch:
                continue
            if not self._can_call():
                self.display.agent_done(self.config.name, "cooldown")
                self._stop.wait(30)
                continue

            self.display.agent_thinking(self.config.name)

            # Scan markets
            opps = []
            if self.toolkit:
                self.display.tool_call(self.config.name, "scan_markets")
                opps = self.toolkit.scan_markets(limit=30, daily_only=True)

            shown_opps = opps[:20]
            market_lines = "\n".join(
                f"  #{i + 1} {o['side']} \"{o['market']}\" "
                f"@ ${o['price']:.3f} ({o['payout_multiple']:.1f}x, "
                f"vol ${o['volume_24h']:,.0f})"
                for i, o in enumerate(shown_opps)
            )

            # Build findings text from batch
            findings_text = "\n\n".join(
                f"--- {item.get('agent', 'Research')} ---\n"
                f"FINDING: {item.get('finding', {}).get('title', '?')}\n"
                f"Signal: {item.get('finding', {}).get('signal', '?')}\n"
                f"Confidence: {item.get('finding', {}).get('confidence', '?')}"
                for item in batch
            )

            reasoning_context = self.db.build_reasoning_context()
            prompt = (
                f"== RESEARCH FINDINGS ({len(batch)} new) ==\n{findings_text}\n\n"
                f"== AVAILABLE POLYMARKET MARKETS ({len(shown_opps)} shown) ==\n"
                f"{market_lines}\n\n"
            )
            if reasoning_context:
                prompt += f"{reasoning_context}\n\n"
            prompt += (
                "Step 1: Create hypotheses from promising findings.\n"
                "Step 2: Cross-reference against Polymarket prices.\n"
                "Step 3: Propose trades when external sources diverge >= 5%.\n\n"
                "For each promising finding, output:\n"
                "  HYPOTHESIS: [title — specific event]\n"
                "  Thesis: [why the market is wrong]\n"
                "  Category: [sports/politics/crypto/other]\n\n"
                "For each trade, output:\n"
                "  TRADE #N (market number)\n"
                "  Side: YES or NO\n"
                "  Probability: [decimal from external source]\n"
                "  Thesis: why the market is wrong\n"
                "  Edge: [external prob] vs [market price] = [difference]\n"
                "  Hypothesis: [title]\n\n"
                "NO TRADES if research has no external probability numbers.\n"
                "ALWAYS create hypotheses from findings even if no trade yet."
            )

            resp = self.backend.send(
                prompt,
                system_prompt=self.system_prompt,
                session_id=self._session_id,
            )
            self._track(resp)

            # Parse trade proposals
            proposals = []
            for m in re.finditer(
                r"TRADE\s*#(\d+).*?Side:\s*(YES|NO)",
                resp.text, re.IGNORECASE | re.DOTALL,
            ):
                idx = int(m.group(1)) - 1
                side = m.group(2).upper()
                if 0 <= idx < len(shown_opps):
                    prob = _parse_trade_probability(resp.text, idx + 1)
                    hyp_ref = _parse_trade_hypothesis_ref(resp.text, idx + 1)
                    proposals.append({
                        "idx": idx,
                        "side": side,
                        "opp": shown_opps[idx],
                        "probability": prob,
                        "hypothesis_ref": hyp_ref,
                    })

            # Parse sell proposals
            sell_proposals = []
            for sm in re.finditer(
                r"SELL:\s*(\S+).*?Reason:\s*(.+?)(?:\n\n|\nTRADE|\nHYPOTHESIS|\nSELL|\Z)",
                resp.text, re.IGNORECASE | re.DOTALL,
            ):
                token_id = sm.group(1).strip()
                reason = sm.group(2).strip()[:200]
                sell_proposals.append({
                    "token_id": token_id,
                    "reason": reason,
                })

            if sell_proposals:
                for sp in sell_proposals:
                    self.bus.emit(
                        "sell_proposal", self.config.name,
                        sp,
                        content=f"EXIT {sp['token_id']}: {sp['reason'][:80]}",
                    )

            # Parse and create hypotheses
            hyp_count = 0
            for hm in re.finditer(
                r"HYPOTHESIS:\s*(.+?)(?:\n\s*Thesis:\s*(.+?))?(?:\n\s*Category:\s*(.+?))?(?:\n\n|\nTRADE|\Z)",
                resp.text, re.IGNORECASE | re.DOTALL,
            ):
                title = hm.group(1).strip()[:200]
                thesis = (hm.group(2) or "").strip()[:500]
                category = (hm.group(3) or "other").strip().lower()
                existing = _find_hypothesis_by_title(self.db, title)
                if not existing:
                    self.db.add_hypothesis(
                        title=title, thesis=thesis, category=category,
                    )
                    hyp_count += 1

            if proposals:
                for p in proposals:
                    self.bus.emit(
                        "trade_proposal", self.config.name,
                        p, content=f"{p['side']} \"{p['opp']['market'][:50]}\"",
                    )
            parts = []
            if proposals:
                parts.append(f"{len(proposals)} trades")
            if sell_proposals:
                parts.append(f"{len(sell_proposals)} exits")
            if hyp_count:
                parts.append(f"{hyp_count} hypotheses")
            if parts:
                self.display.agent_done(self.config.name, ", ".join(parts))
            else:
                self.display.agent_done(self.config.name, "no edge")

    def _collect_batch(
        self, q: queue.Queue, max_wait: float, min_items: int,
    ) -> list[dict]:
        """Collect findings from queue. Wait up to max_wait or until min_items."""
        batch: list[dict] = []
        deadline = time.time() + max_wait
        while time.time() < deadline and not self._stop.is_set():
            remaining = max(0.1, deadline - time.time())
            try:
                msg = q.get(timeout=min(10, remaining))
                batch.append(msg.data)
                if len(batch) >= min_items:
                    # Drain extras immediately available
                    while not q.empty():
                        try:
                            batch.append(q.get_nowait().data)
                        except queue.Empty:
                            break
                    return batch
            except queue.Empty:
                continue
        return batch


# ── Trade Executor ─────────────────────────────────────────────

class TradeExecutor:
    """Subscribes to trade_proposal events, runs risk checks + execution.

    Pure Python — no LLM calls. Runs in its own thread.
    """

    def __init__(
        self,
        toolkit: ToolKit,
        trader: PaperTrader,
        db: OrgDB,
        bus: EventBus,
        display: Display,
    ) -> None:
        self.toolkit = toolkit
        self.trader = trader
        self.db = db
        self.bus = bus
        self.display = display
        self._proposals_q = bus.subscribe("trade_proposal")
        self._sells_q = bus.subscribe("sell_proposal")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sell_thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="trade-executor", daemon=True,
        )
        self._thread.start()
        self._sell_thread = threading.Thread(
            target=self._run_sells, name="sell-executor", daemon=True,
        )
        self._sell_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._sell_thread:
            self._sell_thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._proposals_q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                self._execute(msg.data)
            except Exception as e:
                self.bus.post(
                    "Trader", "all", "alert",
                    f"Execution error: {str(e)[:100]}",
                )

    def _execute(self, proposal: dict) -> None:
        opp = proposal["opp"]
        proposed_side = proposal["side"]
        agent_prob = proposal.get("probability")

        # Kelly sizing
        if agent_prob and agent_prob > opp["price"]:
            cost = self.toolkit.scanner.resize_with_edge(opp["price"], agent_prob)
            if cost <= 0:
                cost = opp["recommended_bet"]
        else:
            cost = opp["recommended_bet"]

        token_id = opp["token_id"]
        outcome = opp["outcome"]
        if proposed_side.lower() != opp["side"].lower():
            outcome = "No" if opp["outcome"].lower() == "yes" else "Yes"

        # Risk check
        self.display.tool_call("Trader", "check_risk")
        chk = self.toolkit.check_risk(token_id, cost, opp["market"])

        if not chk["approved"]:
            self.bus.post(
                "Trader", "Reasoning", "rejection",
                f"\"{opp['market'][:50]}\" blocked: {chk['reason']}",
            )
            return

        # Execute
        price = opp["price"]
        size = round(cost / price, 1) if price > 0 else 0.1

        self.display.tool_call("Trader", "execute_buy")
        result = self.toolkit.execute_buy(
            token_id, opp["market"], outcome, size,
            end_date=opp.get("end_date", ""),
        )

        if result["success"]:
            self.bus.post("Trader", "all", "trade", result["message"])
            # Link to hypothesis
            hyp_ref = proposal.get("hypothesis_ref", "")
            hyp = _find_hypothesis_by_title(self.db, hyp_ref) if hyp_ref else None
            if hyp:
                self.db.link_position_hypothesis(token_id, hyp["id"])
                self.db.update_hypothesis(hyp["id"], status="traded")
            self.bus.emit(
                "trade_result", "Trader",
                {"success": True, "market": opp["market"][:100]},
            )
        else:
            self.bus.post(
                "Trader", "all", "alert", f"Failed: {result['message']}",
            )

    def _run_sells(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sells_q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                self._execute_sell(msg.data)
            except Exception as e:
                self.bus.post(
                    "Trader", "all", "alert",
                    f"Sell error: {str(e)[:100]}",
                )

    def _execute_sell(self, proposal: dict) -> None:
        token_id = proposal["token_id"]
        reason = proposal.get("reason", "thesis invalidated")

        # Verify we actually hold this position
        pos = next(
            (p for p in self.trader.positions if p.token_id == token_id),
            None,
        )
        if not pos:
            return

        self.display.tool_call("Trader", "execute_sell")
        result = self.toolkit.execute_sell(token_id)

        if result["success"]:
            self.bus.post(
                "Trader", "all", "trade",
                f"[EXIT] Sold \"{pos.market_question[:40]}\" — {reason}",
            )
            # Update linked hypothesis
            db_pos = self.db._fetchone(
                "SELECT hypothesis_id FROM positions WHERE token_id = ?",
                (token_id,),
            )
            if db_pos and db_pos.get("hypothesis_id"):
                self.db.update_hypothesis(
                    db_pos["hypothesis_id"], status="dismissed",
                )
            self.bus.emit(
                "trade_result", "Trader",
                {"success": True, "sell": True, "token_id": token_id},
            )
        else:
            self.bus.post(
                "Trader", "all", "alert",
                f"Exit failed {token_id}: {result['message']}",
            )


# ── Swarm ──────────────────────────────────────────────────────

class Swarm:
    """Top-level orchestrator. Creates workers, manages lifecycle."""

    def __init__(
        self,
        db: OrgDB,
        bus: EventBus,
        toolkit: ToolKit,
        trader: PaperTrader,
        display: Display,
        configs: list[AgentConfig] | None = None,
        daily_only: bool = True,
        fleet_mode: bool = False,
    ) -> None:
        self.db = db
        self.bus = bus
        self.toolkit = toolkit
        self.trader = trader
        self.display = display
        self.daily_only = daily_only
        self.fleet_mode = fleet_mode
        self.configs = configs or load_configs()
        self.workers: dict[str, AgentWorker] = {}
        self.monitor: MonitorThread | None = None
        self.trade_executor: TradeExecutor | None = None
        self.inspiration: InspirationEngine | None = None
        # Shared queue so multiple Reasoning agents compete for findings
        self._shared_findings_q: queue.Queue = bus.subscribe("finding")

    def _start_fast_trade_listener(self) -> None:
        """Listen for high-confidence surfaced findings and try to match
        them directly against Polymarket markets — bypasses Reasoning."""
        q = self.bus.subscribe("finding")

        def _listener():
            while not self._stopped.is_set():
                try:
                    msg = q.get(timeout=5)
                except queue.Empty:
                    continue
                if not msg.data.get("surfaced"):
                    continue
                finding = msg.data.get("finding", {})
                signal = finding.get("signal", "")

                # Extract probability from signal (e.g. "62.5%" or "0.63")
                prob_match = re.search(
                    r"(\d{1,2}(?:\.\d+)?)\s*%|(?:probability|prob)[:\s]*0?\.(\d{2,3})",
                    signal, re.IGNORECASE,
                )
                if not prob_match:
                    continue
                if prob_match.group(1):
                    ext_prob = float(prob_match.group(1)) / 100
                else:
                    ext_prob = float(f"0.{prob_match.group(2)}")
                if ext_prob < 0.05 or ext_prob > 0.95:
                    continue

                # Try to match against Polymarket markets
                try:
                    opps = self.toolkit.scan_markets(limit=30, daily_only=True)
                except Exception:
                    continue
                title_lower = finding.get("title", "").lower()
                for opp in opps:
                    market_lower = opp["market"].lower()
                    # Simple keyword overlap match
                    title_words = set(
                        w for w in title_lower.split() if len(w) > 3
                    )
                    market_words = set(
                        w for w in market_lower.split() if len(w) > 3
                    )
                    overlap = title_words & market_words
                    if len(overlap) < 2:
                        continue

                    # Check edge
                    price = opp["price"]
                    edge = abs(ext_prob - price)
                    if edge < 0.05:
                        continue

                    side = "YES" if ext_prob > price else "NO"
                    self.bus.emit(
                        "trade_proposal", "FastPath",
                        {
                            "idx": 0,
                            "side": side,
                            "opp": opp,
                            "probability": ext_prob,
                            "hypothesis_ref": finding.get("title", ""),
                        },
                        content=f"[FAST] {side} \"{opp['market'][:50]}\" "
                                f"edge={edge:.0%}",
                    )
                    self.display.agent_done(
                        msg.data.get("agent", "FastPath"),
                        f"FAST TRADE edge={edge:.0%}",
                    )
                    break  # one trade per finding

        self._stopped = threading.Event()
        t = threading.Thread(target=_listener, name="fast-trade", daemon=True)
        t.start()
        self._fast_trade_thread = t

    def start(self) -> None:
        """Start all threads: monitor, trade executor, inspiration, agents."""
        self._stopped = threading.Event()

        # Monitor (zero cost, always on)
        self.monitor = MonitorThread(
            self.trader, self.db, self.bus, self.display, interval=60,
        )
        self.monitor.start()

        # Trade executor (zero LLM cost)
        self.trade_executor = TradeExecutor(
            self.toolkit, self.trader, self.db, self.bus, self.display,
        )
        self.trade_executor.start()

        # Inspiration engine (zero cost by default)
        self.inspiration = InspirationEngine(
            self.db, self.bus, interval_s=300,
            agent_configs=self.configs,
        )
        self.inspiration.start()

        # Spawn configured agents (skip in fleet mode — user deploys manually)
        if not self.fleet_mode:
            for cfg in self.configs:
                if cfg.enabled:
                    self.spawn_agent(cfg)

        # Fast trade path — bypasses Reasoning for high-confidence findings
        self._start_fast_trade_listener()

        # Push initial portfolio to display immediately
        p = self.trader.positions
        self.display.update_portfolio(
            bankroll=self.trader.bankroll,
            deployed=sum(pos.cost for pos in p),
            value=self.trader.total_value,
            pnl=sum(pos.pnl for pos in p),
            positions=len(p),
        )

        self.bus.post(
            "System", "all", "info",
            f"Swarm started: {len(self.workers)} agents, "
            f"monitor + trader + fast-path + inspiration running",
        )

    def spawn_agent(self, config: AgentConfig) -> AgentWorker:
        """Dynamically add an agent at runtime."""
        backend = create_backend(
            config.backend_type,
            model=config.model,
            tools=config.tools if config.tools is not None else None,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )
        worker = AgentWorker(
            config, backend, self.bus, self.db, self.display,
            toolkit=self.toolkit if config.role == "reasoning" else None,
        )
        if config.role == "reasoning":
            worker._shared_findings_q = self._shared_findings_q
        self.workers[config.name] = worker
        worker.start()
        self.bus.post(
            "System", "all", "info",
            f"Agent spawned: {config.name} ({config.model}, {config.backend_type})",
        )
        return worker

    def kill_agent(self, name: str) -> None:
        """Remove an agent at runtime."""
        if name in self.workers:
            self.workers[name].stop()
            del self.workers[name]
            self.bus.post("System", "all", "info", f"Agent killed: {name}")

    def spawn_from_preset(self, preset_key: str) -> AgentWorker:
        """Spawn an agent from a named preset template."""
        from .agent_config import PRESETS
        template = PRESETS.get(preset_key)
        if not template:
            raise ValueError(f"Unknown preset: {preset_key!r}")
        import copy
        config = copy.deepcopy(template)
        config.name = next_agent_name(
            list(self.configs) + [w.config for w in self.workers.values()],
            config.role,
        )
        self.configs.append(config)
        return self.spawn_agent(config)

    def stop(self) -> None:
        """Gracefully shut down everything."""
        if hasattr(self, "_stopped"):
            self._stopped.set()
        for w in self.workers.values():
            w.stop()
        if self.trade_executor:
            self.trade_executor.stop()
        if self.inspiration:
            self.inspiration.stop()
        if self.monitor:
            self.monitor.stop()

        total_cost = sum(w.total_cost for w in self.workers.values())
        total_in = sum(w.total_input_tokens for w in self.workers.values())
        total_out = sum(w.total_output_tokens for w in self.workers.values())
        self.bus.post(
            "System", "all", "info",
            f"Swarm stopped. Cost: ${total_cost:.4f}, "
            f"Tokens: {total_in + total_out:,}",
        )

    def run(self) -> None:
        """Start the swarm and block until KeyboardInterrupt."""
        self.display.start()
        self.display.banner()
        self.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

        self.stop()
        total_in = sum(w.total_input_tokens for w in self.workers.values())
        total_out = sum(w.total_output_tokens for w in self.workers.values())
        self.display.shutdown(
            f"[bold]Final:[/bold] Bankroll ${self.trader.bankroll:.2f} | "
            f"Positions: {len(self.trader.positions)} | "
            f"Value: ${self.trader.total_value:.2f} | "
            f"Tokens: {total_in + total_out:,}"
        )
        self.db.close()
