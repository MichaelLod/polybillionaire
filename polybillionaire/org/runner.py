"""Organisation runner — orchestrates Claude Code agent sessions.

Cycle flow:
  1. CEO    — reviews portfolio + institutional memory, sets strategy
  2. Research — explores the web for signals, writes leads to DB
  3. Python  — scans Polymarket for available markets
  4. Reasoning — matches research findings to markets, proposes trades
  5. Python  — mechanical risk checks + trade execution
  6. CEO    — summarises cycle, updates direction
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from ..client import PolymarketClient
from ..trader import PaperTrader
from .agents import create_agents
from .bus import Bus
from .dashboard import Dashboard
from .db import OrgDB
from .display import Display
from .tools import ToolKit


def _recent_context(bus: Bus, last_n: int = 12) -> str:
    msgs = bus.log[-last_n:]
    if not msgs:
        return "(no messages yet)"
    return "\n".join(
        f"[{m.kind.upper()}] {m.sender} -> {m.to}: {m.content}"
        for m in msgs
    )


# ── Parsers ─────────────────────────────────────────────────────

def _parse_findings(text: str) -> list[dict]:
    """Parse Research output into structured findings."""
    findings: list[dict] = []
    # Match FINDING N: title, then capture subsequent fields
    blocks = re.split(r"(?=FINDING\s+\d+\s*:)", text, flags=re.IGNORECASE)
    for block in blocks:
        m = re.match(r"FINDING\s+\d+\s*:\s*(.+)", block, re.IGNORECASE)
        if not m:
            continue
        title = m.group(1).strip()

        source = ""
        sm = re.search(r"Source:\s*(.+)", block, re.IGNORECASE)
        if sm:
            source = sm.group(1).strip()

        signal = ""
        sg = re.search(r"Signal:\s*(.+)", block, re.IGNORECASE)
        if sg:
            signal = sg.group(1).strip()

        confidence = "medium"
        cm = re.search(r"Confidence:\s*(high|medium|low)", block, re.IGNORECASE)
        if cm:
            confidence = cm.group(1).lower()

        hypothesis_ref = ""
        hm = re.search(r"Hypothesis:\s*(.+)", block, re.IGNORECASE)
        if hm:
            hypothesis_ref = hm.group(1).strip()

        findings.append({
            "title": title[:200],
            "source": source[:300],
            "signal": signal[:300],
            "confidence": confidence,
            "hypothesis_ref": hypothesis_ref,
        })
    return findings


def _parse_ceo_direction(text: str) -> str | None:
    """Extract DIRECTION: ... from CEO output."""
    m = re.search(r"DIRECTION:\s*(.+?)(?=\n\n|\nNEW HYPOTHESIS|\nDISMISS|\nSTALE|\Z)",
                  text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _parse_ceo_new_hypotheses(text: str) -> list[dict]:
    """Extract NEW HYPOTHESIS blocks from CEO output."""
    hyps: list[dict] = []
    blocks = re.split(r"(?=NEW HYPOTHESIS:)", text, flags=re.IGNORECASE)
    for block in blocks:
        m = re.match(r"NEW HYPOTHESIS:\s*(.+)", block, re.IGNORECASE)
        if not m:
            continue
        title = m.group(1).strip()

        thesis = ""
        tm = re.search(r"Thesis:\s*(.+)", block, re.IGNORECASE)
        if tm:
            thesis = tm.group(1).strip()

        category = ""
        cm = re.search(r"Category:\s*(\w+)", block, re.IGNORECASE)
        if cm:
            category = cm.group(1).lower()

        hyps.append({"title": title[:200], "thesis": thesis[:500], "category": category})
    return hyps


def _parse_ceo_dismissals(text: str) -> list[str]:
    """Extract DISMISS HYPOTHESIS: ... from CEO output."""
    return [
        m.group(1).strip()
        for m in re.finditer(r"DISMISS HYPOTHESIS:\s*(.+)", text, re.IGNORECASE)
    ]


def _parse_ceo_stale(text: str) -> list[str]:
    """Extract STALE HYPOTHESIS: ... from CEO output."""
    return [
        m.group(1).strip()
        for m in re.finditer(r"STALE HYPOTHESIS:\s*(.+)", text, re.IGNORECASE)
    ]


def _parse_trade_edge(text: str, trade_num: int) -> tuple[float | None, float | None]:
    """Extract edge estimates from Reasoning's TRADE #N block.

    Looks for patterns like:
      Edge: 50% vs 97% = -47%
      Edge: 0.50 vs 0.97
    """
    pattern = (
        rf"TRADE\s*#{trade_num}\b.*?"
        rf"Edge:\s*(\d+(?:\.\d+)?)\s*%?\s*vs\s*(\d+(?:\.\d+)?)\s*%?"
        rf".*?(?=TRADE\s*#|\Z)"
    )
    m = re.search(pattern, text, re.I | re.DOTALL)
    if not m:
        return None, None
    our = float(m.group(1))
    market = float(m.group(2))
    # Normalize to 0-1 if given as percentages
    if our > 1:
        our /= 100
    if market > 1:
        market /= 100
    return our, market


def _parse_trade_probability(text: str, trade_num: int) -> float | None:
    """Extract Probability: ... from a TRADE #N block.

    Looks for patterns like:
      Probability: 0.70
      Probability: 70%
    Falls back to the edge-based estimate if present.
    """
    pattern = (
        rf"TRADE\s*#{trade_num}\b.*?"
        rf"Probability:\s*(\d+(?:\.\d+)?)\s*%?"
        rf".*?(?=TRADE\s*#|\Z)"
    )
    m = re.search(pattern, text, re.I | re.DOTALL)
    if m:
        val = float(m.group(1))
        if val > 1:
            val /= 100
        if 0 < val < 1:
            return val
    # Fall back to Edge: our_estimate vs market
    our, _ = _parse_trade_edge(text, trade_num)
    return our


def _parse_trade_hypothesis_ref(text: str, trade_num: int) -> str:
    """Extract Hypothesis: ... from a TRADE #N block."""
    pattern = (
        rf"TRADE\s*#{trade_num}\b.*?"
        rf"Hypothesis:\s*(.+?)(?:\n|$)"
        rf".*?(?=TRADE\s*#|\Z)"
    )
    m = re.search(pattern, text, re.I | re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_research_assignments(text: str, num_agents: int) -> dict[str, str]:
    """Extract ASSIGN RESEARCH-N: ... from CEO output.

    Returns a dict mapping agent name -> assignment text.
    Falls back to a generic assignment if CEO didn't use the format.
    """
    assignments: dict[str, str] = {}
    for m in re.finditer(
        r"ASSIGN\s+(RESEARCH-\d+)\s*:\s*(.+?)(?=\nASSIGN\s+RESEARCH-|\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    ):
        name = m.group(1).upper()
        task = m.group(2).strip()
        if task:
            assignments[name] = task

    # If CEO didn't assign specific tasks, give all agents the full directive
    if not assignments:
        for i in range(num_agents):
            assignments[f"RESEARCH-{i + 1}"] = ""

    return assignments


def _find_hypothesis_by_title(db: OrgDB, title: str) -> dict | None:
    """Fuzzy-match a hypothesis by title."""
    if not title or title.lower() == "new":
        return None
    hyps = db.get_active_hypotheses()
    title_lower = title.lower()
    for h in hyps:
        if title_lower in h["title"].lower() or h["title"].lower() in title_lower:
            return h
    return None


# ── Organization ────────────────────────────────────────────────

class Organization:
    def __init__(
        self,
        client: PolymarketClient,
        trader: PaperTrader,
        *,
        model: str = "sonnet",
        ceo_model: str = "haiku",
        research_model: str = "haiku",
        reasoning_model: str = "sonnet",
        max_proposals: int = 5,
        scan_limit: int = 30,
        num_researchers: int = 2,
        simple: bool = False,
        daily_only: bool = True,
    ) -> None:
        self.bus = Bus()
        self.trader = trader
        self.toolkit = ToolKit(client, trader)
        self.max_proposals = max_proposals
        self.scan_limit = scan_limit
        self.num_researchers = num_researchers
        self.daily_only = daily_only
        self.db = OrgDB()

        if simple:
            self.display: Dashboard | Display = Display()
        else:
            self.display = Dashboard(
                model=model, db=self.db, num_researchers=num_researchers,
                trader=self.trader,
            )

        self.bus.on_message(self.display.render)
        self.ceo, self.researchers, self.reasoning = create_agents(
            self.display, model=model, num_researchers=num_researchers,
            ceo_model=ceo_model, research_model=research_model,
            reasoning_model=reasoning_model,
        )
        self._last_ceo_text: str = ""

    def _push_portfolio(self) -> None:
        p = self.trader.positions
        self.display.update_portfolio(
            bankroll=self.trader.bankroll,
            deployed=sum(pos.cost for pos in p),
            value=self.trader.total_value,
            pnl=sum(pos.pnl for pos in p),
            positions=len(p),
        )

    def _settle_resolved(self, cycle: int | None = None) -> int:
        """Collect winnings from resolved markets — the compounding engine."""
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
                    updated_cycle=cycle,
                )
                self.db.cascade_lead_status(db_pos["hypothesis_id"], status)
        if settled:
            self._push_portfolio()
        return len(settled)

    def _enforce_stop_losses(self, cycle: int | None = None) -> int:
        """Refresh prices and auto-sell any position past the stop-loss."""
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
                        updated_cycle=cycle,
                    )
                    self.db.cascade_lead_status(
                        db_pos["hypothesis_id"], "stopped_out",
                    )
        if alerts:
            self._push_portfolio()
        return stopped

    def _sync_positions(self, cycle: int | None = None) -> None:
        """Sync positions from trader into DB. Logs closed positions."""
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
        closed = self.db.sync_positions(pos_data, cycle=cycle)
        for pos in closed:
            pnl = pos["pnl"] or 0
            mkt = pos["market_question"] or "unknown"
            self.bus.post(
                "Trader", "all", "info",
                f"Position closed: \"{mkt[:50]}\" | PnL: ${pnl:+.4f}",
            )
            if pos.get("hypothesis_id"):
                # Don't overwrite specific statuses set by settle/stop-loss
                hyp = self.db._fetchone(
                    "SELECT status FROM hypotheses WHERE id = ?",
                    (pos["hypothesis_id"],),
                )
                if hyp and hyp["status"] not in ("won", "lost", "stopped_out"):
                    self.db.update_hypothesis(
                        pos["hypothesis_id"], status="resolved",
                        updated_cycle=cycle,
                    )
                    self.db.cascade_lead_status(
                        pos["hypothesis_id"], "resolved",
                    )

    def _apply_ceo_output(self, text: str, cycle: int) -> None:
        """Parse CEO output and apply direction/hypothesis changes to DB."""
        # Direction update
        direction = _parse_ceo_direction(text)
        if direction:
            self.db.set_direction(direction, cycle=cycle)

        # New hypotheses
        current_dir = self.db.get_direction()
        dir_id = current_dir["id"] if current_dir else None
        for h in _parse_ceo_new_hypotheses(text):
            self.db.add_hypothesis(
                title=h["title"], thesis=h["thesis"],
                category=h["category"], direction_id=dir_id,
                cycle=cycle,
            )

        # Dismissals
        for title in _parse_ceo_dismissals(text):
            hyp = _find_hypothesis_by_title(self.db, title)
            if hyp:
                self.db.update_hypothesis(hyp["id"], status="dismissed",
                                          updated_cycle=cycle)
                self.db.cascade_lead_status(hyp["id"], "dismissed")

        # Stale
        for title in _parse_ceo_stale(text):
            hyp = _find_hypothesis_by_title(self.db, title)
            if hyp:
                self.db.update_hypothesis(hyp["id"], status="stale",
                                          updated_cycle=cycle)
                self.db.cascade_lead_status(hyp["id"], "stale")

    def _apply_research_output(self, text: str, cycle: int) -> list[int]:
        """Parse Research findings and create leads in DB. Returns lead IDs."""
        findings = _parse_findings(text)
        lead_ids: list[int] = []
        for f in findings:
            # Try to link to existing hypothesis
            hyp_id = None
            if f["hypothesis_ref"]:
                hyp = _find_hypothesis_by_title(self.db, f["hypothesis_ref"])
                if hyp:
                    hyp_id = hyp["id"]

            lead_id = self.db.add_lead(
                title=f["title"], source=f["source"], signal=f["signal"],
                confidence=f["confidence"], hypothesis_id=hyp_id,
                cycle=cycle,
            )
            # Also log to research trail
            self.db.log_research(
                lead_id, cycle, f["signal"] or f["title"],
                confidence=f["confidence"],
            )
            lead_ids.append(lead_id)
        return lead_ids

    def _apply_reasoning_output(
        self, text: str, cycle: int, shown_opps: list[dict],
        lead_ids: list[int],
    ) -> None:
        """Parse Reasoning trade proposals and create/update hypotheses."""
        for m in re.finditer(
            r"TRADE\s*#(\d+).*?Side:\s*(YES|NO)",
            text, re.IGNORECASE | re.DOTALL,
        ):
            idx = int(m.group(1)) - 1
            if idx < 0 or idx >= len(shown_opps):
                continue
            opp = shown_opps[idx]
            side = m.group(2).upper()
            trade_num = idx + 1

            # Extract edge estimates
            our_prob, mkt_price = _parse_trade_edge(text, trade_num)

            # Extract thesis
            thesis = ""
            tp = re.search(
                rf"TRADE\s*#{trade_num}\b.*?"
                rf"(?:Thesis|Why|Reason)[:\-—]?\s*(.*?)"
                rf"(?=TRADE\s*#|\Z)",
                text, re.I | re.DOTALL,
            )
            if tp:
                thesis = tp.group(1).strip()[:300]

            # Check if hypothesis already exists for this market
            hyp_ref = _parse_trade_hypothesis_ref(text, trade_num)
            hyp = _find_hypothesis_by_title(self.db, hyp_ref) if hyp_ref else None

            if hyp:
                # Update existing hypothesis with edge info
                updates: dict = {
                    "market_question": opp["market"],
                    "side": side,
                    "status": "edge_found",
                    "updated_cycle": cycle,
                }
                if our_prob is not None:
                    updates["our_probability"] = our_prob
                if mkt_price is not None:
                    updates["market_price"] = mkt_price
                if our_prob is not None and mkt_price is not None:
                    updates["edge"] = our_prob - mkt_price
                self.db.update_hypothesis(hyp["id"], **updates)
            else:
                # Create new hypothesis from trade proposal
                edge = None
                if our_prob is not None and mkt_price is not None:
                    edge = our_prob - mkt_price

                current_dir = self.db.get_direction()
                dir_id = current_dir["id"] if current_dir else None

                hyp_id = self.db.add_hypothesis(
                    title=opp["market"][:200],
                    thesis=thesis,
                    category="",
                    direction_id=dir_id,
                    cycle=cycle,
                )
                self.db.update_hypothesis(
                    hyp_id,
                    market_question=opp["market"],
                    our_probability=our_prob,
                    market_price=mkt_price,
                    edge=edge,
                    side=side,
                    status="edge_found",
                )

                # Link any related leads to this hypothesis
                for lid in lead_ids:
                    lead = self.db._fetchone("SELECT * FROM leads WHERE id = ?", (lid,))
                    if lead and not lead["hypothesis_id"]:
                        self.db.update_lead(lid, hypothesis_id=hyp_id)

    # ── Cycle ───────────────────────────────────────────────────

    def run_cycle(self, cycle: int) -> dict:
        cycle_start = time.time()
        self.display.cycle_start(cycle)
        self._push_portfolio()

        # 0 ── Free monitoring (always runs, 0 tokens) ──────────
        n_settled = self._settle_resolved(cycle)
        n_stopped = self._enforce_stop_losses(cycle)
        self._sync_positions(cycle)
        self.db.expire_stale_leads()

        # Log cycle start
        self.db.start_cycle(
            cycle, self.trader.bankroll, len(self.trader.positions),
        )

        # ── Decision gate: is there work worth spending tokens on? ──
        _in_power = hasattr(self.display, "power_mode") and self.display.power_mode
        capital_freed = n_settled > 0 or n_stopped > 0
        min_bet = self.trader.bankroll * self.trader.risk.max_bet_fraction
        can_deploy = (
            min_bet > 0.01
            and len(self.trader.positions) < self.trader.risk.max_positions
        )
        has_pending_hints = (
            hasattr(self.display, "get_pending_hints")
            and bool(self.display.get_pending_hints())
        )

        run_full = _in_power or capital_freed or can_deploy or has_pending_hints

        if not run_full:
            tag = "capital freed" if capital_freed else "waiting for resolutions"
            self.bus.post(
                "System", "all", "info",
                f"Monitor cycle — portfolio deployed, {tag}",
            )
            self.display.agent_done("CEO", "monitor")
            self.display.agent_done("Reasoning", "monitor")
            self.display.agent_done("Trader", "monitor")
            self.db.end_cycle(
                cycle,
                bankroll_end=self.trader.bankroll,
                trades_executed=0, trades_rejected=0,
                input_tokens=0, output_tokens=0,
                duration_s=round(time.time() - cycle_start, 1),
                strategy=self._last_ceo_text[:200] if self._last_ceo_text else "",
                summary=f"Monitor — {tag}. 0 tokens.",
            )
            return {
                "type": "monitor", "executed": 0, "rejected": 0,
                "is_hold": False, "capital_freed": capital_freed,
            }

        # 1 ── CEO sets strategy ─────────────────────────────────

        # Collect human hints
        hints_block = ""
        has_hints = False
        if hasattr(self.display, "get_pending_hints"):
            pending = self.display.get_pending_hints()
            if pending:
                has_hints = True
                hint_lines = "\n".join(f"  - {h}" for h in pending)
                hints_block = (
                    f"\n\n== HUMAN INTUITION ==\n"
                    f"The human operator shared these hints. Investigate them — "
                    f"if any look promising, create a NEW HYPOTHESIS and tell "
                    f"Research to dig in. If not useful, briefly explain why.\n"
                    f"{hint_lines}"
                )
                for h in pending:
                    self.db.add_hint(h, cycle)
                self.display.mark_hints_seen()

        # Skip CEO when direction is fresh and no new hints.
        # Pace mode: skip 2 of every 3 cycles. Power mode: skip every other.
        _skip_freq = 2 if _in_power else 3  # run CEO every Nth cycle
        skip_ceo = (
            cycle > 1
            and cycle % _skip_freq != 1
            and self._last_ceo_text
            and not has_hints
        )

        if skip_ceo:
            ceo_text = self._last_ceo_text
            self.display.agent_done("CEO", "reusing strategy")
            self.bus.post("CEO", "all", "info", "(Reusing previous direction — skipped)")
        else:
            self.display.agent_thinking("CEO")
            portfolio = self.toolkit.get_portfolio()
            memory_context = self.db.build_ceo_context()

            daily_note = (
                "\n\n== CONSTRAINT: DAILY TRADES ONLY ==\n"
                "We ONLY trade markets that resolve within 24 hours. "
                "Focus research on events happening TODAY. No long-dated positions.\n"
            ) if self.daily_only else ""

            # Calculate drought — how many cycles since last trade
            last_trade_cycle = self.db._fetchone(
                "SELECT MAX(id) as c FROM cycles WHERE trades_executed > 0",
            )
            ltc = (last_trade_cycle["c"] or 0) if last_trade_cycle else 0
            drought = cycle - ltc if ltc else cycle
            drought_warning = ""
            if drought >= 5:
                drought_warning = (
                    f"\n\n== CRITICAL: TRADE DROUGHT ({drought} CYCLES, ZERO TRADES) ==\n"
                    f"You have gone {drought} consecutive cycles without executing a "
                    f"single trade. The bankroll is IDLE — compounding cannot happen "
                    f"at zero. Your current strategy is FAILING because it produces "
                    f"no trades. You MUST:\n"
                    f"1. DISMISS hypotheses about future events that cannot be traded TODAY\n"
                    f"2. Look at what is tradeable RIGHT NOW — sports games, daily events\n"
                    f"3. Output STRATEGY: DEPLOY and assign research to find cross-referenced "
                    f"edge on TODAY's markets\n"
                    f"Waiting for 'perfect' setups while the bankroll sits idle is the "
                    f"OPPOSITE of exponential growth. TRADE or explain why NOTHING on "
                    f"Polymarket today has >= 5% edge vs bookmakers.\n"
                )

            # Hypothesis cap warning
            active_count = self.db._fetchone(
                "SELECT COUNT(*) as c FROM hypotheses WHERE status = 'active'",
            )
            n_active = active_count["c"] if active_count else 0
            hyp_cap_warning = ""
            if n_active >= 5:
                hyp_cap_warning = (
                    f"\n\n== HYPOTHESIS CAP: {n_active} ACTIVE (max 5) ==\n"
                    f"You have {n_active} active hypotheses. DISMISS at least "
                    f"{n_active - 4} before creating any new ones.\n"
                )

            ceo_text = self.ceo.act(
                f"Cycle {cycle}.\n"
                f"Portfolio: {json.dumps(portfolio, indent=2)}\n\n"
                f"{memory_context}\n\n"
                f"Set strategy and direct Research on what to investigate."
                f"{daily_note}{hints_block}{drought_warning}{hyp_cap_warning}"
            )
            self.bus.post("CEO", "all", "directive", ceo_text.strip())
            self._apply_ceo_output(ceo_text, cycle)
            self.display.agent_done("CEO")
            self._last_ceo_text = ceo_text

        # ── Light cycle: skip research+reasoning if CEO says HOLD ──
        # Power mode overrides — always run full pipeline
        _in_power = hasattr(self.display, "power_mode") and self.display.power_mode
        _ceo_upper = ceo_text.upper()
        has_deploy = "DEPLOY" in _ceo_upper or "BALANCED" in _ceo_upper
        is_hold = (
            not _in_power
            and (
                "STRATEGY: HOLD" in _ceo_upper
                or "SET STRATEGY: HOLD" in _ceo_upper
                or "\nHOLD" in _ceo_upper
                or not has_deploy
            )
        )
        if is_hold:
            self.bus.post(
                "CEO", "all", "info",
                "Strategy is HOLD — light cycle (skipping research + reasoning to conserve tokens)",
            )
            self.display.agent_done("Reasoning", "skipped (HOLD)")
            self.display.agent_done("Trader", "idle")

            self.db.end_cycle(
                cycle,
                bankroll_end=self.trader.bankroll,
                trades_executed=0, trades_rejected=0,
                input_tokens=self.ceo.total_input_tokens,
                output_tokens=self.ceo.total_output_tokens,
                duration_s=round(time.time() - cycle_start, 1),
                strategy=ceo_text[:200],
                summary="Light cycle — HOLD strategy, research skipped.",
            )
            return {"type": "hold", "executed": 0, "rejected": 0, "is_hold": True}

        # 2 ── Research explores the web (parallel) ───────────────
        research_context = self.db.build_research_context()
        assignments = _parse_research_assignments(
            ceo_text, len(self.researchers),
        )

        def _run_researcher(agent, assignment):
            self.display.agent_thinking(agent.name)
            self.display.tool_call(agent.name, "WebSearch")

            prompt = f"Cycle {cycle}. CEO directive:\n{ceo_text}\n\n"
            if assignment:
                prompt += f"== YOUR ASSIGNMENT ==\n{assignment}\n\n"
            if research_context:
                prompt += f"{research_context}\n\n"
            prompt += (
                "Search the web for signals that prediction markets "
                "haven't priced in yet. Report findings as:\n"
                "  FINDING N: [title]\n"
                "  Source: [where]\n"
                "  Signal: [what it means]\n"
                "  Confidence: [high/medium/low]\n"
                "  Hypothesis: [which hypothesis this relates to, or 'new']"
            )

            result = agent.act(prompt)
            self.display.agent_done(agent.name)
            return agent.name, result

        # Fan out research to all agents in parallel
        all_findings: list[str] = []
        all_lead_ids: list[int] = []

        with ThreadPoolExecutor(max_workers=len(self.researchers)) as pool:
            futures = {}
            for agent in self.researchers:
                assignment = assignments.get(agent.name.upper(), "")
                futures[pool.submit(_run_researcher, agent, assignment)] = agent

            for future in as_completed(futures):
                agent = futures[future]
                try:
                    name, result = future.result()
                    all_findings.append(f"--- {name} ---\n{result.strip()}")
                    leads = self._apply_research_output(result, cycle)
                    all_lead_ids.extend(leads)
                    self.bus.post(name, "Reasoning", "info", result.strip())
                except Exception as e:
                    self.bus.post(
                        agent.name, "all", "alert",
                        f"Research failed: {str(e)[:100]}",
                    )
                    self.display.agent_done(agent.name, "error")

        research_text = "\n\n".join(all_findings)
        lead_ids = all_lead_ids

        # 3 ── Python scans Polymarket ───────────────────────────
        self.display.tool_call("Reasoning", "scan_markets")
        opps = self.toolkit.scan_markets(
            limit=self.scan_limit, daily_only=self.daily_only,
        )

        for o in opps[:20]:
            self.db.snapshot_market(
                o["market"], o["price"], o.get("volume_24h", 0),
                o.get("token_id", ""), cycle,
            )

        shown_opps = opps[:20]
        market_lines = "\n".join(
            f"  #{i + 1} {o['side']} \"{o['market']}\" "
            f"@ ${o['price']:.3f} ({o['payout_multiple']:.1f}x, "
            f"vol ${o['volume_24h']:,.0f})"
            for i, o in enumerate(shown_opps)
        )

        # 4 ── Reasoning matches findings to markets ─────────────
        self.display.agent_thinking("Reasoning")

        reasoning_context = self.db.build_reasoning_context()
        daily_reasoning = (
            "== CONSTRAINT: DAILY TRADES ONLY ==\n"
            "All markets below resolve within 24h. "
            "Assess probability of outcome by end of day.\n\n"
        ) if self.daily_only else ""

        reasoning_prompt = (
            f"Cycle {cycle}.\n\n"
            f"{daily_reasoning}"
            f"== RESEARCH FINDINGS ==\n{research_text}\n\n"
            f"== AVAILABLE POLYMARKET MARKETS ({len(shown_opps)} shown) ==\n"
            f"{market_lines}\n\n"
        )
        if reasoning_context:
            reasoning_prompt += f"{reasoning_context}\n\n"
        reasoning_prompt += (
            f"Cross-reference Research-1's external odds against Polymarket "
            f"prices. Use Research-2's edge analysis to confirm WHY the market "
            f"is wrong. Only trade when external sources diverge >= 5%.\n\n"
            f"For each trade, output:\n"
            f"  TRADE #N (market number)\n"
            f"  Side: YES or NO\n"
            f"  Probability: [decimal from external source, e.g. 0.63]\n"
            f"  External source: [bookmaker/model and their number]\n"
            f"  Thesis: why the market is wrong\n"
            f"  Edge: [external prob] vs [market price] = [difference]\n"
            f"  Hypothesis: [title of hypothesis this tests]\n\n"
            f"More trades with real edge = faster compounding. Propose ALL "
            f"trades where external sources show >= 5% edge. "
            f"Max {self.max_proposals} trades. "
            f"NO TRADES if research has no external probability numbers."
        )

        reasoning_text = self.reasoning.act(reasoning_prompt)
        self._apply_reasoning_output(reasoning_text, cycle, shown_opps, lead_ids)

        # Parse TRADE #N and Side for execution
        trade_proposals: list[dict] = []
        for m in re.finditer(
            r"TRADE\s*#(\d+).*?Side:\s*(YES|NO)",
            reasoning_text,
            re.IGNORECASE | re.DOTALL,
        ):
            idx = int(m.group(1)) - 1
            side = m.group(2).upper()
            if 0 <= idx < len(shown_opps) and idx not in [t["idx"] for t in trade_proposals]:
                trade_proposals.append({"idx": idx, "side": side})

        if not trade_proposals:
            self.bus.post(
                "Reasoning", "all", "info",
                "No actionable edge found. Holding cash this cycle.",
            )
            self.display.agent_done("Reasoning", "no edge")
        else:
            for tp in trade_proposals:
                idx = tp["idx"]
                opp = shown_opps[idx]
                thesis = ""
                pattern = (
                    rf"TRADE\s*#{idx + 1}\b.*?"
                    rf"(?:Thesis|Why|Reason)[:\-—]?\s*(.*?)"
                    rf"(?=TRADE\s*#|\Z)"
                )
                match = re.search(pattern, reasoning_text, re.I | re.DOTALL)
                if match:
                    thesis = match.group(1).strip()[:300]

                self.bus.post(
                    "Reasoning", "Trader", "proposal",
                    f"{tp['side']} \"{opp['market'][:55]}\"\n"
                    f"Price: ${opp['price']:.3f} | "
                    f"Payout: {opp['payout_multiple']:.1f}x\n"
                    + (f"Thesis: {thesis}" if thesis else ""),
                )

            self.display.agent_done("Reasoning", f"{len(trade_proposals)} trades")

        # 5 ── Risk checks + Execution ───────────────────────────
        executed = 0
        rejected = 0

        if trade_proposals:
            self.display.agent_thinking("Trader")

            for tp in trade_proposals:
                idx = tp["idx"]
                opp = shown_opps[idx]

                # Use agent's probability estimate for Kelly sizing
                agent_prob = _parse_trade_probability(reasoning_text, idx + 1)
                if agent_prob and agent_prob > opp["price"]:
                    cost = self.toolkit.scanner.resize_with_edge(
                        opp["price"], agent_prob,
                    )
                    if cost <= 0:
                        cost = opp["recommended_bet"]
                else:
                    cost = opp["recommended_bet"]

                proposed_side = tp["side"]
                token_id = opp["token_id"]
                outcome = opp["outcome"]

                if proposed_side.lower() != opp["side"].lower():
                    outcome = "No" if opp["outcome"].lower() == "yes" else "Yes"

                self.display.tool_call("Trader", "check_risk")
                chk = self.toolkit.check_risk(token_id, cost, opp["market"])

                if not chk["approved"]:
                    self.bus.post(
                        "Trader", "Reasoning", "rejection",
                        f"\"{opp['market'][:50]}\" blocked: {chk['reason']}",
                    )
                    self.db.log_rejection(
                        cycle, opp["market"], proposed_side,
                        cost, chk["reason"],
                    )
                    rejected += 1
                    continue

                price = opp["price"]
                size = round(cost / price, 1) if price > 0 else 0.1

                self.display.tool_call("Trader", "execute_buy")
                result = self.toolkit.execute_buy(
                    token_id, opp["market"], outcome, size,
                    end_date=opp.get("end_date", ""),
                )

                if result["success"]:
                    self.bus.post("Trader", "all", "trade", result["message"])
                    # Link position to hypothesis
                    hyp_ref = _parse_trade_hypothesis_ref(
                        reasoning_text, idx + 1,
                    )
                    hyp = _find_hypothesis_by_title(self.db, hyp_ref) if hyp_ref else None
                    if hyp:
                        self.db.link_position_hypothesis(
                            token_id, hyp["id"], cycle,
                        )
                        self.db.update_hypothesis(
                            hyp["id"], status="traded", updated_cycle=cycle,
                        )
                    executed += 1
                else:
                    self.bus.post(
                        "Trader", "all", "alert",
                        f"Failed: {result['message']}",
                    )

            self._push_portfolio()
            self._sync_positions(cycle)

        self.display.agent_done(
            "Trader",
            f"{executed} filled" if executed else "idle",
        )

        # 6 ── CEO summarises (skip if nothing happened — save tokens) ──
        has_news = executed > 0 or rejected > 0 or bool(all_findings)
        if has_news:
            self.display.agent_thinking("CEO")
            ctx = _recent_context(self.bus, last_n=20)
            summary = self.ceo.act(
                f"Cycle {cycle} done. What happened:\n{ctx}\n\n"
                f"Summarise briefly. If our DIRECTION or hypotheses should evolve "
                f"based on what we learned, output updates using:\n"
                f"  DIRECTION: <updated thesis>\n"
                f"  DISMISS HYPOTHESIS: <title>\n"
                f"  STALE HYPOTHESIS: <title>\n"
                f"  NEW HYPOTHESIS: <title>"
            )
            self.bus.post("CEO", "all", "summary", summary.strip())
            self._apply_ceo_output(summary, cycle)
            self.display.agent_done("CEO")
        else:
            summary = "No trades or findings — CEO summary skipped to conserve tokens."
            self.display.agent_done("CEO", "skipped (no news)")

        self._push_portfolio()

        # Log cycle end
        agents = [self.ceo, *self.researchers, self.reasoning]
        self.db.end_cycle(
            cycle,
            bankroll_end=self.trader.bankroll,
            trades_executed=executed,
            trades_rejected=rejected,
            input_tokens=sum(a.total_input_tokens for a in agents),
            output_tokens=sum(a.total_output_tokens for a in agents),
            duration_s=round(time.time() - cycle_start, 1),
            strategy=ceo_text[:200],
            summary=summary[:500],
        )
        return {
            "type": "full", "executed": executed, "rejected": rejected,
            "is_hold": False,
        }

    # ── Session management ──────────────────────────────────────

    def clear_sessions(self, reset_db: bool = False) -> None:
        for agent in [self.ceo, *self.researchers, self.reasoning]:
            agent.clear_session()
        if reset_db:
            self.db.reset()

    # ── Adaptive wait ──────────────────────────────────────────

    def _nearest_resolution_secs(self) -> int | None:
        """Seconds until the earliest open position resolves."""
        now = datetime.now(timezone.utc)
        nearest = None
        for pos in self.trader.positions:
            if not pos.end_date:
                continue
            try:
                end = datetime.fromisoformat(pos.end_date.replace("Z", "+00:00"))
                delta = int((end - now).total_seconds())
                if delta > 0 and (nearest is None or delta < nearest):
                    nearest = delta
            except ValueError:
                continue
        return nearest

    def _compute_wait_time(
        self, cycle_result: dict, interval: int,
    ) -> tuple[int, str]:
        """Pick a natural wait time based on what just happened.

        Returns (seconds, reason) so the display can explain the wait.

        Core idea: monitor cycles are FREE so repeat often.
        Full cycles cost tokens so space them by context.
        """
        _power = hasattr(self.display, "power_mode") and self.display.power_mode
        if _power:
            return min(interval, 30), "power"

        cycle_type = cycle_result.get("type", "full")
        n_pos = len(self.trader.positions)
        executed = cycle_result.get("executed", 0)

        # Monitor cycles cost 0 tokens -- repeat frequently.
        # The moment a position resolves, the next monitor detects it
        # and the decision gate triggers a full cycle to redeploy.
        if cycle_type == "monitor":
            nearest = self._nearest_resolution_secs()
            if nearest and nearest > 180:
                # Check every 3 min or at resolution, whichever sooner
                wait = min(nearest, 180)
                return wait, f"monitoring, next resolution ~{nearest // 60}m"
            return 60, "monitoring positions"

        # HOLD cycles -- CEO said wait
        if cycle_type == "hold":
            if n_pos >= 4:
                nearest = self._nearest_resolution_secs()
                if nearest and nearest > 120:
                    wait = min(nearest, 3600)
                    return wait, f"holding, next resolution ~{wait // 60}m"
                return interval * 2, "holding, portfolio deployed"
            return min(interval * 2, 900), "holding"

        # Full cycles below -- cost tokens, space by outcome
        active_hyps = self.db._fetchone(
            "SELECT COUNT(*) as c FROM hypotheses WHERE status = 'active'",
        )
        n_hyps = active_hyps["c"] if active_hyps else 0

        # Just traded + positions near cap -- ease off
        if executed > 0 and n_pos >= 3:
            nearest = self._nearest_resolution_secs()
            if nearest and nearest > 300:
                wait = min(nearest // 2, 1800)
                return max(wait, interval), f"traded, next event ~{nearest // 60}m"
            return interval, "traded, monitoring"

        # ── Thin portfolio + active hypotheses → keep hunting ──────
        if n_pos < 3 and n_hyps > 0:
            return max(60, interval // 4), "hunting — active leads"

        # ── Trade drought → cycle fast to find opportunities ───────
        last_trade = self.db._fetchone(
            "SELECT MAX(id) as c FROM cycles WHERE trades_executed > 0",
        )
        ltc = (last_trade["c"] or 0) if last_trade else 0
        current_cycle = (self.db._fetchone(
            "SELECT MAX(id) as c FROM cycles",
        ) or {}).get("c", 0)
        drought = current_cycle - ltc if ltc else current_cycle
        if drought >= 3 and n_pos < 3:
            return max(45, interval // 5), "drought — searching"

        # ── Nothing found, no leads → longer pause to save tokens ──
        if executed == 0 and n_hyps == 0 and n_pos == 0:
            return min(interval * 2, 900), "idle — conserving tokens"

        return interval, "pace"

    # ── Main loop ───────────────────────────────────────────────

    def run(
        self, *, cycles: int | None = None, interval: int = 60,
    ) -> None:
        if isinstance(self.display, Dashboard) and cycles is not None:
            self.display.total_cycles = cycles

        self.display.start()
        self.display.banner()

        agents = [self.ceo, *self.researchers, self.reasoning]
        has_sessions = [a.name for a in agents if a.history]
        if has_sessions:
            self.bus.post(
                "System", "all", "info",
                f"Resuming sessions: {', '.join(has_sessions)}",
            )
        else:
            self.bus.post("System", "all", "info", "Fresh sessions started.")

        n = 0
        try:
            while cycles is None or n < cycles:
                n += 1
                result = self.run_cycle(n)
                if cycles is not None and n >= cycles:
                    break
                wait_time, wait_reason = self._compute_wait_time(
                    result or {}, interval,
                )
                self.display.wait(wait_time, reason=wait_reason)
        except KeyboardInterrupt:
            pass

        total_in = sum(a.total_input_tokens for a in agents)
        total_out = sum(a.total_output_tokens for a in agents)
        self.display.shutdown(
            f"[bold]Final:[/bold] Bankroll ${self.trader.bankroll:.2f} | "
            f"Positions: {len(self.trader.positions)} | "
            f"Value: ${self.trader.total_value:.2f} | "
            f"Tokens: {total_in + total_out:,} ({total_in:,} in / {total_out:,} out)"
        )
        self.db.close()
