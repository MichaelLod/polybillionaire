"""Inspiration engine — feeds research direction without a per-cycle CEO.

Zero-cost by default: generates direction from DB state.
Optional: periodic cheap LLM call for strategic review.
Per-agent topic assignment for power mode.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent_config import AgentConfig
    from .backend import AgentBackend
    from .bus import EventBus
    from .db import OrgDB


class InspirationEngine:
    """Generates research direction from DB state and human hints.

    Sources of inspiration (zero-cost, pure Python):
      1. Human hints (pending in DB)
      2. Active hypotheses that need evidence
      3. Stale hypotheses needing refresh
      4. Trade performance (what categories win?)
      5. Time-of-day awareness (sports in evening, politics daytime)

    In power mode, generates per-agent assignments based on agent roles.
    Optional: LLM strategic review on a slow cadence.
    """

    def __init__(
        self,
        db: OrgDB,
        bus: EventBus,
        backend: AgentBackend | None = None,
        interval_s: float = 300,
        llm_review_interval_s: float = 1800,
        agent_configs: list[AgentConfig] | None = None,
    ) -> None:
        self.db = db
        self.bus = bus
        self.backend = backend
        self.interval_s = interval_s
        self.llm_review_interval_s = llm_review_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_llm_review: float = 0
        self._assigned_leads: dict[str, int] = {}

        # Categorize agents by role prefix
        self._scanner_agents: list[str] = []
        self._diver_agents: list[str] = []
        self._contrarian_agents: list[str] = []
        if agent_configs:
            for c in agent_configs:
                if c.name.startswith("Scanner"):
                    self._scanner_agents.append(c.name)
                elif c.name.startswith("Diver"):
                    self._diver_agents.append(c.name)
                elif c.name.startswith("Contrarian"):
                    self._contrarian_agents.append(c.name)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="inspiration", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def generate(self) -> str:
        """Build direction from current DB state. Zero LLM cost."""
        parts: list[str] = []

        # 1. Current direction (if any)
        d = self.db.get_direction()
        if d:
            parts.append(f"CURRENT THESIS: {d['thesis']}")

        # 2. Human hints
        hints = self.db._fetchall(
            "SELECT text FROM hints WHERE status = 'pending' ORDER BY id DESC LIMIT 5",
        )
        if hints:
            hint_lines = "\n".join(f"  - {h['text']}" for h in hints)
            parts.append(
                f"HUMAN HINTS (investigate these first):\n{hint_lines}"
            )

        # 3. Active hypotheses — what we're tracking
        hyps = self.db.get_active_hypotheses()
        if hyps:
            needs_evidence = [
                h for h in hyps if h["status"] in ("active", "investigating")
            ]
            traded = [h for h in hyps if h["status"] == "traded"]

            if needs_evidence:
                lines = [f"  - {h['title']}" for h in needs_evidence[:5]]
                parts.append(
                    "HYPOTHESES NEEDING EVIDENCE (find supporting data):\n"
                    + "\n".join(lines)
                )
            if traded:
                lines = [
                    f"  - {h['title']} (edge: {h['edge']:+.0%})"
                    if h.get("edge") else f"  - {h['title']}"
                    for h in traded[:5]
                ]
                parts.append(
                    "ACTIVE TRADES (monitor for updates):\n"
                    + "\n".join(lines)
                )
        else:
            parts.append(
                "NO ACTIVE HYPOTHESES — find fresh opportunities!\n"
                "Look for: sports games today, breaking political news, "
                "crypto events, weather-related markets."
            )

        # 4. What's worked before (recent wins)
        recent_wins = self.db._fetchall(
            """SELECT title, edge FROM hypotheses
               WHERE status = 'won' ORDER BY updated_cycle DESC LIMIT 3""",
        )
        if recent_wins:
            lines = [f"  - {w['title']}" for w in recent_wins]
            parts.append(
                "RECENT WINS (look for similar patterns):\n"
                + "\n".join(lines)
            )

        # 5. Drought awareness
        positions = self.db.get_positions()
        if not positions and not hyps:
            parts.append(
                "URGENT: Zero positions AND zero hypotheses. "
                "The bankroll is 100% idle. Find ANYTHING tradeable today."
            )

        return "\n\n".join(parts)

    def generate_assignments(self) -> dict[str, str]:
        """Per-agent direction based on role. Zero LLM cost.

        Scanners: get general direction (their prompt specializes them).
        Divers: assigned to hottest unprocessed leads for deep research.
        Contrarians: assigned active hypotheses to attack.
        """
        general = self.generate()
        assignments: dict[str, str] = {}

        # Scanners — their system prompt already specializes by vertical.
        # Give them general direction + emphasis on speed.
        for name in self._scanner_agents:
            assignments[name] = general

        # Divers — assign each to a different hot lead for deep research.
        leads = self.db.get_active_leads()
        # Prioritize: high confidence first, then medium, skip already-assigned
        hot_leads = sorted(
            [l for l in leads if l.get("status") in ("new", "investigating")],
            key=lambda l: (
                0 if l.get("confidence") == "high" else
                1 if l.get("confidence") == "medium" else 2
            ),
        )

        # Also check abandoned trails for divers to pick up
        abandoned = self.db.get_abandoned_trails(exclude_agent="", limit=10)

        for i, name in enumerate(self._diver_agents):
            if i < len(hot_leads):
                lead = hot_leads[i]
                self._assigned_leads[name] = lead["id"]
                assignments[name] = (
                    f"== DEEP DIVE ASSIGNMENT ==\n"
                    f"Topic: {lead['title']}\n"
                    f"Source: {lead.get('source', 'unknown')}\n"
                    f"Signal: {lead.get('signal', 'unknown')}\n"
                    f"Current confidence: {lead.get('confidence', 'unknown')}\n\n"
                    f"Go DEEP on this specific topic. Cross-reference with 2+ "
                    f"independent sources. Get hard probability numbers.\n\n"
                    f"{general}"
                )
            elif i - len(hot_leads) < len(abandoned):
                trail = abandoned[i - len(hot_leads)]
                import json
                crumbs = json.loads(trail.get("breadcrumbs", "[]") or "[]")
                trail_summary = "\n".join(
                    f"  {j+1}. {c[:100]}" for j, c in enumerate(crumbs[-3:])
                )
                assignments[name] = (
                    f"== PICK UP ABANDONED TRAIL ==\n"
                    f"Topic: {trail['topic'][:100]}\n"
                    f"Previous depth: {trail['depth']}\n"
                    f"Breadcrumbs:\n{trail_summary}\n\n"
                    f"Continue this exploration. Go DEEPER.\n\n"
                    f"{general}"
                )
            else:
                # No leads to dive into — explore freely
                assignments[name] = (
                    f"== FREE EXPLORATION ==\n"
                    f"No specific assignment. Search for new opportunities "
                    f"that scanners might have missed.\n\n"
                    f"{general}"
                )

        # Contrarians — assign each to a different active hypothesis to attack.
        hyps = self.db.get_active_hypotheses()
        for i, name in enumerate(self._contrarian_agents):
            if i < len(hyps):
                h = hyps[i]
                edge_str = f"{h['edge']:+.0%}" if h.get("edge") else "unknown"
                assignments[name] = (
                    f"== CONTRARIAN ASSIGNMENT ==\n"
                    f"Attack this hypothesis: {h['title']}\n"
                    f"Current edge: {edge_str}\n"
                    f"Status: {h.get('status', 'active')}\n\n"
                    f"Find evidence that this hypothesis is WRONG. Search for:\n"
                    f"- Counter-evidence and contradicting data\n"
                    f"- Has the edge already closed?\n"
                    f"- Risks and biases the org is ignoring\n\n"
                    f"{general}"
                )
            else:
                # No hypotheses to attack — look for overconfidence signals
                assignments[name] = (
                    f"== CONTRARIAN: GENERAL SKEPTICISM ==\n"
                    f"No active hypotheses to attack. Search for:\n"
                    f"- Markets where the crowd is clearly biased\n"
                    f"- Events where consensus is wrong\n"
                    f"- Overreaction to recent news\n\n"
                    f"{general}"
                )

        return assignments

    def _maybe_llm_review(self) -> str | None:
        """Optional: use a cheap LLM to synthesize a strategic review."""
        if not self.backend:
            return None
        now = time.time()
        if now - self._last_llm_review < self.llm_review_interval_s:
            return None

        context = self.generate()
        prompt = (
            f"You are a trading strategist. Based on the current state below, "
            f"write a 2-3 sentence direction for what research agents should "
            f"focus on RIGHT NOW. Be specific — name categories, events, "
            f"or markets.\n\n{context}"
        )
        resp = self.backend.send(prompt)
        self._last_llm_review = now

        if resp.text and not resp.text.startswith("[error"):
            self.db.set_direction(resp.text)
            return resp.text
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                # Try LLM review first (if backend configured + enough time elapsed)
                llm_direction = self._maybe_llm_review()

                # Generate per-agent assignments if we have agent roles
                has_roles = bool(
                    self._scanner_agents or self._diver_agents
                    or self._contrarian_agents
                )
                if has_roles:
                    assignments = self.generate_assignments()
                    general = llm_direction or self.generate()
                    self.bus.emit(
                        "direction_update", "Inspiration",
                        {"direction": general, "assignments": assignments},
                        content=general[:200],
                    )
                else:
                    # Classic mode — single direction for all
                    direction = llm_direction or self.generate()
                    self.bus.emit(
                        "direction_update", "Inspiration",
                        {"direction": direction},
                        content=direction[:200],
                    )

                # If no stored direction yet, save the auto-generated one
                if not self.db.get_direction():
                    general = llm_direction or self.generate()
                    self.db.set_direction(general[:500])

            except Exception as e:
                self.bus.post(
                    "Inspiration", "all", "alert",
                    f"Inspiration error: {str(e)[:100]}",
                )

            self._stop.wait(self.interval_s)
