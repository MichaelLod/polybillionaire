"""ClaudeCodeAgent — wraps the ``claude`` CLI for agent reasoning.

Each agent is a real Claude Code session.  No API key needed — uses
the same auth you already have.  Sessions persist via ``--resume``.

Internally delegates to :class:`ClaudeCliBackend` from ``backend.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .backend import AgentBackend, ClaudeCliBackend

SESSION_DIR = Path("org_sessions")


class ClaudeCodeAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "sonnet",
        tools: list[str] | None = None,
        display: Any = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools
        self.display = display
        self.backend = backend or ClaudeCliBackend(model=model, tools=tools)
        self._session_id: str | None = None
        self._sessions_file = SESSION_DIR / "sessions.json"
        self._load_session_id()
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ── Core ────────────────────────────────────────────────────

    def act(self, prompt: str) -> str:
        """Send a prompt to the backend and return the text response."""
        resp = self.backend.send(
            prompt,
            system_prompt=self.system_prompt,
            session_id=self._session_id,
        )

        # Persist session id for --resume next time
        if resp.session_id:
            self._session_id = resp.session_id
            self._save_session_id()

        # Track tokens
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost += resp.cost_usd
        if self.display:
            self.display.update_tokens(
                self.name, resp.input_tokens, resp.output_tokens,
            )

        return resp.text

    # ── Session persistence ─────────────────────────────────────

    def _load_session_id(self) -> None:
        if not self._sessions_file.exists():
            return
        try:
            data = json.loads(self._sessions_file.read_text())
            self._session_id = data.get(self.name)
        except (json.JSONDecodeError, TypeError):
            pass

    def _save_session_id(self) -> None:
        self._sessions_file.parent.mkdir(exist_ok=True)
        data: dict[str, str] = {}
        if self._sessions_file.exists():
            try:
                data = json.loads(self._sessions_file.read_text())
            except (json.JSONDecodeError, TypeError):
                pass
        data[self.name] = self._session_id  # type: ignore[assignment]
        self._sessions_file.write_text(json.dumps(data, indent=2))

    def clear_session(self) -> None:
        self._session_id = None
        if not self._sessions_file.exists():
            return
        try:
            data = json.loads(self._sessions_file.read_text())
            data.pop(self.name, None)
            self._sessions_file.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, TypeError):
            pass

    @property
    def history(self) -> list:
        """Non-empty when a session exists (for runner status display)."""
        return ["session"] if self._session_id else []
