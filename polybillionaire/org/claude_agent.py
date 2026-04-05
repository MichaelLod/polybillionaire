"""ClaudeCodeAgent — wraps the ``claude`` CLI for agent reasoning.

Each agent is a real Claude Code session.  No API key needed — uses
the same auth you already have.  Sessions persist via ``--resume``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

SESSION_DIR = Path("org_sessions")


class ClaudeCodeAgent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "sonnet",
        tools: list[str] | None = None,
        display: Any = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools  # e.g. ["WebSearch", "WebFetch"] or None for default
        self.display = display
        self._session_id: str | None = None
        self._sessions_file = SESSION_DIR / "sessions.json"
        self._load_session_id()
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ── Core ────────────────────────────────────────────────────

    def act(self, prompt: str) -> str:
        """Send a prompt to a Claude Code session and return the text response."""
        cmd = [
            "claude",
            "-p",
            "--output-format", "json",
            "--model", self.model,
        ]

        if self.tools is not None:
            cmd.extend(["--tools", ",".join(self.tools) if self.tools else ""])

        if self._session_id:
            cmd.extend(["--resume", self._session_id])
        else:
            cmd.extend(["--system-prompt", self.system_prompt])

        # Researcher with web tools gets more time
        timeout = 600 if self.tools and "WebSearch" in self.tools else 300

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()[:200]
            return f"[error: {err}]"

        # Parse the CLI JSON envelope (skip any warning lines before JSON)
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        if json_start < 0:
            return stdout
        try:
            envelope = json.loads(stdout[json_start:])
        except json.JSONDecodeError:
            return stdout

        # Persist session id for --resume next time
        if sid := envelope.get("session_id"):
            self._session_id = sid
            self._save_session_id()

        # Track tokens
        usage = envelope.get("usage", {})
        input_tok = int(usage.get("input_tokens", 0))
        output_tok = int(usage.get("output_tokens", 0))
        self.total_input_tokens += input_tok
        self.total_output_tokens += output_tok
        if self.display:
            self.display.update_tokens(self.name, input_tok, output_tok)

        return str(envelope.get("result", ""))

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
