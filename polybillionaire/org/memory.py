"""Persistent memory store for agents — survives across sessions."""

from __future__ import annotations

import json
from pathlib import Path


class AgentMemory:
    def __init__(self, path: str = "agent_memories.json") -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def save(self, agent: str, key: str, value: str) -> None:
        self._data.setdefault(agent, {})[key] = value
        self._persist()

    def get(self, agent: str, key: str) -> str | None:
        return self._data.get(agent, {}).get(key)

    def get_all(self, agent: str) -> dict[str, str]:
        return dict(self._data.get(agent, {}))

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, TypeError):
                self._data = {}

    def _persist(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2))
