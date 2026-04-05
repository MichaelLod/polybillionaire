"""Agent configuration — YAML-driven, persistent across restarts.

Each agent in the swarm is defined by an AgentConfig. The YAML file
(``agents.yaml``) is the source of truth. The TUI reads it; runtime
spawn/kill writes back to it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Lazy-import yaml only when needed (avoid hard dep for CLI-only users)
_YAML_AVAILABLE: bool | None = None


def _yaml():
    global _YAML_AVAILABLE
    try:
        import yaml
        _YAML_AVAILABLE = True
        return yaml
    except ImportError:
        _YAML_AVAILABLE = False
        return None


# ── Config Dataclass ───────────────────────────────────────────

@dataclass
class AgentConfig:
    name: str
    role: Literal["research", "reasoning", "monitor", "inspiration"]
    backend_type: Literal["claude-cli", "anthropic-api", "openai-compat", "lm-studio"] = "claude-cli"
    model: str = "sonnet"
    system_prompt_key: str = ""   # key into agents.py PROMPTS registry
    tools: list[str] | None = None  # None = backend default, [] = no tools

    # Throttling
    min_interval_s: float = 60.0
    max_cost_per_hour: float = 1.0
    cooldown_after_error_s: float = 30.0

    # Backend-specific
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 16384
    timeout: float = 300.0

    # Runtime (not serialized)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize for YAML (omit defaults and runtime fields)."""
        d: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "backend_type": self.backend_type,
            "model": self.model,
        }
        if self.system_prompt_key:
            d["system_prompt_key"] = self.system_prompt_key
        if self.tools:
            d["tools"] = self.tools
        if self.min_interval_s != 60.0:
            d["min_interval_s"] = self.min_interval_s
        if self.max_cost_per_hour != 1.0:
            d["max_cost_per_hour"] = self.max_cost_per_hour
        if self.api_key:
            d["api_key"] = self.api_key
        if self.base_url:
            d["base_url"] = self.base_url
        if self.max_tokens != 4096:
            d["max_tokens"] = self.max_tokens
        if self.timeout != 300.0:
            d["timeout"] = self.timeout
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AgentConfig:
        """Deserialize from YAML dict."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ── Default Swarm Presets ──────────────────────────────────────

DEFAULT_CONFIGS: list[AgentConfig] = [
    AgentConfig(
        name="Research-1",
        role="research",
        backend_type="lm-studio",
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234",
        system_prompt_key="RESEARCH_NUMBERS_PROMPT",
        tools=[],
        min_interval_s=30,
        max_cost_per_hour=0.0,
    ),
    AgentConfig(
        name="Research-2",
        role="research",
        backend_type="lm-studio",
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234",
        system_prompt_key="RESEARCH_EDGE_PROMPT",
        tools=[],
        min_interval_s=30,
        max_cost_per_hour=0.0,
    ),
    AgentConfig(
        name="Reasoning",
        role="reasoning",
        backend_type="lm-studio",
        model="google/gemma-4-26b-a4b",
        base_url="http://localhost:1234",
        system_prompt_key="REASONING_PROMPT",
        tools=[],
        min_interval_s=30,
        max_cost_per_hour=0.0,
    ),
]

_LM_STUDIO_DEFAULTS = dict(
    backend_type="lm-studio",
    model="google/gemma-4-26b-a4b",
    base_url="http://localhost:1234",
    max_cost_per_hour=0.0,
)


def _power_configs() -> list[AgentConfig]:
    """4 scanners + 7 divers + 2 contrarians + 2 reasoning = 15 LLM agents.
    Monitor + Inspiration auto-spawned = 17 total."""
    configs: list[AgentConfig] = []

    # 4 Scanners — broad sweeps, one per vertical
    for name, prompt_key in [
        ("Scanner-Sports", "SCANNER_SPORTS_PROMPT"),
        ("Scanner-Politics", "SCANNER_POLITICS_PROMPT"),
        ("Scanner-Crypto", "SCANNER_CRYPTO_PROMPT"),
        ("Scanner-News", "SCANNER_NEWS_PROMPT"),
    ]:
        configs.append(AgentConfig(
            name=name, role="research", system_prompt_key=prompt_key,
            min_interval_s=15, **_LM_STUDIO_DEFAULTS,
        ))

    # 7 Deep Divers — assigned to hottest leads by InspirationEngine
    for i in range(1, 8):
        configs.append(AgentConfig(
            name=f"Diver-{i}", role="research",
            system_prompt_key="DEEP_DIVER_PROMPT",
            min_interval_s=10, **_LM_STUDIO_DEFAULTS,
        ))

    # 2 Contrarians — attack active hypotheses
    for i in range(1, 3):
        configs.append(AgentConfig(
            name=f"Contrarian-{i}", role="research",
            system_prompt_key="CONTRARIAN_PROMPT",
            min_interval_s=20, **_LM_STUDIO_DEFAULTS,
        ))

    # 2 Reasoning — evaluate findings, propose trades (shared findings queue)
    # tools=[] disables web search so Reasoning outputs TRADE blocks directly
    for i in range(1, 3):
        configs.append(AgentConfig(
            name=f"Reasoning-{i}", role="reasoning",
            system_prompt_key="REASONING_PROMPT",
            tools=[], max_tokens=32768,
            min_interval_s=10, **_LM_STUDIO_DEFAULTS,
        ))

    return configs


POWER_CONFIGS = _power_configs()

# Quick-spawn templates for the TUI [+] button
PRESETS: dict[str, AgentConfig] = {
    "research-haiku": AgentConfig(
        name="Research-N",
        role="research",
        backend_type="claude-cli",
        model="haiku",
        system_prompt_key="RESEARCH_NUMBERS_PROMPT",
        tools=["WebSearch", "WebFetch"],
        min_interval_s=90,
        max_cost_per_hour=0.50,
    ),
    "research-sonnet": AgentConfig(
        name="Research-N",
        role="research",
        backend_type="claude-cli",
        model="sonnet",
        system_prompt_key="RESEARCH_NUMBERS_PROMPT",
        tools=["WebSearch", "WebFetch"],
        min_interval_s=120,
        max_cost_per_hour=1.50,
    ),
    "research-opus": AgentConfig(
        name="Research-N",
        role="research",
        backend_type="claude-cli",
        model="opus",
        system_prompt_key="RESEARCH_EDGE_PROMPT",
        tools=["WebSearch", "WebFetch"],
        min_interval_s=180,
        max_cost_per_hour=3.00,
    ),
    "research-local": AgentConfig(
        name="Research-N",
        role="research",
        backend_type="openai-compat",
        model="local-model",
        system_prompt_key="RESEARCH_NUMBERS_PROMPT",
        base_url="http://localhost:1234/v1",
        min_interval_s=30,
        max_cost_per_hour=0.0,
    ),
    "research-gemma": AgentConfig(
        name="Research-N",
        role="research",
        backend_type="lm-studio",
        model="google/gemma-4-26b-a4b",
        system_prompt_key="RESEARCH_NUMBERS_PROMPT",
        base_url="http://localhost:1234",
        min_interval_s=30,
        max_cost_per_hour=0.0,
    ),
}


# ── YAML I/O ───────────────────────────────────────────────────

DEFAULT_YAML_PATH = Path("agents.yaml")


def load_configs(path: Path = DEFAULT_YAML_PATH) -> list[AgentConfig]:
    """Load agent configs from YAML. Falls back to defaults if missing."""
    if not path.exists():
        return [copy.deepcopy(c) for c in DEFAULT_CONFIGS]

    yaml = _yaml()
    if yaml is None:
        return [copy.deepcopy(c) for c in DEFAULT_CONFIGS]

    raw = yaml.safe_load(path.read_text()) or {}
    agents = raw.get("agents", [])
    if not agents:
        return [copy.deepcopy(c) for c in DEFAULT_CONFIGS]

    return [AgentConfig.from_dict(a) for a in agents]


def save_configs(configs: list[AgentConfig], path: Path = DEFAULT_YAML_PATH) -> None:
    """Write agent configs to YAML."""
    yaml = _yaml()
    if yaml is None:
        import json
        data = {"agents": [c.to_dict() for c in configs]}
        path.write_text(json.dumps(data, indent=2))
        return

    data = {"agents": [c.to_dict() for c in configs]}
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def next_agent_name(configs: list[AgentConfig], role: str = "research") -> str:
    """Generate the next available name like Research-3."""
    prefix = role.capitalize()
    existing = {c.name for c in configs if c.role == role}
    n = 1
    while f"{prefix}-{n}" in existing:
        n += 1
    return f"{prefix}-{n}"
