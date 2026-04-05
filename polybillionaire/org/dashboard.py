"""Animated full-screen TUI dashboard for the trading organisation.

Layout:
  +------------------------------------------------------+
  | Header: cycle, bankroll, positions, P&L, tokens      |
  +------------------------------------------------------+
  | Org chart: CEO -> Research Pool -> Reasoning -> Trader|
  +------------------------------------------------------+
  | Status line: active agents                            |
  +----------------------------+-------------------------+
  | Main content area          | Sidebar                 |
  |  [f] Activity feed         |  Org Memory             |
  |  [p] Positions + reasoning |  (direction, hypotheses)|
  |  [h] Hypotheses + leads    |                         |
  |  [1-6] Agent windows       |  Your Intuition         |
  |                            |  (hints input)          |
  +----------------------------+-------------------------+
  | Footer: keybinds                                     |
  +------------------------------------------------------+

Navigation:
  f/p/h     — switch main content view
  1-6       — toggle agent detail windows
  j/k       — navigate lists (positions, hypotheses)
  Esc       — back to feed / close agent windows
  Up/Down   — scroll in agent windows
"""

from __future__ import annotations

import os
import signal
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass

from rich.box import DOUBLE, ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from .bus import Message
from .db import OrgDB

# ── Colours ──────────────────────────────────────────────────────────────────

AGENT_COLORS = {
    "CEO": "magenta",
    "Research": "cyan",
    "Reasoning": "yellow",
    "Trader": "green",
    "System": "blue",
}

KIND_LABELS: dict[str, tuple[str, str]] = {
    "directive": ("DIRECTIVE", "magenta"),
    "info": ("INFO", "dim"),
    "proposal": ("PROPOSAL", "cyan"),
    "approval": ("APPROVED", "green"),
    "rejection": ("REJECTED", "red"),
    "trade": ("TRADE", "bold green"),
    "alert": ("ALERT", "bold red"),
    "summary": ("SUMMARY", "bold white"),
}

STATUS_DOT = {
    "idle": "[dim]\u25cb[/dim]",
    "thinking": "[bold green]\u25c9[/bold green]",
    "working": "[bold yellow]\u25cf[/bold yellow]",
    "done": "[green]\u25cf[/green]",
}

CONF_COLOR = {"high": "green", "medium": "yellow", "low": "red"}

HYP_INDICATOR = {
    "traded": "[green]$[/green]",
    "edge_found": "[yellow]![/yellow]",
    "investigating": "[cyan]?[/cyan]",
    "active": "[white].[/white]",
    "stale": "[dim]-[/dim]",
    "dismissed": "[dim]x[/dim]",
    "resolved": "[green]\u2713[/green]",
}


def _color(name: str) -> str:
    return AGENT_COLORS.get(name, AGENT_COLORS.get(name.split("-")[0], "white"))


# ── Ship classes ─────────────────────────────────────────────────────────────
# Each ship class maps to an agent type. Sprites use block elements for a
# half-block pixel art look. Ships face right (→) for scouts/interceptors,
# upward (↑) for probes, or are symmetric for stations.

@dataclass
class ShipClass:
    """Definition of a fleet ship class."""
    key: str           # deploy keybind (lowercase)
    label: str         # display name
    agent_prefix: str  # agent name prefix (e.g. "Scanner", "Diver")
    max_count: int     # max deployable
    color: str         # base color
    sprite: list[str]  # ASCII art lines (centered)
    exhaust: str       # exhaust character pattern


# Sprites designed with block elements — each is ~5-7 lines tall, <16 wide
# Using: █ ▓ ▒ ░ ▀ ▄ ▌ ▐ ► ◄ ▸ ◂ ◆ ◈

SHIP_SPRITES: dict[str, list[str]] = {
    "scout": [          # Scanner — fast corvette, faces right
        "   ░▒▓█►  ",
        "  ▒▓████▓▸",
        " ░▓██████►",
        "  ▒▓████▓▸",
        "   ░▒▓█►  ",
    ],
    "probe": [          # Diver — torpedo, faces down for diving
        "    ▄█▄    ",
        "   ▐███▌   ",
        "   ▐█▓█▌   ",
        "    ▓▒▓    ",
        "    ░▒░    ",
        "     ▀     ",
    ],
    "interceptor": [    # Contrarian — angular attack ship
        "  █▓░ ░▓█  ",
        "   ▓███▓   ",
        "  ▒█████▒  ",
        "   ▓███▓   ",
        "  █▓░ ░▓█  ",
    ],
    "battlecruiser": [  # Reasoning — heavy capital ship
        "  ░▒▓████▓▒░  ",
        "   ▓██████▓   ",
        "  ▒████████▒  ",
        "   ▓██████▓   ",
        "  ░▒▓████▓▒░  ",
    ],
    "mothership": [     # System — central station, always present
        "    ▄███▄    ",
        "  ░▓█████▓░  ",
        "  ▒███████▒  ",
        "  ░▓█████▓░  ",
        "    ▀███▀    ",
    ],
    "beacon": [         # Inspiration — pulsing signal tower
        "     ░     ",
        "    ▒█▒    ",
        "   ░▓█▓░   ",
        "    ▒█▒    ",
        "     ░     ",
    ],
}

SHIP_CLASSES: dict[str, ShipClass] = {
    "scout": ShipClass(
        key="s", label="Scout", agent_prefix="Scanner",
        max_count=4, color="cyan", exhaust="·∘○",
        sprite=SHIP_SPRITES["scout"],
    ),
    "probe": ShipClass(
        key="d", label="Probe", agent_prefix="Diver",
        max_count=7, color="green", exhaust="░▒▓",
        sprite=SHIP_SPRITES["probe"],
    ),
    "interceptor": ShipClass(
        key="c", label="Interceptor", agent_prefix="Contrarian",
        max_count=2, color="red", exhaust="·•●",
        sprite=SHIP_SPRITES["interceptor"],
    ),
    "battlecruiser": ShipClass(
        key="r", label="Battlecruiser", agent_prefix="Reasoning",
        max_count=1, color="yellow", exhaust="═══",
        sprite=SHIP_SPRITES["battlecruiser"],
    ),
}

# Map agent name prefix → ship class key
AGENT_SHIP_MAP: dict[str, str] = {
    "Scanner": "scout",
    "Diver": "probe",
    "Contrarian": "interceptor",
    "Reasoning": "battlecruiser",
    "System": "mothership",
    "Monitor": "mothership",
    "Inspiration": "beacon",
}

# Deploy order for scanners (each deploy picks the next type)
SCANNER_ORDER = [
    "Scanner-Sports", "Scanner-Politics", "Scanner-Crypto", "Scanner-News",
]

# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class AgentStatus:
    status: str = "idle"
    activity: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


# ── Self-animating wrapper ───────────────────────────────────────────────────


class _Animated:
    def __init__(self, dash: "Dashboard") -> None:
        self.dash = dash

    def __rich__(self) -> Layout:
        self.dash._tick()
        try:
            return self.dash._build()
        except Exception as e:
            return Layout(Panel(f"[red]Render error: {e}[/red]"))


# ── Dashboard ────────────────────────────────────────────────────────────────


class Dashboard:
    """Full-screen animated TUI with org chart, positions, and agent windows."""

    def __init__(
        self,
        model: str = "",
        console: Console | None = None,
        db: OrgDB | None = None,
        num_researchers: int = 3,
        trader: object | None = None,
    ) -> None:
        self.console = console or Console()
        self.model = model
        self.db = db
        self.trader = trader  # PaperTrader — for selling from the TUI
        self.cycle = 0
        self.total_cycles: int | None = None

        self._research_names = [f"Research-{i + 1}" for i in range(num_researchers)]
        self._all_agent_names = [
            *self._research_names, "Reasoning", "Trader",
        ]
        self.agents: dict[str, AgentStatus] = {}
        self.messages: list[Message] = []
        self.portfolio: dict[str, float | int] = {
            "bankroll": 0.0, "deployed": 0.0,
            "value": 0.0, "pnl": 0.0, "positions": 0,
        }

        # Animation state
        self._active_flows: dict[str, int] = {}
        self._frame = 0
        self._footer_extra = ""
        self._live: Live | None = None

        # View mode: "balls" | "feed" | "positions" | "hypotheses"
        self._view_mode = "balls"
        self._selected_agent: str | None = None
        self._selected_index = 0
        self._burst_frames: dict[str, int] = {}  # finding burst animation
        self._agent_grid_pos: dict[str, tuple[int, int]] = {}  # (gx, gy) grid coords

        # 4th dimension trail state — cached from DB
        self._trail_cache: dict[str, dict] = {}  # agent -> {depth, status, topic}
        self._trail_cache_frame = 0  # last frame we refreshed
        self._surface_burst: dict[str, int] = {}  # agent -> frames remaining
        self._particle_trails: list[tuple[int, int, int, str]] = []  # (x, y, ttl, color)

        # Fleet command — swarm reference set after construction
        self.swarm: object | None = None  # set to Swarm instance by cli.py
        self._projectiles: list[dict] = []  # {x, y, tx, ty, ttl, char, color}

        # Parallax starfield — 3 layers seeded once
        import random as _rng
        self._stars: list[tuple[int, int, int]] = []  # (x, y, layer)
        self._stars_seeded = False

        # Hint input mode — when True, all keystrokes go to hint buffer
        self._input_mode = False

        # Sell confirmation
        self._sell_confirm: str | None = None  # token_id pending confirmation
        self._sell_result: str | None = None  # result message to show briefly
        self._sell_result_ttl: int = 0

        # Per-agent windows (number keys 1-6)
        self._open_windows: list[str] = []
        self._last_output: dict[str, str] = {}
        self._window_scroll: dict[str, int] = {}

        # Key mapping: number keys -> agent names (rebuilt dynamically)
        self._key_to_agent: dict[str, str] = {}
        self._agent_to_key: dict[str, str] = {}

        # Power mode — burns tokens faster for aggressive trading
        self.power_mode = False

        # Hints
        self.hints: list[dict[str, str]] = []
        self._input_buffer = ""
        self._input_thread: threading.Thread | None = None
        self._old_term_settings = None

    # ── Lifecycle ───────────────────────────────────────────

    def start(self) -> None:
        self._live = Live(
            _Animated(self),
            console=self.console,
            refresh_per_second=4,
            screen=True,
        )
        self._live.start()
        self._start_input_thread()

    def stop(self) -> None:
        self._stop_input_thread()
        if self._live:
            self._live.stop()
            self._live = None

    # ── Keyboard input ────────────────────────────────────────

    def _start_input_thread(self) -> None:
        try:
            fd = sys.stdin.fileno()
            self._old_term_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except (termios.error, OSError):
            return
        self._input_thread = threading.Thread(
            target=self._read_input, daemon=True,
        )
        self._input_thread.start()

    def _stop_input_thread(self) -> None:
        if self._old_term_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN,
                    self._old_term_settings,
                )
            except (termios.error, OSError):
                pass
            self._old_term_settings = None

    def _read_input(self) -> None:
        fd = sys.stdin.fileno()
        try:
            while self._live:
                data = os.read(fd, 1)
                if not data:
                    break
                ch = data.decode("utf-8", errors="replace")

                if ch == "\x03":  # Ctrl+C
                    if self._input_mode:
                        self._input_mode = False
                        self._input_buffer = ""
                        continue
                    os.kill(os.getpid(), signal.SIGINT)
                    break

                # ── Hint input mode: all keys go to buffer ──
                if self._input_mode:
                    if ch == "\x1b":  # Escape — cancel input
                        self._input_mode = False
                        self._input_buffer = ""
                    elif ch in ("\n", "\r"):  # Enter — submit
                        text = self._input_buffer.strip()
                        if text:
                            self.hints.append({"text": text, "status": "pending"})
                        self._input_buffer = ""
                        self._input_mode = False
                    elif ch in ("\x7f", "\x08"):  # Backspace
                        self._input_buffer = self._input_buffer[:-1]
                    elif ch == "\x15":  # Ctrl+U — clear line
                        self._input_buffer = ""
                    elif ch >= " ":
                        self._input_buffer += ch
                    continue

                # ── Navigation mode ──

                # Sell confirmation pending
                if self._sell_confirm:
                    if ch in ("y", "Y"):
                        self._execute_sell(self._sell_confirm)
                        self._sell_confirm = None
                    else:
                        self._sell_confirm = None
                        self._sell_result = "Cancelled"
                        self._sell_result_ttl = 12
                    continue

                # "x" sells the selected position
                if ch == "x" and self._view_mode == "positions":
                    self._initiate_sell()
                    continue

                # "g" toggles power mode (go!)
                if ch == "g":
                    self.power_mode = not self.power_mode
                    continue

                # Fleet deploy (lowercase) / recall (uppercase)
                deploy_keys = {sc.key: k for k, sc in SHIP_CLASSES.items()}
                recall_keys = {sc.key.upper(): k for k, sc in SHIP_CLASSES.items()}
                if ch in deploy_keys:
                    self._deploy_ship(deploy_keys[ch])
                    continue
                if ch in recall_keys:
                    self._recall_ship(recall_keys[ch])
                    continue

                # "i" enters hint input mode
                if ch == "i":
                    self._input_mode = True
                    self._input_buffer = ""
                    continue

                # View mode switching (toggle: press again to return to balls)
                if ch in ("f", "p", "h"):
                    target = {"f": "feed", "p": "positions", "h": "hypotheses"}[ch]
                    self._selected_agent = None
                    self._open_windows.clear()
                    self._window_scroll.clear()
                    if self._view_mode == target:
                        self._view_mode = "balls"
                    else:
                        self._view_mode = target
                    self._selected_index = 0
                    continue

                # j/k navigation within list views
                if ch == "j" and self._view_mode in ("positions", "hypotheses"):
                    self._selected_index += 1
                    continue
                if ch == "k" and self._view_mode in ("positions", "hypotheses"):
                    self._selected_index = max(0, self._selected_index - 1)
                    continue

                # Number keys select/expand agent balls
                if ch in self._key_to_agent:
                    name = self._key_to_agent[ch]
                    if self._selected_agent == name:
                        self._selected_agent = None  # deselect
                    else:
                        self._selected_agent = name
                    self._view_mode = "balls"
                    continue

                # Escape sequence handling
                if ch == "\x1b":
                    import select as sel
                    if sel.select([fd], [], [], 0.05)[0]:
                        ch2 = os.read(fd, 1).decode("utf-8", errors="replace")
                        if ch2 == "[" and sel.select([fd], [], [], 0.05)[0]:
                            ch3 = os.read(fd, 1).decode("utf-8", errors="replace")
                            # Arrow keys in agent windows
                            if self._open_windows:
                                last = self._open_windows[-1]
                                cur = self._window_scroll.get(last, 0)
                                if ch3 == "A":  # Up
                                    self._window_scroll[last] = max(0, cur - 3)
                                elif ch3 == "B":  # Down
                                    self._window_scroll[last] = cur + 3
                            # Arrow keys in list views
                            elif self._view_mode in ("positions", "hypotheses"):
                                if ch3 == "A":  # Up
                                    self._selected_index = max(0, self._selected_index - 1)
                                elif ch3 == "B":  # Down
                                    self._selected_index += 1
                    else:
                        # Standalone Escape — back to ball grid
                        self._open_windows.clear()
                        self._window_scroll.clear()
                        self._selected_agent = None
                        self._view_mode = "balls"
                        self._selected_index = 0
                    continue
        except (EOFError, OSError):
            pass

    # ── Display interface (same API as Display) ────────────────

    def banner(self) -> None:
        pass

    def cycle_start(self, n: int) -> None:
        self.cycle = n
        self._footer_extra = ""
        for s in self.agents.values():
            s.status = "idle"
            s.activity = ""

    def render(self, msg: Message) -> None:
        self.messages.append(msg)
        self._ensure_agent(msg.sender)
        self.agents[msg.sender].status = "working"
        self.agents[msg.sender].activity = msg.kind
        self._last_output[msg.sender] = msg.content.strip()
        if msg.sender in self._open_windows:
            self._window_scroll[msg.sender] = 999999
        if msg.to in self.agents and msg.sender in self.agents:
            self._active_flows[f"{msg.sender}->{msg.to}"] = 16
        elif msg.to == "all" and msg.sender in self.agents:
            for name in self.agents:
                if name != msg.sender:
                    self._active_flows[f"{msg.sender}->{name}"] = 10
        # Trigger burst glow on findings
        if msg.kind == "finding":
            self._burst_frames[msg.sender] = 8
            # If this agent was diving, trigger a dramatic surface burst
            if msg.sender in self._trail_cache:
                self._surface_burst[msg.sender] = 16
            # Fire data projectile toward Reasoning (battlecruiser)
            sender_pos = self._agent_grid_pos.get(msg.sender)
            reasoning_pos = None
            for aname in self.agents:
                if aname.startswith("Reasoning"):
                    reasoning_pos = self._agent_grid_pos.get(aname)
                    break
            if sender_pos and reasoning_pos:
                sx = sender_pos[0] * 18 + 9
                sy = sender_pos[1] * 9 + 4
                tx = reasoning_pos[0] * 18 + 9
                ty = reasoning_pos[1] * 9 + 4
                self._projectiles.append({
                    "x": float(sx), "y": float(sy),
                    "tx": float(tx), "ty": float(ty),
                    "ttl": 20, "char": "◆", "color": "bold cyan",
                })

    def _ensure_agent(self, name: str) -> None:
        if name not in self.agents:
            self.agents[name] = AgentStatus()
            # Auto-assign next available number key
            for k in "123456789":
                if k not in self._key_to_agent:
                    self._key_to_agent[k] = name
                    self._agent_to_key[name] = k
                    break
            # Auto-place on grid canvas
            if name not in self._agent_grid_pos:
                self._auto_place_agent(name)

    def _auto_place_agent(self, name: str) -> None:
        """Place a new agent at the next free grid position."""
        occupied = set(self._agent_grid_pos.values())
        # Spiral outward from center to find free spot
        # Grid coords: (0,0) is top-left of the grid
        max_cols, max_rows = 6, 4
        for gy in range(max_rows):
            for gx in range(max_cols):
                if (gx, gy) not in occupied:
                    self._agent_grid_pos[name] = (gx, gy)
                    return
        # Fallback: stack at (0, 0)
        self._agent_grid_pos[name] = (0, 0)

    # ── Fleet command ───────────────────────────────────────────

    def _get_ship_class(self, name: str) -> str | None:
        """Map an agent name to its ship class key."""
        prefix = name.split("-")[0]
        return AGENT_SHIP_MAP.get(prefix)

    def _fleet_counts(self) -> dict[str, int]:
        """Count deployed ships per class."""
        counts: dict[str, int] = {}
        for name in self.agents:
            sc = self._get_ship_class(name)
            if sc and sc in SHIP_CLASSES:
                counts[sc] = counts.get(sc, 0) + 1
        return counts

    def _deploy_ship(self, class_key: str) -> None:
        """Deploy one ship of the given class via swarm."""
        if not self.swarm:
            self._footer_extra = "No swarm connected"
            return
        sc = SHIP_CLASSES.get(class_key)
        if not sc:
            return
        # Check max count
        current = self._fleet_counts().get(class_key, 0)
        if current >= sc.max_count:
            self._footer_extra = f"{sc.label} fleet full ({sc.max_count}/{sc.max_count})"
            return
        # Find the right config to spawn
        from .agent_config import POWER_CONFIGS
        import copy
        # For scouts, deploy in order: Sports→Politics→Crypto→News
        if class_key == "scout":
            deployed_scanners = [
                n for n in self.agents if n.startswith("Scanner-")
            ]
            for scanner_name in SCANNER_ORDER:
                if scanner_name not in deployed_scanners:
                    cfg = next(
                        (c for c in POWER_CONFIGS if c.name == scanner_name),
                        None,
                    )
                    if cfg:
                        self.swarm.spawn_agent(copy.deepcopy(cfg))
                        self._footer_extra = f"Deployed {scanner_name}"
                        return
            self._footer_extra = "All scanners deployed"
            return
        # For others, find a matching config template
        for cfg in POWER_CONFIGS:
            if cfg.name.startswith(sc.agent_prefix):
                # Check if this specific one is already deployed
                if cfg.name not in self.agents:
                    self.swarm.spawn_agent(copy.deepcopy(cfg))
                    self._footer_extra = f"Deployed {cfg.name}"
                    return
        self._footer_extra = f"No more {sc.label}s available"

    def _recall_ship(self, class_key: str) -> None:
        """Recall (kill) the most recently deployed ship of this class."""
        if not self.swarm:
            self._footer_extra = "No swarm connected"
            return
        sc = SHIP_CLASSES.get(class_key)
        if not sc:
            return
        # Find deployed agents of this class (reverse order = recall newest)
        deployed = [
            n for n in reversed(list(self.agents.keys()))
            if self._get_ship_class(n) == class_key
        ]
        if not deployed:
            self._footer_extra = f"No {sc.label}s deployed"
            return
        name = deployed[0]
        self.swarm.kill_agent(name)
        # Remove from dashboard state
        if name in self.agents:
            del self.agents[name]
        if name in self._agent_grid_pos:
            del self._agent_grid_pos[name]
        if name in self._key_to_agent.values():
            key = self._agent_to_key.pop(name, None)
            if key:
                self._key_to_agent.pop(key, None)
        self._footer_extra = f"Recalled {name}"

    def agent_thinking(self, name: str) -> None:
        self._ensure_agent(name)
        self.agents[name].status = "thinking"
        self.agents[name].activity = "thinking..."

    def agent_done(self, name: str, summary: str = "") -> None:
        self._ensure_agent(name)
        self.agents[name].status = "done"
        self.agents[name].activity = summary or "done"

    def tool_call(self, name: str, tool_name: str) -> None:
        self._ensure_agent(name)
        self.agents[name].status = "working"
        self.agents[name].activity = tool_name

    def update_portfolio(self, **kw: float) -> None:
        self.portfolio.update(kw)

    def get_pending_hints(self) -> list[str]:
        return [h["text"] for h in self.hints if h["status"] == "pending"]

    def mark_hints_seen(self) -> None:
        for h in self.hints:
            if h["status"] == "pending":
                h["status"] = "seen"

    def update_tokens(
        self, name: str, input_tokens: int, output_tokens: int,
    ) -> None:
        self._ensure_agent(name)
        self.agents[name].input_tokens += input_tokens
        self.agents[name].output_tokens += output_tokens

    def wait(self, seconds: int, reason: str = "") -> None:
        mode = "POWER" if self.power_mode else reason or "pace"
        for remaining in range(seconds, 0, -1):
            # If power mode toggled mid-wait, cut the wait short
            if self.power_mode and remaining > 30:
                self._footer_extra = ""
                return
            self._footer_extra = f"Next cycle in {remaining}s [{mode}]"
            for s in self.agents.values():
                s.status = "idle"
                s.activity = ""
            time.sleep(1)
            # Re-check mode each second (power overrides reason)
            mode = "POWER" if self.power_mode else reason or "pace"
        self._footer_extra = ""

    def shutdown(self, summary: str) -> None:
        self.stop()
        self.console.print()
        self.console.print(Rule(" Shutdown ", style="bold red"))
        self.console.print(summary)
        self.console.print()

    # ── Sell from TUI ──────────────────────────────────────────

    def _initiate_sell(self) -> None:
        if not self.db or not self.trader:
            self._sell_result = "No trader connected"
            self._sell_result_ttl = 12
            return
        positions = self.db.get_positions()
        if not positions or self._selected_index >= len(positions):
            return
        pos = positions[self._selected_index]
        self._sell_confirm = pos["token_id"]
        self._sell_result = None

    def _execute_sell(self, token_id: str) -> None:
        if not self.trader or not hasattr(self.trader, "sell"):
            self._sell_result = "No trader"
            self._sell_result_ttl = 12
            return
        ok, msg = self.trader.sell(token_id)
        self._sell_result = msg
        self._sell_result_ttl = 20

    # ── Animation tick ──────────────────────────────────────────

    def _tick(self) -> None:
        self._frame += 1
        expired = [k for k, v in self._active_flows.items() if v <= 0]
        for k in expired:
            del self._active_flows[k]
        for k in list(self._active_flows):
            self._active_flows[k] -= 1
        # Decay finding burst animations
        burst_done = [k for k, v in self._burst_frames.items() if v <= 0]
        for k in burst_done:
            del self._burst_frames[k]
        for k in list(self._burst_frames):
            self._burst_frames[k] -= 1
        if self._sell_result_ttl > 0:
            self._sell_result_ttl -= 1
            if self._sell_result_ttl <= 0:
                self._sell_result = None
        # Decay surface burst
        sb_done = [k for k, v in self._surface_burst.items() if v <= 0]
        for k in sb_done:
            del self._surface_burst[k]
        for k in list(self._surface_burst):
            self._surface_burst[k] -= 1
        # Decay particle trails
        self._particle_trails = [
            (x, y, ttl - 1, c) for x, y, ttl, c in self._particle_trails if ttl > 1
        ]
        # Advance projectiles toward target
        import math as _m
        alive = []
        for proj in self._projectiles:
            proj["ttl"] -= 1
            if proj["ttl"] <= 0:
                continue
            dx = proj["tx"] - proj["x"]
            dy = proj["ty"] - proj["y"]
            dist = _m.sqrt(dx * dx + dy * dy)
            if dist < 2:
                continue  # arrived
            speed = 3.0
            proj["x"] += dx / dist * speed
            proj["y"] += dy / dist * speed * 0.5  # half speed vertical (aspect)
            alive.append(proj)
        self._projectiles = alive
        # Refresh trail cache from DB every ~8 frames (2 sec at 4fps)
        if self.db and self._frame - self._trail_cache_frame >= 8:
            self._trail_cache_frame = self._frame
            old_cache = dict(self._trail_cache)
            self._trail_cache.clear()
            for name in list(self.agents.keys()):
                try:
                    trail = self.db.get_agent_active_trail(name)
                    if trail:
                        self._trail_cache[name] = {
                            "depth": trail["depth"],
                            "status": trail["status"],
                            "topic": trail.get("topic", ""),
                        }
                except Exception:
                    pass
            # Detect agents that just surfaced (were in old cache, not in new)
            for name, old in old_cache.items():
                if name not in self._trail_cache and old.get("status") == "exploring":
                    self._surface_burst[name] = 16

    # ── Layout ──────────────────────────────────────────────────

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", size=34),
        )

        main_parts: list[Layout] = [
            Layout(name="header", size=3),
            Layout(name="content", ratio=1, minimum_size=8),
            Layout(name="fleet", size=3),
            Layout(name="footer", size=1),
        ]
        layout["main"].split_column(*main_parts)

        layout["sidebar"].split_column(
            Layout(name="memory", ratio=3),
            Layout(name="hints", ratio=2, minimum_size=8),
        )

        # Populate header / fleet / footer / sidebar
        layout["header"].update(self._hdr())
        layout["fleet"].update(self._fleet_panel())
        layout["footer"].update(self._ftr())
        layout["memory"].update(self._memory_panel())
        layout["hints"].update(self._hints_panel())

        # Main content: ball grid handles both normal + selected agent views
        if self._view_mode == "feed":
            active_agent = self._most_active_agent()
            if active_agent:
                layout["content"].split_row(
                    Layout(name="feed", ratio=1),
                    Layout(name="live_agent", ratio=1),
                )
                layout["feed"].update(self._feed())
                layout["live_agent"].update(self._live_agent_panel(active_agent))
            else:
                layout["content"].update(self._feed())
        elif self._view_mode == "positions":
            layout["content"].update(self._positions_view())
        elif self._view_mode == "hypotheses":
            layout["content"].update(self._hypotheses_view())
        else:
            # Default: space fleet view
            layout["content"].update(self._ball_grid())

        return layout

    def _fleet_panel(self) -> Panel:
        """Bottom fleet command bar showing ship counts and deploy keybinds."""
        counts = self._fleet_counts()
        parts = []
        for key, sc in SHIP_CLASSES.items():
            cur = counts.get(key, 0)
            max_c = sc.max_count
            if cur > 0:
                bar = "█" * cur + "░" * (max_c - cur)
                style = f"bold {sc.color}"
            else:
                bar = "░" * max_c
                style = "dim"
            parts.append(
                f"[{style}]{sc.key.upper()}[/{style}] "
                f"[bold]{sc.label}[/bold] "
                f"[{style}]{bar}[/{style}] "
                f"{cur}/{max_c}"
            )
        fleet_str = "  │  ".join(parts)
        total = sum(counts.values())
        total_max = sum(sc.max_count for sc in SHIP_CLASSES.values())
        extra = ""
        if self._footer_extra:
            extra = f"  [dim]│  {self._footer_extra}[/dim]"
        return Panel(
            Text.from_markup(
                f"  {fleet_str}  [dim]│[/dim]  "
                f"Fleet [bold]{total}[/bold]/{total_max}{extra}"
            ),
            title="[bold]FLEET COMMAND[/bold]",
            title_align="left",
            border_style="dim yellow",
            padding=(0, 1),
        )

    # ── Header ──────────────────────────────────────────────────

    def _hdr(self) -> Panel:
        p = self.portfolio
        ps = "green" if p["pnl"] >= 0 else "red"
        total_tok = sum(
            a.input_tokens + a.output_tokens for a in self.agents.values()
        )
        tok_part = f"  |  Tokens [dim]{total_tok:,}[/dim]" if total_tok else ""
        n_agents = sum(1 for a in self.agents.values() if a.status != "idle" or a.input_tokens > 0)
        mode_badge = (
            "  [bold red]POWER[/bold red]" if self.power_mode
            else ""
        )
        return Panel(
            Text.from_markup(
                f"[bold cyan]POLYBILLIONAIRE[/bold cyan]{mode_badge}   "
                f"Agents [bold]{n_agents}[/bold]  |  "
                f"Portfolio [bold]${p['value']:.2f}[/bold]  |  "
                f"Free [bold]${p['bankroll']:.2f}[/bold]  |  "
                f"In Positions [bold]${p['value'] - p['bankroll']:.2f}[/bold] "
                f"({p['positions']} open)  |  "
                f"P&L [{ps}]${p['pnl']:+.4f}[/{ps}]{tok_part}"
            ),
            style="cyan",
        )

    # ── Status line ────────────────────────────────────────────

    def _status_line(self) -> Text:
        active = [
            (n, s) for n, s in self.agents.items()
            if s.status in ("thinking", "working")
        ]
        if active:
            parts = []
            for name, st in active:
                c = _color(name)
                act = (st.activity or "working")[:20]
                parts.append(f"[{c}]{name}[/{c}] {act}")
            return Text.from_markup(
                f"  [bold]\u25b8[/bold] {' \u2502 '.join(parts)}"
            )
        if self._footer_extra:
            return Text.from_markup(f"  [dim]{self._footer_extra}[/dim]")
        return Text.from_markup("  [dim]Idle[/dim]")

    # ── Ship renderer ────────────────────────────────────────────

    _STATUS_COLORS = {
        "idle": "white", "thinking": "yellow", "working": "green",
        "cooldown": "blue", "done": "green", "error": "red",
    }

    def _seed_stars(self, w: int, h: int) -> None:
        """Seed the parallax starfield once. 3 layers at different densities."""
        import random
        rng = random.Random(42)
        self._stars = []
        # Layer 0 (far, dim, dense): ~3% fill
        for _ in range(int(w * h * 0.025)):
            self._stars.append((rng.randint(0, w - 1), rng.randint(0, h - 1), 0))
        # Layer 1 (mid): ~1.5%
        for _ in range(int(w * h * 0.012)):
            self._stars.append((rng.randint(0, w - 1), rng.randint(0, h - 1), 1))
        # Layer 2 (near, bright, sparse): ~0.4%
        for _ in range(int(w * h * 0.004)):
            self._stars.append((rng.randint(0, w - 1), rng.randint(0, h - 1), 2))
        self._stars_seeded = True

    def _ball_grid(self) -> Panel:
        """Space Fleet Command — ships float in a parallax starfield.
        Each agent is rendered as its ship class sprite. Active ships
        show engine exhaust. Findings trigger projectile animations.
        """
        import math

        sidebar_w = 36
        canvas_w = max(30, self.console.width - sidebar_w)
        canvas_h = max(8, self.console.height - 10)  # leave room for fleet panel
        t = self._frame * 0.3

        # Seed starfield on first render or resize
        if not self._stars_seeded:
            self._seed_stars(canvas_w, canvas_h)

        # Ship sprite dimensions
        sprite_h = 5  # all sprites are 5-6 lines
        cell_w = 18
        cell_h = sprite_h + 4

        # ── Compute ship positions ─────────────────────────────
        agent_positions: dict[str, tuple[int, int]] = {}  # name -> (px, py)
        agent_colors: dict[str, str] = {}
        selected = self._selected_agent
        center_x, center_y = canvas_w // 2, (canvas_h - 4) // 2

        for name in sorted(self.agents.keys()):
            st = self.agents.get(name, AgentStatus())
            gx, gy = self._agent_grid_pos.get(name, (0, 0))

            base_x = gx * cell_w + cell_w // 2
            base_y = gy * cell_h + cell_h // 2

            # Drift — each ship has unique orbital motion
            phase = hash(name) % 100 * 0.0628
            active = st.status in ("thinking", "working")
            speed = 0.6 if active else 0.3
            amp_x = 4 if active else 2
            amp_y = 1.5 if active else 0.8
            drift_x = math.sin(t * speed + phase) * amp_x
            drift_y = math.cos(t * speed * 0.7 + phase * 1.3) * amp_y
            drift_x += math.sin(t * 1.1 + phase * 2.7) * 1.2
            drift_y += math.cos(t * 0.8 + phase * 3.1) * 0.5

            if selected == name:
                px = center_x
                py = center_y - 2
            elif selected:
                dx_from_center = base_x - center_x
                dy_from_center = base_y - center_y
                px = int(center_x + dx_from_center * 1.3 + drift_x)
                py = int(center_y + dy_from_center * 1.3 + drift_y)
            else:
                px = int(base_x + drift_x)
                py = int(base_y + drift_y)

            # Resolve ship color
            ship_key = self._get_ship_class(name)
            sc = SHIP_CLASSES.get(ship_key or "")
            base_color = sc.color if sc else "white"
            if self._surface_burst.get(name, 0) > 0:
                color = "bold bright_white"
            elif active:
                color = f"bold {base_color}"
            elif st.status == "idle" and (st.input_tokens + st.output_tokens) == 0:
                color = f"dim {base_color}"
            else:
                color = base_color
            agent_colors[name] = color
            agent_positions[name] = (px, py)

            # Engine exhaust particles for active ships
            if active and self._frame % 3 == 0:
                ex = px - 2 if ship_key in ("scout", "battlecruiser") else px
                ey = py + 3 if ship_key == "probe" else py
                self._particle_trails.append((ex, ey, 12, base_color))

        # ── Build 2D canvas ────────────────────────────────────
        chars = [[" "] * canvas_w for _ in range(canvas_h)]
        styles = [[""] * canvas_w for _ in range(canvas_h)]

        # ── Parallax starfield ─────────────────────────────────
        star_chars = [".", "·", "*"]
        star_styles = ["bright_black", "dim", "bold white"]
        star_speeds = [0.15, 0.35, 0.7]
        for sx, sy, layer in self._stars:
            # Parallax drift — each layer scrolls at different speed
            drift = int(t * star_speeds[layer])
            dx = (sx + drift) % canvas_w
            dy = sy % canvas_h
            if 0 <= dx < canvas_w and 0 <= dy < canvas_h:
                # Twinkle: some stars blink
                if layer == 2:
                    twinkle = math.sin(t * 2.0 + sx * 0.3 + sy * 0.7)
                    if twinkle < -0.3:
                        continue  # star blinks off
                    if twinkle > 0.7:
                        chars[dy][dx] = "✦"
                        styles[dy][dx] = "bold cyan"
                        continue
                chars[dy][dx] = star_chars[layer]
                styles[dy][dx] = star_styles[layer]

        # ── Render exhaust particle trails ─────────────────────
        for px, py, ttl, pc in self._particle_trails:
            if 0 <= px < canvas_w and 0 <= py < canvas_h:
                fade = ttl / 12.0
                if fade > 0.7:
                    chars[py][px] = "░"
                    styles[py][px] = pc
                elif fade > 0.4:
                    chars[py][px] = "·"
                    styles[py][px] = f"dim {pc}"
                elif fade > 0.15:
                    chars[py][px] = "."
                    styles[py][px] = "bright_black"

        # ── Render projectiles ─────────────────────────────────
        for proj in self._projectiles:
            px, py = int(proj["x"]), int(proj["y"])
            if 0 <= px < canvas_w and 0 <= py < canvas_h:
                chars[py][px] = proj["char"]
                styles[py][px] = proj["color"]

        # ── Stamp ship sprites ─────────────────────────────────
        render_order = sorted(
            agent_positions.keys(),
            key=lambda n: (n == selected,),
        )

        for name in render_order:
            st = self.agents.get(name, AgentStatus())
            color = agent_colors.get(name, "white")
            ax, ay = agent_positions[name]
            is_fore = name == selected
            is_dim = selected and not is_fore
            sb = self._surface_burst.get(name, 0)

            # Get the right sprite
            ship_key = self._get_ship_class(name) or "mothership"
            sprite_lines = SHIP_SPRITES.get(ship_key, SHIP_SPRITES["mothership"])

            # Compute sprite bounds
            sh = len(sprite_lines)
            sw = max(len(line) for line in sprite_lines)
            ox = ax - sw // 2
            oy = ay - sh // 2

            # Surface burst: expanding ring around ship
            if sb > 0:
                ring_r = (16 - sb) * 1.2 + 2
                for ring_a in range(0, 360, 8):
                    rad = math.radians(ring_a)
                    rx = int(ax + math.cos(rad) * ring_r)
                    ry = int(ay + math.sin(rad) * ring_r * 0.5)
                    if 0 <= rx < canvas_w and 0 <= ry < canvas_h:
                        chars[ry][rx] = "◈" if sb > 8 else "◇"
                        styles[ry][rx] = "bold bright_white" if sb > 8 else "dim white"

            # Stamp sprite characters
            draw_style = f"bold {color}" if is_fore else ("dim" if is_dim else color)
            for ly, line in enumerate(sprite_lines):
                for lx, ch in enumerate(line):
                    if ch == " ":
                        continue
                    px, py = ox + lx, oy + ly
                    if 0 <= px < canvas_w and 0 <= py < canvas_h:
                        chars[py][px] = ch
                        styles[py][px] = draw_style

            # Label below ship
            key = self._agent_to_key.get(name, "")
            tok = st.input_tokens + st.output_tokens
            tok_s = f"{tok:,}" if tok else "0"
            act = (st.activity or st.status)[:12]
            label1 = f"{name} [{key}]"
            label2 = f"{act} {tok_s}t"
            label_y = oy + sh
            label_x = ax - len(label1) // 2
            lbl_style = f"bold {color}" if is_fore else ("dim" if is_dim else "bold")
            for i, ch in enumerate(label1):
                px = label_x + i
                if 0 <= px < canvas_w and 0 <= label_y < canvas_h:
                    chars[label_y][px] = ch
                    styles[label_y][px] = lbl_style
            label_x2 = ax - len(label2) // 2
            if 0 <= label_y + 1 < canvas_h:
                for i, ch in enumerate(label2):
                    px = label_x2 + i
                    if 0 <= px < canvas_w:
                        chars[label_y + 1][px] = ch
                        styles[label_y + 1][px] = "dim"

            # Selected ship: show output log
            if is_fore:
                raw = self._last_output.get(name, "")
                if raw:
                    log_start_y = label_y + 2
                    log_lines = raw.strip().split("\n")
                    max_log = canvas_h - log_start_y - 1
                    for li, line in enumerate(log_lines[-max_log:]):
                        ly = log_start_y + li
                        if ly >= canvas_h:
                            break
                        lx = max(2, ax - len(line) // 2)
                        for ci, ch in enumerate(line[:canvas_w - lx - 1]):
                            px = lx + ci
                            if 0 <= px < canvas_w:
                                chars[ly][px] = ch
                                styles[ly][px] = "dim"

        # ── Convert canvas to Rich Text ────────────────────────
        result = Text()
        for y in range(canvas_h):
            x = 0
            while x < canvas_w:
                style = styles[y][x]
                span_start = x
                while x < canvas_w and styles[y][x] == style:
                    x += 1
                segment = "".join(chars[y][span_start:x])
                result.append(segment, style=style or None)
            result.append("\n")

        return Panel(result, border_style="dim cyan", padding=0)

    def _mini_dots(self, exclude: str = "") -> Text:
        """Collapsed agent dots: (●) R-2 thinking  (○) Reasoning idle"""
        parts: list[str] = []
        for name, st in self.agents.items():
            if name == exclude:
                continue
            color = self._STATUS_COLORS.get(st.status, "white")
            active = st.status in ("thinking", "working")
            dot = f"[{color}]\u25cf[/{color}]" if active else f"[dim]\u25cb[/dim]"
            key = self._agent_to_key.get(name, "")
            key_lbl = f"[dim][{key}][/dim]" if key else ""
            act = (st.activity or st.status)[:10]
            parts.append(f" ({dot}) {name}{key_lbl} [dim]{act}[/dim]")
        return Text.from_markup("".join(parts) or " [dim]no other agents[/dim]")

    def _snippet(self, name: str, max_len: int) -> str:
        raw = self._last_output.get(name, "")
        if not raw:
            return ""
        for line in raw.split("\n"):
            line = line.strip()
            if line and not line.startswith("---") and not line.startswith("=="):
                text = line[:max_len]
                if len(line) > max_len:
                    text += "\u2026"
                return text
        return raw[:max_len].replace("\n", " ")

    # ── Positions view (master-detail) ─────────────────────────

    def _positions_view(self) -> Panel:
        if not self.db:
            return Panel(
                "[dim]No database connected[/dim]",
                title="[bold green]Positions[/bold green]",
                border_style="green",
            )

        positions = self.db.get_positions()
        if not positions:
            return Panel(
                "[dim]No open positions[/dim]\n\n"
                "[dim]Trades will appear here once the org executes them.[/dim]",
                title="[bold green]Positions[/bold green]",
                border_style="green",
                padding=(1, 2),
            )

        # Clamp selection
        self._selected_index = max(0, min(self._selected_index, len(positions) - 1))

        inner = Layout()
        inner.split_row(
            Layout(name="list", ratio=2, minimum_size=30),
            Layout(name="detail", ratio=3, minimum_size=40),
        )

        inner["list"].update(self._positions_list(positions))
        inner["detail"].update(
            self._position_detail(positions[self._selected_index])
        )

        # Sell confirmation / result banner
        subtitle = ""
        if self._sell_confirm:
            pos = next(
                (p for p in positions if p["token_id"] == self._sell_confirm),
                None,
            )
            name = (pos["market_question"][:30] if pos else "?")
            subtitle = (
                f"[bold red] SELL \"{name}\"? "
                f"[bold]y[/bold] confirm / any key cancel [/bold red]"
            )
        elif self._sell_result:
            subtitle = f"[bold yellow] {self._sell_result} [/bold yellow]"

        return Panel(
            inner,
            title="[bold green]Positions[/bold green]  [dim]j/k navigate  x sell[/dim]",
            subtitle=subtitle or None,
            border_style="green",
            padding=(0, 0),
        )

    def _positions_list(self, positions: list[dict]) -> Panel:
        table = Table(
            show_header=True, box=ROUNDED, expand=True,
            border_style="dim", padding=(0, 1),
        )
        table.add_column("", width=1)
        table.add_column("Market", ratio=3)
        table.add_column("Side", width=4)
        table.add_column("Entry", width=7, justify="right")
        table.add_column("Now", width=7, justify="right")
        table.add_column("P&L", width=9, justify="right")
        table.add_column("Expires", width=12)

        for i, p in enumerate(positions):
            selected = i == self._selected_index
            marker = "[bold cyan]\u25b8[/bold cyan]" if selected else " "
            pnl = p.get("pnl") or 0
            pc = "green" if pnl >= 0 else "red"

            question = (p.get("market_question") or "?")[:28]
            if len(p.get("market_question") or "") > 28:
                question += "\u2026"

            end = ""
            if p.get("end_date"):
                end = p["end_date"][:10]

            row_style = "on grey15" if selected else ""
            table.add_row(
                marker,
                question,
                p.get("side", "")[:3],
                f"${p.get('entry_price', 0):.3f}",
                f"${p.get('current_price', 0):.3f}",
                f"[{pc}]${pnl:+.4f}[/{pc}]",
                f"[dim]{end}[/dim]",
                style=row_style,
            )

        return Panel(table, border_style="dim", padding=(0, 0))

    def _position_detail(self, pos: dict) -> Panel:
        lines: list[str] = []

        # Position header
        question = pos.get("market_question") or "Unknown"
        lines.append(f"[bold]{question}[/bold]")
        outcome = pos.get("outcome", "")
        side = pos.get("side", "")
        if outcome or side:
            lines.append(f"  {outcome} ({side})")
        lines.append("")

        # Price block
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", 0)
        pnl = pos.get("pnl", 0)
        size = pos.get("size", 0)
        cost = pos.get("cost", 0)
        pc = "green" if pnl >= 0 else "red"
        pct = ((current - entry) / entry * 100) if entry else 0

        lines.append(
            f"  Entry [bold]${entry:.4f}[/bold]  "
            f"Current [bold]${current:.4f}[/bold]  "
            f"P&L [{pc}]${pnl:+.4f} ({pct:+.1f}%)[/{pc}]"
        )
        lines.append(f"  Size [bold]{size:.2f}[/bold]  Cost [bold]${cost:.4f}[/bold]")
        meta_parts: list[str] = []
        if pos.get("entry_cycle"):
            meta_parts.append(f"Cycle {pos['entry_cycle']}")
        if pos.get("end_date"):
            end = pos["end_date"][:16].replace("T", " ")
            meta_parts.append(f"Expires [bold]{end}[/bold]")
        if meta_parts:
            lines.append(f"  [dim]{' | '.join(meta_parts)}[/dim]")
        lines.append("")

        # Linked hypothesis reasoning chain
        hyp_id = pos.get("hypothesis_id")
        if hyp_id and self.db:
            h = self.db.get_hypothesis(hyp_id)
            if h:
                lines.extend(self._reasoning_chain(h))
            else:
                lines.append("[dim]Hypothesis not found[/dim]")
        else:
            lines.append("[dim]No linked hypothesis[/dim]")
            lines.append(
                "[dim]Position was opened without a specific thesis.[/dim]"
            )

        avail = max(5, self.console.height - 22)
        if len(lines) > avail:
            lines = lines[:avail - 1]
            lines.append("[dim]...[/dim]")

        return Panel(
            Text.from_markup("\n".join(lines), overflow="fold"),
            title="[bold]Detail[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _reasoning_chain(self, h: dict) -> list[str]:
        """Build the visual reasoning chain for a hypothesis."""
        lines: list[str] = []

        ind = HYP_INDICATOR.get(h["status"], "[dim]?[/dim]")
        edge_str = f" [yellow]edge {h['edge']:+.0%}[/yellow]" if h.get("edge") else ""
        side_str = f" {h['side']}" if h.get("side") else ""
        lines.append(
            f"[bold dim]HYPOTHESIS[/bold dim]  "
            f"[dim]{h['status']}[/dim]"
        )
        lines.append(f"  {ind} [cyan]{h['title']}[/cyan]{side_str}{edge_str}")

        if h.get("thesis") and h["thesis"] != h["title"]:
            thesis = h["thesis"]
            for i in range(0, len(thesis), 55):
                chunk = thesis[i:i + 55]
                lines.append(f"    [dim]{chunk}[/dim]")

        if h.get("market_price") is not None:
            mp = h["market_price"]
            op = h.get("our_probability")
            price_str = f"  Market ${mp:.3f}"
            if op is not None:
                price_str += f"  Our est {op:.0%}"
            lines.append(f"[dim]{price_str}[/dim]")

        lines.append("")

        # Supporting leads
        if not self.db:
            return lines
        leads = self.db.get_leads_for_hypothesis(h["id"])
        if leads:
            lines.append(f"[bold dim]LEADS[/bold dim]  [dim]({len(leads)})[/dim]")
            for i, lead in enumerate(leads[:6]):
                conf = lead.get("confidence") or "?"
                cc = CONF_COLOR.get(conf, "dim")
                is_last = i == min(len(leads), 6) - 1
                connector = "\u2514" if is_last else "\u251c"
                lines.append(
                    f"  [{cc}]{connector}[/{cc}] "
                    f"[{cc}]{conf:>6}[/{cc}]  "
                    f"{(lead['title'] or '')[:38]}"
                )
                if lead.get("signal"):
                    signal = lead["signal"][:50]
                    pipe = " " if is_last else "\u2502"
                    lines.append(f"  [dim]{pipe}         {signal}[/dim]")

                # Most recent research trail entry
                trail = self.db.get_research_trail(lead["id"])
                if trail:
                    latest = trail[-1]
                    finding = (latest["finding"] or "")[:48]
                    pipe = " " if is_last else "\u2502"
                    lines.append(
                        f"  [dim]{pipe}         "
                        f"C{latest['cycle']} [{latest.get('agent', 'R')}]: "
                        f"{finding}[/dim]"
                    )
        else:
            lines.append("[dim]No leads yet[/dim]")

        return lines

    # ── Hypotheses view ────────────────────────────────────────

    def _hypotheses_view(self) -> Panel:
        if not self.db:
            return Panel(
                "[dim]No database connected[/dim]",
                title="[bold yellow]Hypotheses[/bold yellow]",
                border_style="yellow",
            )

        hyps = self.db.get_active_hypotheses()
        if not hyps:
            return Panel(
                "[dim]No active hypotheses yet[/dim]\n\n"
                "[dim]The CEO agent will generate hypotheses during cycles.[/dim]",
                title="[bold yellow]Hypotheses[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )

        self._selected_index = max(0, min(self._selected_index, len(hyps) - 1))

        inner = Layout()
        inner.split_row(
            Layout(name="list", ratio=2, minimum_size=30),
            Layout(name="detail", ratio=3, minimum_size=40),
        )

        inner["list"].update(self._hypotheses_list(hyps))
        inner["detail"].update(self._hypothesis_detail(hyps[self._selected_index]))

        return Panel(
            inner,
            title="[bold yellow]Hypotheses[/bold yellow]  [dim]j/k navigate[/dim]",
            border_style="yellow",
            padding=(0, 0),
        )

    def _hypotheses_list(self, hyps: list[dict]) -> Panel:
        table = Table(
            show_header=True, box=ROUNDED, expand=True,
            border_style="dim", padding=(0, 1),
        )
        table.add_column("", width=1)
        table.add_column("", width=1)
        table.add_column("Title", ratio=3)
        table.add_column("Status", width=13)
        table.add_column("Edge", width=6, justify="right")
        table.add_column("Side", width=4)

        for i, h in enumerate(hyps):
            selected = i == self._selected_index
            marker = "[bold cyan]\u25b8[/bold cyan]" if selected else " "
            ind = HYP_INDICATOR.get(h["status"], "[dim]?[/dim]")

            title = (h["title"] or "?")[:30]
            if len(h.get("title") or "") > 30:
                title += "\u2026"

            edge = ""
            if h.get("edge") is not None:
                ec = "green" if h["edge"] >= 0 else "red"
                edge = f"[{ec}]{h['edge']:+.0%}[/{ec}]"

            row_style = "on grey15" if selected else ""
            table.add_row(
                marker, ind, title, h["status"],
                edge, h.get("side") or "",
                style=row_style,
            )

        return Panel(table, border_style="dim", padding=(0, 0))

    def _hypothesis_detail(self, h: dict) -> Panel:
        lines: list[str] = []

        # Header
        lines.append(f"[bold]{h['title']}[/bold]")
        lines.append(
            f"  [dim]Status:[/dim] {h['status']}  "
            f"[dim]Category:[/dim] {h.get('category') or 'none'}"
        )
        lines.append("")

        # Thesis
        if h.get("thesis") and h["thesis"] != h["title"]:
            lines.append("[bold dim]THESIS[/bold dim]")
            thesis = h["thesis"]
            for i in range(0, len(thesis), 55):
                lines.append(f"  {thesis[i:i + 55]}")
            lines.append("")

        # Market data
        if h.get("market_question"):
            lines.append(f"[bold dim]MARKET[/bold dim]")
            lines.append(f"  {h['market_question']}")
            parts = []
            if h.get("market_price") is not None:
                parts.append(f"Price ${h['market_price']:.3f}")
            if h.get("our_probability") is not None:
                parts.append(f"Our est {h['our_probability']:.0%}")
            if h.get("edge") is not None:
                ec = "green" if h["edge"] >= 0 else "red"
                parts.append(f"Edge [{ec}]{h['edge']:+.0%}[/{ec}]")
            if h.get("side"):
                parts.append(f"Side {h['side']}")
            if parts:
                lines.append(f"  {' | '.join(parts)}")
            lines.append("")

        # Linked positions
        if self.db:
            db_positions = self.db.get_positions()
            linked = [p for p in db_positions if p.get("hypothesis_id") == h["id"]]
            if linked:
                lines.append(
                    f"[bold dim]POSITIONS[/bold dim]  [dim]({len(linked)})[/dim]"
                )
                for p in linked[:5]:
                    pnl = p.get("pnl", 0)
                    pc = "green" if pnl >= 0 else "red"
                    q = (p.get("market_question") or "?")[:35]
                    lines.append(
                        f"  {p.get('side', '')} {q}  "
                        f"[{pc}]${pnl:+.4f}[/{pc}]"
                    )
                lines.append("")

            # P&L from closed trades
            pnl_data = self.db.get_hypothesis_pnl(h["id"])
            if pnl_data and pnl_data.get("trades", 0) > 0:
                tc = "green" if pnl_data["total_pnl"] >= 0 else "red"
                lines.append("[bold dim]TRACK RECORD[/bold dim]")
                lines.append(
                    f"  {pnl_data['trades']} trades  "
                    f"W {pnl_data.get('wins', 0)} / L {pnl_data.get('losses', 0)}  "
                    f"Total [{tc}]${pnl_data['total_pnl']:+.4f}[/{tc}]"
                )
                lines.append("")

        # Leads + research chain
        if self.db:
            leads = self.db.get_leads_for_hypothesis(h["id"])
            if leads:
                lines.append(
                    f"[bold dim]LEADS[/bold dim]  [dim]({len(leads)})[/dim]"
                )
                for i, lead in enumerate(leads[:8]):
                    conf = lead.get("confidence") or "?"
                    cc = CONF_COLOR.get(conf, "dim")
                    status = lead.get("status", "new")
                    is_last = i == min(len(leads), 8) - 1
                    connector = "\u2514" if is_last else "\u251c"
                    lines.append(
                        f"  [{cc}]{connector}[/{cc}] "
                        f"[{cc}]{conf:>6}[/{cc}]  "
                        f"{(lead['title'] or '')[:35]}  "
                        f"[dim]{status}[/dim]"
                    )
                    if lead.get("source"):
                        pipe = " " if is_last else "\u2502"
                        lines.append(
                            f"  [dim]{pipe}         {lead['source'][:45]}[/dim]"
                        )

                    trail = self.db.get_research_trail(lead["id"])
                    for entry in trail[-2:]:
                        finding = (entry["finding"] or "")[:45]
                        pipe = " " if is_last else "\u2502"
                        lines.append(
                            f"  [dim]{pipe}         "
                            f"C{entry['cycle']}: {finding}[/dim]"
                        )

        avail = max(5, self.console.height - 22)
        if len(lines) > avail:
            lines = lines[:avail - 1]
            lines.append("[dim]...[/dim]")

        return Panel(
            Text.from_markup("\n".join(lines), overflow="fold"),
            title="[bold]Detail[/bold]",
            border_style="yellow",
            padding=(0, 1),
        )

    # ── Agent detail window ────────────────────────────────────

    def _agent_window(self, name: str) -> Panel:
        color = _color(name)
        st = self.agents.get(name, AgentStatus())
        dot = STATUS_DOT.get(st.status, STATUS_DOT["idle"])
        tok = st.input_tokens + st.output_tokens

        filtered = [
            m for m in self.messages if m.sender == name or m.to == name
        ]

        all_lines: list[str] = []
        for msg in filtered:
            all_lines.extend(self._build_msg_lines(msg))

        avail = max(
            5, (self.console.height - 18) // max(1, len(self._open_windows)),
        )

        max_scroll = max(0, len(all_lines) - avail)
        scroll = min(self._window_scroll.get(name, max_scroll), max_scroll)
        self._window_scroll[name] = scroll

        visible = all_lines[scroll:scroll + avail] if all_lines else [
            "[dim]No output yet...[/dim]",
        ]

        scroll_info = (
            f" {int(scroll / max_scroll * 100)}%"
            if max_scroll > 0 else ""
        )
        key = self._agent_to_key.get(name, "")

        return Panel(
            Text.from_markup("\n".join(visible), overflow="fold"),
            title=(
                f"{dot} [bold {color}]{name}[/bold {color}]  "
                f"[dim]{tok:,} tok{scroll_info}[/dim]"
            ),
            subtitle=f"[dim][{key}] close  \u2191\u2193 scroll[/dim]",
            border_style=color,
            padding=(0, 1),
        )

    # ── Live agent panel (auto-follows active agent) ────────────

    def _most_active_agent(self) -> str | None:
        """Return the name of the currently working/thinking agent."""
        for name, st in self.agents.items():
            if st.status in ("thinking", "working"):
                return name
        # If nobody's active, show the last agent that produced output
        if self.messages:
            return self.messages[-1].sender
        return None

    def _live_agent_panel(self, name: str) -> Panel:
        """Show the latest output from the active agent — auto-scrolled."""
        color = _color(name)
        st = self.agents.get(name, AgentStatus())
        dot = STATUS_DOT.get(st.status, STATUS_DOT["idle"])
        tok = st.input_tokens + st.output_tokens

        raw = self._last_output.get(name, "")
        if not raw:
            return Panel(
                Text.from_markup(f"[dim]Waiting for {name}...[/dim]"),
                title=f"{dot} [bold {color}]{name}[/bold {color}]",
                border_style=color,
                padding=(0, 1),
            )

        avail = max(5, self.console.height - 20)
        lines = raw.split("\n")
        if len(lines) > avail:
            lines = lines[-avail:]

        active = st.status in ("thinking", "working")
        box = DOUBLE if active else ROUNDED
        bstyle = f"bold {color}" if active else color

        return Panel(
            Text.from_markup(
                "\n".join(lines), overflow="fold",
            ),
            title=(
                f"{dot} [bold {color}]{name}[/bold {color}]  "
                f"[dim]{tok:,} tok[/dim]"
            ),
            box=box,
            border_style=bstyle,
            padding=(0, 1),
        )

    # ── Activity feed (when no windows open) ───────────────────

    def _feed(self) -> Panel:
        avail = self.console.height - 18
        if avail < 4:
            avail = 4

        all_lines: list[str] = []
        for msg in self.messages:
            all_lines.extend(self._build_msg_lines(msg))

        if len(all_lines) > avail:
            all_lines = all_lines[-avail:]

        content = (
            "\n".join(all_lines)
            if all_lines
            else "[dim]Waiting for agents...\n\n"
                 "  [bold]p[/bold] positions  "
                 "[bold]h[/bold] hypotheses  "
                 "[bold]1[/bold]-[bold]6[/bold] agents  "
                 "[bold]i[/bold] hint[/dim]"
        )
        return Panel(
            Text.from_markup(content, overflow="fold"),
            title="[bold]Activity Feed[/bold]",
            border_style="dim",
            padding=(0, 1),
        )

    def _build_msg_lines(self, msg: Message) -> list[str]:
        color = _color(msg.sender)
        label, ls = KIND_LABELS.get(msg.kind, (msg.kind.upper(), "white"))
        target = f" -> {msg.to}" if msg.to != "all" else ""
        block: list[str] = [
            f"[{color}]\u2502[/{color}] "
            f"[{ls}]\\[{label}][/{ls}] "
            f"[bold {color}]{msg.sender}[/bold {color}]"
            f"[dim]{target}[/dim]"
        ]
        for ln in msg.content.strip().split("\n"):
            block.append(f"[{color}]\u2502[/{color}]   {ln}")
        block.append("")
        return block

    # ── Memory panel ────────────────────────────────────────────

    def _memory_panel(self) -> Panel:
        if not self.db:
            return Panel(
                "[dim]No database connected[/dim]",
                title="[bold]Org Memory[/bold]",
                border_style="dim",
                padding=(0, 1),
            )

        lines: list[str] = []

        d = self.db.get_direction()
        if d:
            thesis = d["thesis"]
            if len(thesis) > 60:
                thesis = thesis[:57] + "..."
            lines.append("[bold dim]DIRECTION[/bold dim]")
            lines.append(f"  [italic]{thesis}[/italic]")
        else:
            lines.append("[dim]No direction set yet[/dim]")

        lines.append("")

        hyps = self.db.get_active_hypotheses()
        if hyps:
            lines.append(
                f"[bold dim]HYPOTHESES[/bold dim] [dim]({len(hyps)})[/dim]",
            )
            for h in hyps[:8]:
                title = h["title"]
                if len(title) > 26:
                    title = title[:23] + "..."
                status = h["status"]
                ind = HYP_INDICATOR.get(status, "[dim].[/dim]")

                edge_str = ""
                if h["edge"] is not None:
                    sign = "+" if h["edge"] >= 0 else ""
                    ec = "green" if h["edge"] >= 0 else "red"
                    edge_str = f" [{ec}]{sign}{h['edge']:.0%}[/{ec}]"

                side_str = f" {h['side']}" if h["side"] else ""
                lines.append(f"  {ind} {title}{side_str}{edge_str}")
        else:
            lines.append("[dim]No hypotheses yet[/dim]")

        lines.append("")

        active_leads = self.db.get_active_leads()
        recent_rejs = self.db.get_recent_rejections(limit=100)
        stats: list[str] = []
        if active_leads:
            stats.append(f"{len(active_leads)} leads")
        if recent_rejs:
            stats.append(f"{len(recent_rejs)} rejected")
        if stats:
            lines.append(f"[dim]{' | '.join(stats)}[/dim]")

        if self.cycle > 0:
            c = self.db.get_cycle(self.cycle)
            if c and c.get("duration_s"):
                trades = c.get("trades_executed", 0)
                t = f" | {trades} trades" if trades else ""
                lines.append(
                    f"[dim]Cycle {self.cycle}: {c['duration_s']:.0f}s{t}[/dim]",
                )

        return Panel(
            "\n".join(lines),
            title="[bold]Org Memory[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

    # ── Hints panel ─────────────────────────────────────────────

    def _hints_panel(self) -> Panel:
        lines: list[str] = []
        for h in self.hints:
            if h["status"] == "pending":
                lines.append(f"[yellow]  [bold]>[/bold] {h['text']}[/yellow]")
            else:
                lines.append(f"[dim]  [green]\u2713[/green] {h['text']}[/dim]")

        if not lines:
            lines.append("[dim]  No hints yet[/dim]")

        cursor = "\u258c" if self._frame % 4 < 2 else " "
        if self._input_mode:
            lines.append(
                f"\n[bold yellow]> {self._input_buffer}{cursor}[/bold yellow]",
            )
            lines.append("[dim]  Enter submit | Esc cancel[/dim]")
        else:
            lines.append(f"\n[dim]  press [bold]i[/bold] to type a hint[/dim]")

        border = "bold yellow" if self._input_mode else "yellow"
        title = (
            "[bold yellow on red] TYPING HINT [/bold yellow on red]"
            if self._input_mode
            else "[bold yellow]Your Intuition[/bold yellow]"
        )
        return Panel(
            "\n".join(lines),
            title=title,
            border_style=border,
            padding=(1, 1),
        )

    # ── Footer ──────────────────────────────────────────────────

    def _ftr(self) -> Text:
        if self._input_mode:
            return Text.from_markup(
                "  [bold yellow]HINT INPUT[/bold yellow]  "
                "[dim]Enter submit  |  Esc cancel[/dim]"
            )

        # View indicator
        mode_labels = {
            "balls": "swarm", "feed": "feed",
            "positions": "positions", "hypotheses": "hypotheses",
        }
        current = mode_labels.get(self._view_mode, "swarm")
        if self._selected_agent:
            key = self._agent_to_key.get(self._selected_agent, "")
            current = f"{self._selected_agent} [{key}]"

        if self._sell_confirm:
            return Text.from_markup(
                "  [bold red]CONFIRM SELL[/bold red]  "
                "[dim]y confirm  |  any key cancel[/dim]"
            )

        nav = ""
        if self._view_mode == "positions" and not self._open_windows:
            nav = "  [dim]j/k navigate  x sell[/dim]"
        elif self._view_mode == "hypotheses" and not self._open_windows:
            nav = "  [dim]j/k navigate[/dim]"
        elif self._open_windows:
            nav = "  [dim]\u2191\u2193 scroll[/dim]"

        power_label = (
            "[bold red]POWER[/bold red]" if self.power_mode
            else "[dim]pace[/dim]"
        )
        parts = (
            f"  [bold cyan]{current}[/bold cyan]{nav}  [dim]|[/dim]  "
            f"[dim][bold]f[/bold]eed  "
            f"[bold]p[/bold]os  "
            f"[bold]h[/bold]yp  "
            f"[bold]1[/bold]-[bold]9[/bold] select  "
            f"[bold]s[/bold]cout  "
            f"[bold]d[/bold]ive  "
            f"[bold]c[/bold]ontra  "
            f"[bold]r[/bold]eason  "
            f"SHIFT=recall  "
            f"[bold]i[/bold] hint  "
            f"[bold]g[/bold] {power_label}  "
            f"Esc back[/dim]"
        )

        if self.model:
            m = self.model if len(self.model) <= 20 else self.model[:17] + "..."
            parts += f"  [dim]|  {m}[/dim]"

        return Text.from_markup(parts)
