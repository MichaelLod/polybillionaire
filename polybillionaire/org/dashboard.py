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
            "CEO", *self._research_names, "Reasoning", "Trader",
        ]
        self.agents: dict[str, AgentStatus] = {
            n: AgentStatus() for n in self._all_agent_names
        }
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

        # View mode: "feed" | "positions" | "hypotheses"
        self._view_mode = "feed"
        self._selected_index = 0

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

        # Key mapping: number keys -> agent names
        self._key_to_agent: dict[str, str] = {}
        self._agent_to_key: dict[str, str] = {}
        for i, name in enumerate(self._all_agent_names):
            key = str(i + 1)
            self._key_to_agent[key] = name
            self._agent_to_key[name] = key

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

                # "i" enters hint input mode
                if ch == "i":
                    self._input_mode = True
                    self._input_buffer = ""
                    continue

                # View mode switching
                if ch in ("f", "p", "h"):
                    self._open_windows.clear()
                    self._window_scroll.clear()
                    self._view_mode = {
                        "f": "feed", "p": "positions", "h": "hypotheses",
                    }[ch]
                    self._selected_index = 0
                    continue

                # j/k navigation within list views
                if ch == "j" and self._view_mode in ("positions", "hypotheses"):
                    self._selected_index += 1
                    continue
                if ch == "k" and self._view_mode in ("positions", "hypotheses"):
                    self._selected_index = max(0, self._selected_index - 1)
                    continue

                # Number keys toggle agent windows
                if ch in self._key_to_agent:
                    name = self._key_to_agent[ch]
                    self._view_mode = "feed"
                    if name in self._open_windows:
                        self._open_windows.remove(name)
                        self._window_scroll.pop(name, None)
                    else:
                        self._open_windows.append(name)
                        self._window_scroll[name] = 999999
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
                        # Standalone Escape — back to feed
                        self._open_windows.clear()
                        self._window_scroll.clear()
                        self._view_mode = "feed"
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

    def _ensure_agent(self, name: str) -> None:
        if name not in self.agents:
            self.agents[name] = AgentStatus()

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
        if self._sell_result_ttl > 0:
            self._sell_result_ttl -= 1
            if self._sell_result_ttl <= 0:
                self._sell_result = None

    # ── Layout ──────────────────────────────────────────────────

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="main", ratio=3),
            Layout(name="sidebar", size=34),
        )

        org_height = max(8, len(self._research_names) + 5)

        main_parts: list[Layout] = [
            Layout(name="header", size=3),
            Layout(name="org", size=org_height),
            Layout(name="status", size=1),
        ]

        if self._open_windows:
            for wname in self._open_windows:
                main_parts.append(Layout(name=f"win_{wname}", ratio=1))
        elif self._view_mode == "positions":
            main_parts.append(Layout(name="positions", ratio=1, minimum_size=10))
        elif self._view_mode == "hypotheses":
            main_parts.append(Layout(name="hypotheses", ratio=1, minimum_size=10))
        else:
            # Default: feed + live agent output side by side
            feed_row = Layout(name="feed_row", ratio=1, minimum_size=8)
            main_parts.append(feed_row)

        main_parts.append(Layout(name="footer", size=1))
        layout["main"].split_column(*main_parts)

        layout["sidebar"].split_column(
            Layout(name="memory", ratio=3),
            Layout(name="hints", ratio=2, minimum_size=8),
        )

        # Populate
        layout["header"].update(self._hdr())
        layout["org"].update(self._org())
        layout["status"].update(self._status_line())

        if self._open_windows:
            for wname in self._open_windows:
                layout[f"win_{wname}"].update(self._agent_window(wname))
        elif self._view_mode == "positions":
            layout["positions"].update(self._positions_view())
        elif self._view_mode == "hypotheses":
            layout["hypotheses"].update(self._hypotheses_view())
        else:
            # Split feed row: activity feed + live agent output
            active_agent = self._most_active_agent()
            if active_agent:
                layout["feed_row"].split_row(
                    Layout(name="feed", ratio=1),
                    Layout(name="live_agent", ratio=1),
                )
                layout["feed"].update(self._feed())
                layout["live_agent"].update(self._live_agent_panel(active_agent))
            else:
                layout["feed_row"].update(self._feed())

        layout["footer"].update(self._ftr())
        layout["memory"].update(self._memory_panel())
        layout["hints"].update(self._hints_panel())

        return layout

    # ── Header ──────────────────────────────────────────────────

    def _hdr(self) -> Panel:
        c = self.cycle
        total = f"/{self.total_cycles}" if self.total_cycles else ""
        p = self.portfolio
        ps = "green" if p["pnl"] >= 0 else "red"
        total_tok = sum(
            a.input_tokens + a.output_tokens for a in self.agents.values()
        )
        tok_part = f"  |  Tokens [dim]{total_tok:,}[/dim]" if total_tok else ""
        mode_badge = (
            "  [bold red]POWER[/bold red]" if self.power_mode
            else ""
        )
        return Panel(
            Text.from_markup(
                f"[bold cyan]POLYBILLIONAIRE TRADING ORG[/bold cyan]{mode_badge}   "
                f"Cycle [bold]{c}{total}[/bold]  |  "
                f"Bankroll [bold]${p['bankroll']:.2f}[/bold]  |  "
                f"Positions [bold]{p['positions']}[/bold]  |  "
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

    # ── Org chart ───────────────────────────────────────────────

    def _org(self) -> Panel:
        table = Table(
            show_header=False, show_edge=False, box=None,
            padding=0, expand=True,
        )
        table.add_column(ratio=2, vertical="middle")
        table.add_column(ratio=1, vertical="middle", justify="center")
        table.add_column(ratio=3, vertical="middle")
        table.add_column(ratio=1, vertical="middle", justify="center")
        table.add_column(ratio=2, vertical="middle")
        table.add_column(ratio=1, vertical="middle", justify="center")
        table.add_column(ratio=2, vertical="middle")

        table.add_row(
            self._card("CEO"),
            self._arrow("CEO", "Research"),
            self._research_pool(),
            self._arrow("Research", "Reasoning"),
            self._card("Reasoning"),
            self._arrow("Reasoning", "Trader"),
            self._card("Trader"),
        )
        return Panel(
            table,
            title="[bold]Organization[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _card(self, name: str) -> Panel:
        st = self.agents.get(name, AgentStatus())
        color = _color(name)
        dot = STATUS_DOT.get(st.status, STATUS_DOT["idle"])
        act = (st.activity or st.status)[:12]
        tok = st.input_tokens + st.output_tokens
        tok_str = f"{tok:,}" if tok else "0"

        key = self._agent_to_key.get(name, "")
        is_open = name in self._open_windows
        key_style = "bold green" if is_open else "dim"
        key_label = f" [{key_style}][{key}][/{key_style}]" if key else ""

        snippet = self._snippet(name, 28)

        active = st.status in ("thinking", "working")
        box = DOUBLE if active else ROUNDED
        if active:
            bstyle = f"bold {color}"
        elif is_open:
            bstyle = color
        else:
            bstyle = "dim"
        if st.status == "thinking" and self._frame % 4 < 2:
            bstyle = color

        content = (
            f"{dot} [bold]{name}[/bold]{key_label}\n"
            f"[dim]{act}[/dim]  [dim]{tok_str} tok[/dim]"
        )
        if snippet:
            content += f"\n[dim]\"{snippet}\"[/dim]"

        return Panel(content, box=box, border_style=bstyle, height=5)

    def _research_pool(self) -> Panel:
        cards: list[str] = []
        for i, rname in enumerate(self._research_names):
            st = self.agents.get(rname, AgentStatus())
            dot = STATUS_DOT.get(st.status, STATUS_DOT["idle"])
            act = (st.activity or st.status)[:10]
            tok = st.input_tokens + st.output_tokens
            tok_str = f"{tok:,}" if tok else "0"
            key = self._agent_to_key.get(rname, "")
            is_open = rname in self._open_windows
            ks = "bold green" if is_open else "dim"

            snip = self._snippet(rname, 25)
            snip_part = f" [dim]\"{snip}\"[/dim]" if snip else ""

            cards.append(
                f"{dot} R{i + 1} [dim]{act}[/dim]  "
                f"[dim]{tok_str}[/dim]  [{ks}][{key}][/{ks}]"
                f"{snip_part}"
            )

        any_active = any(
            self.agents.get(n, AgentStatus()).status in ("thinking", "working")
            for n in self._research_names
        )
        any_open = any(n in self._open_windows for n in self._research_names)
        border = "bold cyan" if any_active else ("cyan" if any_open else "dim")
        box = DOUBLE if any_active else ROUNDED

        return Panel(
            "\n".join(cards),
            box=box,
            border_style=border,
            title="[dim]Research Pool[/dim]",
            height=len(self._research_names) + 2,
        )

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

    def _arrow(self, from_name: str, to_name: str) -> Text:
        key = f"{from_name}->{to_name}"
        ttl = self._active_flows.get(key, 0)
        color = _color(from_name)

        if ttl <= 0:
            return Text.from_markup("[dim]\u00b7 \u00b7 \u00b7 \u00b7[/dim]")

        width = 5
        pos = self._frame % (width + 1)
        parts: list[str] = []
        for i in range(width):
            if i == pos:
                parts.append(f"[bold {color}]\u25cf[/bold {color}]")
            elif abs(i - pos) == 1 and pos < width:
                parts.append(f"[{color}]\u25cb[/{color}]")
            else:
                parts.append(f"[dim {color}]\u2500[/dim {color}]")

        tip = (
            f"[bold {color}]\u25b8[/bold {color}]"
            if pos >= width - 1
            else "[dim]\u25b8[/dim]"
        )
        parts.append(tip)
        return Text.from_markup("".join(parts))

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
        mode_labels = {"feed": "feed", "positions": "positions", "hypotheses": "hypotheses"}
        current = mode_labels.get(self._view_mode, "feed")

        open_keys = [
            self._agent_to_key[n]
            for n in self._open_windows
            if n in self._agent_to_key
        ]
        if open_keys:
            current = f"agent {''.join(open_keys)}"

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
            f"[bold]1[/bold]-[bold]6[/bold] agents  "
            f"[bold]i[/bold] hint  "
            f"[bold]g[/bold] {power_label}  "
            f"Esc back[/dim]"
        )

        if self._footer_extra:
            parts += f"  [dim]|  {self._footer_extra}[/dim]"
        if self.model:
            m = self.model if len(self.model) <= 20 else self.model[:17] + "..."
            parts += f"  [dim]|  {m}[/dim]"

        return Text.from_markup(parts)
