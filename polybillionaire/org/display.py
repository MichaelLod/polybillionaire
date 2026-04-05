"""Simple stream-based display (--simple mode) for the trading organisation.

Prints messages sequentially to stdout.  Used as a fallback when the
full TUI dashboard is not wanted (piping, CI, narrow terminals, etc.).
"""

from __future__ import annotations

import time

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from .bus import Message

AGENT_STYLES = {
    "CEO": "bold magenta",
    "Research": "bold cyan",
    "Reasoning": "bold yellow",
    "Trader": "bold green",
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


class Display:
    """Sequential message printer — no Live, no alternate screen."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def banner(self) -> None:
        self.console.print()
        self.console.print(
            Panel(
                "[bold]POLYBILLIONAIRE TRADING ORG[/bold]\n"
                "[dim]Zero-Human Autonomous Trading[/dim]",
                border_style="cyan",
                padding=(1, 4),
            )
        )

    def cycle_start(self, n: int) -> None:
        self.console.print()
        self.console.print(Rule(f" Cycle {n} ", style="bold cyan"))
        self.console.print()

    def render(self, msg: Message) -> None:
        style = AGENT_STYLES.get(msg.sender, "white")
        label, label_style = KIND_LABELS.get(msg.kind, (msg.kind.upper(), "white"))

        target = f" -> {msg.to}" if msg.to != "all" else ""

        header = Text()
        header.append(f"[{label}] ", style=label_style)
        header.append(msg.sender, style=style)
        header.append(target, style="dim")

        self.console.print(header)
        for line in msg.content.strip().split("\n"):
            self.console.print(f"  {line}")
        self.console.print()

    def tool_call(self, agent_name: str, tool_name: str) -> None:
        style = AGENT_STYLES.get(agent_name, "white")
        self.console.print(
            f"  [{style}]{agent_name}[/{style}] [dim italic]using {tool_name}...[/dim italic]"
        )

    def agent_thinking(self, agent_name: str) -> None:
        style = AGENT_STYLES.get(agent_name, "white")
        self.console.print(f"[{style}]{agent_name}[/{style}] [dim]thinking...[/dim]")

    def agent_done(self, name: str, summary: str = "") -> None:
        pass

    def update_portfolio(self, **kw: float) -> None:
        pass

    def update_tokens(self, name: str, input_tokens: int, output_tokens: int) -> None:
        pass

    def get_pending_hints(self) -> list[str]:
        return []

    def mark_hints_seen(self) -> None:
        pass

    def wait(self, seconds: int, reason: str = "") -> None:
        tag = f" ({reason})" if reason else ""
        self.console.print(f"[dim]--- Next cycle in {seconds}s{tag} ---[/dim]\n")
        time.sleep(seconds)

    def shutdown(self, summary: str) -> None:
        self.console.print()
        self.console.print(Rule(" Shutdown ", style="bold red"))
        self.console.print(summary)
        self.console.print()
