"""CLI interface for Polybillionaire."""

from __future__ import annotations

import os
import sys

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .client import PolymarketClient
from .scanner import MarketScanner
from .trader import LiveTrader, PaperTrader

load_dotenv()
console = Console()

BANNER = """
 ____       _       ____  _ _ _ _                   _
|  _ \\ ___ | |_   _| __ )(_) | (_) ___  _ __   __ _(_)_ __ ___
| |_) / _ \\| | | | |  _ \\| | | | / _ \\| '_ \\ / _` | | '__/ _ \\
|  __/ (_) | | |_| | |_) | | | | | (_) | | | | (_| | | | |  __/
|_|   \\___/|_|\\__, |____/|_|_|_|_|\\___/|_| |_|\\__,_|_|_|  \\___|
              |___/
"""


def get_bankroll() -> float:
    return float(os.getenv("BANKROLL", "5.0"))


@click.group()
def main() -> None:
    """Polybillionaire — Polymarket trading toolkit."""
    pass


@main.command()
@click.option("--limit", "-n", default=20, help="Number of markets to scan")
@click.option("--min-volume", default=1000.0, help="Minimum 24h volume")
def scan(limit: int, min_volume: float) -> None:
    """Scan top markets for trading opportunities."""
    console.print(BANNER, style="bold cyan")
    console.print(f"[bold]Scanning top {limit} markets...[/bold]\n")

    with PolymarketClient() as client:
        scanner = MarketScanner(client, bankroll=get_bankroll())
        markets = scanner.scan_top_markets(limit=limit)

        if not markets:
            console.print("[red]No markets found.[/red]")
            return

        table = Table(title="Top Polymarket Markets", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Market", max_width=50)
        table.add_column("Volume", justify="right", style="green")
        table.add_column("Liquidity", justify="right", style="cyan")
        table.add_column("Outcomes", max_width=30)
        table.add_column("End Date", style="dim")

        for i, m in enumerate(markets[:limit], 1):
            outcomes = " / ".join(m.outcomes[:3]) if m.outcomes else "—"
            table.add_row(
                str(i),
                m.question[:50],
                f"${m.volume:,.0f}",
                f"${m.liquidity:,.0f}",
                outcomes,
                m.end_date[:10] if m.end_date else "—",
            )

        console.print(table)
        console.print(f"\n[dim]Found {len(markets)} markets. Use 'pb opportunities' for ranked picks.[/dim]")


@main.command()
@click.option("--limit", "-n", default=30, help="Markets to analyze")
@click.option("--top", "-t", default=10, help="Top opportunities to show")
@click.option("--min-volume", default=5000.0, help="Minimum volume filter")
def opportunities(limit: int, top: int, min_volume: float) -> None:
    """Find and rank trading opportunities."""
    console.print(BANNER, style="bold cyan")
    bankroll = get_bankroll()
    console.print(f"[bold]Finding opportunities (bankroll: ${bankroll:.2f})...[/bold]\n")

    with PolymarketClient() as client:
        scanner = MarketScanner(client, bankroll=bankroll)

        with console.status("[bold green]Scanning markets..."):
            markets = scanner.scan_top_markets(limit=limit)

        with console.status("[bold green]Analyzing opportunities..."):
            opps = scanner.find_opportunities(markets, min_volume=min_volume)

        if not opps:
            console.print("[yellow]No strong opportunities found right now.[/yellow]")
            return

        table = Table(title=f"Top {top} Opportunities", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Market", max_width=40)
        table.add_column("Bet", width=6)
        table.add_column("Price", justify="right")
        table.add_column("Payout", justify="right", style="bold yellow")
        table.add_column("24h Vol", justify="right", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Size", justify="right", style="green")
        table.add_column("Signals", max_width=45)

        for i, o in enumerate(opps[:top], 1):
            side_style = "green" if o.side == "YES" else "red"
            score_style = "bold green" if o.score >= 7 else "bold yellow" if o.score >= 5 else "dim"
            payout_style = "bold green" if o.payout_multiple >= 10 else "bold yellow" if o.payout_multiple >= 3 else ""

            table.add_row(
                str(i),
                o.market.question[:40],
                Text(o.side, style=side_style),
                f"${o.price:.3f}",
                Text(f"{o.payout_multiple:.0f}x", style=payout_style),
                f"${o.volume_24h:,.0f}",
                Text(f"{o.score:.1f}", style=score_style),
                f"${o.recommended_bet:.2f}",
                o.reason[:45],
            )

        console.print(table)

        total_recommended = sum(o.recommended_bet for o in opps[:top])
        console.print(f"\n[bold]Total recommended allocation: ${total_recommended:.2f} / ${bankroll:.2f}[/bold]")
        console.print(
            "\n[dim]Strategy: Long shots (low price, high payout) + Momentum plays (trending markets)[/dim]"
        )
        console.print("[dim]Use 'pb analyze <keyword>' for deep analysis.[/dim]")
        console.print("[dim]Use 'pb paper-buy <keyword>' to paper trade.[/dim]")


@main.command()
@click.argument("query")
def analyze(query: str) -> None:
    """Deep-analyze a specific market (search by keyword)."""
    with PolymarketClient() as client:
        scanner = MarketScanner(client, bankroll=get_bankroll())

        with console.status(f"[bold green]Searching for '{query}'..."):
            markets = client.search_markets(query, limit=5)

        if not markets:
            console.print(f"[red]No markets found for '{query}'.[/red]")
            return

        market = markets[0]
        console.print(Panel(
            f"[bold]{market.question}[/bold]\n\n{market.description[:200]}..."
            if len(market.description) > 200 else
            f"[bold]{market.question}[/bold]\n\n{market.description}",
            title="Market Analysis",
            border_style="cyan",
        ))

        with console.status("[bold green]Fetching orderbook data..."):
            analysis = scanner.analyze_market(market)

        table = Table(title="Token Analysis", show_lines=True)
        table.add_column("Outcome", style="bold")
        table.add_column("Price", justify="right")
        table.add_column("Bid", justify="right", style="green")
        table.add_column("Ask", justify="right", style="red")
        table.add_column("Spread", justify="right")
        table.add_column("Bid Depth", justify="right")
        table.add_column("Ask Depth", justify="right")
        table.add_column("Momentum", justify="right")
        table.add_column("Volatility", justify="right")

        for t in analysis["tokens"]:
            if "error" in t:
                table.add_row(t["outcome"], f"[red]Error: {t['error'][:30]}[/red]", *["—"] * 7)
                continue
            mom_style = "green" if t["momentum"] > 0 else "red" if t["momentum"] < 0 else "dim"
            table.add_row(
                t["outcome"],
                f"${t['midpoint']:.3f}",
                f"${t['best_bid']:.3f}",
                f"${t['best_ask']:.3f}",
                f"{t['spread_pct']:.1f}%",
                f"${t['bid_depth']:.0f}",
                f"${t['ask_depth']:.0f}",
                Text(f"{t['momentum']:+.4f}", style=mom_style),
                f"{t['volatility']:.4f}",
            )

        console.print(table)
        console.print(f"\n[dim]Volume: ${analysis['volume']:,.0f} | Liquidity: ${analysis['liquidity']:,.0f} | Ends: {analysis['end_date'][:10]}[/dim]")


@main.command("paper-buy")
@click.argument("query")
@click.option("--size", "-s", default=10.0, help="Number of shares to buy")
@click.option("--outcome", "-o", default="Yes", help="Outcome to buy (Yes/No)")
def paper_buy(query: str, size: float, outcome: str) -> None:
    """Paper-trade: buy shares in a market."""
    with PolymarketClient() as client:
        markets = client.search_markets(query, limit=3)
        if not markets:
            console.print(f"[red]No markets found for '{query}'.[/red]")
            return

        market = markets[0]
        token = None
        for t in market.tokens:
            if t.get("outcome", "").lower() == outcome.lower():
                token = t
                break
        if not token:
            token = market.tokens[0] if market.tokens else None
        if not token:
            console.print("[red]No tradeable tokens found.[/red]")
            return

        trader = PaperTrader(client, bankroll=get_bankroll())
        ok, msg = trader.buy(
            token_id=token["token_id"],
            market_question=market.question,
            outcome=token.get("outcome", "?"),
            size=size,
        )

        style = "green" if ok else "red"
        console.print(f"[{style}]{msg}[/{style}]")
        if ok:
            console.print(f"[dim]Bankroll remaining: ${trader.bankroll:.2f}[/dim]")


@main.command("paper-sell")
@click.argument("token_id")
@click.option("--size", "-s", default=None, type=float, help="Shares to sell (all if omitted)")
def paper_sell(token_id: str, size: float | None) -> None:
    """Paper-trade: sell a position."""
    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=get_bankroll())
        ok, msg = trader.sell(token_id, size)
        style = "green" if ok else "red"
        console.print(f"[{style}]{msg}[/{style}]")


@main.command()
def portfolio() -> None:
    """Show paper trading portfolio."""
    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=get_bankroll())
        alerts = trader.update_positions()

        console.print(Panel(
            f"[bold]Bankroll:[/bold] ${trader.bankroll:.2f}\n"
            f"[bold]Positions:[/bold] {len(trader.positions)}\n"
            f"[bold]Total Value:[/bold] ${trader.total_value:.2f}",
            title="Paper Portfolio",
            border_style="green",
        ))

        if trader.positions:
            table = Table(title="Open Positions", show_lines=True)
            table.add_column("Market", max_width=40)
            table.add_column("Outcome", width=8)
            table.add_column("Shares", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("PnL", justify="right")
            table.add_column("Token ID", style="dim", max_width=20)

            for p in trader.positions:
                pnl_style = "green" if p.pnl >= 0 else "red"
                table.add_row(
                    p.market_question[:40],
                    p.outcome,
                    f"{p.size:.1f}",
                    f"${p.entry_price:.3f}",
                    f"${p.current_price:.3f}",
                    Text(f"${p.pnl:+.2f}", style=pnl_style),
                    p.token_id[:20] + "...",
                )

            console.print(table)

        if alerts:
            for a in alerts:
                console.print(f"[bold red]{a['message']}[/bold red]")

        if trader.trades:
            console.print(f"\n[dim]Total trades: {len(trader.trades)}[/dim]")


@main.command()
@click.argument("query")
def search(query: str) -> None:
    """Search for markets by keyword."""
    with PolymarketClient() as client:
        with console.status(f"[bold green]Searching '{query}'..."):
            markets = client.search_markets(query)

        if not markets:
            console.print(f"[yellow]No results for '{query}'.[/yellow]")
            return

        table = Table(title=f"Search: {query}", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Market", max_width=55)
        table.add_column("Volume", justify="right", style="green")
        table.add_column("Active", width=6)
        table.add_column("Slug", style="dim", max_width=25)

        for i, m in enumerate(markets, 1):
            table.add_row(
                str(i),
                m.question[:55],
                f"${m.volume:,.0f}",
                "[green]Yes[/green]" if m.active else "[red]No[/red]",
                m.slug[:25],
            )

        console.print(table)


@main.command()
@click.option("--hours", "-h", default=24, help="Hours until resolution (default 24)")
def daily(hours: int) -> None:
    """Show markets resolving within the next N hours."""
    console.print(BANNER, style="bold cyan")
    console.print(f"[bold]Markets resolving within {hours}h...[/bold]\n")

    with PolymarketClient() as client:
        scanner = MarketScanner(client, bankroll=get_bankroll())
        markets = client.get_daily_markets(hours=hours)

        if not markets:
            console.print(f"[yellow]No markets resolving within {hours}h. Trying 48h...[/yellow]")
            markets = client.get_daily_markets(hours=48)
            if not markets:
                console.print("[red]No near-term markets found.[/red]")
                return

        opps = scanner.find_opportunities(markets)

        table = Table(title=f"Daily Markets ({len(markets)} found, {len(opps)} tradeable)", show_lines=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Market", max_width=50)
        table.add_column("Ends", style="dim", width=16)
        table.add_column("Side", width=5)
        table.add_column("Price", justify="right")
        table.add_column("Payout", justify="right", style="bold yellow")
        table.add_column("Vol 24h", justify="right", style="cyan")
        table.add_column("Score", justify="right")

        if opps:
            for i, o in enumerate(opps[:20], 1):
                side_style = "green" if o.side == "YES" else "red"
                ends = o.market.end_date[:16] if o.market.end_date else "—"
                table.add_row(
                    str(i),
                    o.market.question[:50],
                    ends,
                    Text(o.side, style=side_style),
                    f"${o.price:.3f}",
                    f"{o.payout_multiple:.0f}x",
                    f"${o.volume_24h:,.0f}",
                    f"{o.score:.1f}",
                )
        else:
            for i, m in enumerate(markets[:20], 1):
                ends = m.end_date[:16] if m.end_date else "—"
                yes_price = m.outcome_prices[0] if m.outcome_prices else 0
                table.add_row(
                    str(i), m.question[:50], ends, "—",
                    f"${yes_price:.3f}", "—", f"${m.volume_24h:,.0f}", "—",
                )

        console.print(table)
        console.print(f"\n[dim]Use 'pb org --daily' to run the full trading org on daily markets.[/dim]")


@main.command()
def status() -> None:
    """Check Polymarket API status and your config."""
    console.print(BANNER, style="bold cyan")

    with PolymarketClient() as client:
        try:
            client._get("https://clob.polymarket.com", "/ok")
            console.print("[green]CLOB API: Online[/green]")
        except Exception as e:
            console.print(f"[red]CLOB API: Error — {e}[/red]")

    bankroll = get_bankroll()
    has_key = bool(os.getenv("POLY_PRIVATE_KEY"))
    has_api = bool(os.getenv("POLY_API_KEY"))
    console.print(f"\nBankroll: ${bankroll:.2f}")
    console.print(f"Private key: {'[green]Configured[/green]' if has_key else '[yellow]Not set[/yellow]'}")
    console.print(f"API creds: {'[green]Configured[/green]' if has_api else '[yellow]Not set[/yellow]'}")
    console.print(f"Live trading: {'[bold green]READY[/bold green]' if (has_key and has_api) else '[yellow]Paper only[/yellow]'}")
    console.print(f"Max bet: ${bankroll * float(os.getenv('MAX_BET_FRACTION', '0.10')):.2f}")
    console.print(f"Stop loss: {float(os.getenv('STOP_LOSS_PCT', '0.50'))*100:.0f}%")

    if has_key and has_api:
        try:
            trader = LiveTrader.from_env()
            bal = trader.get_balance()
            console.print(f"On-chain balance: {bal}")
        except Exception as e:
            console.print(f"[red]Connection error: {e}[/red]")


# ── Live Trading Commands ────────────────────────────────────────


def _get_live_trader() -> LiveTrader:
    """Create a LiveTrader from env, or fail with a helpful message."""
    if not os.getenv("POLY_PRIVATE_KEY") or not os.getenv("POLY_API_KEY"):
        console.print("[red]Live trading requires POLY_PRIVATE_KEY and POLY_API_KEY in .env[/red]")
        raise SystemExit(1)
    return LiveTrader.from_env()


@main.command("buy")
@click.argument("query")
@click.option("--amount", "-a", default=0.50, help="Dollar amount to spend")
@click.option("--outcome", "-o", default="Yes", help="Outcome to buy (Yes/No)")
@click.option("--limit-price", "-p", default=None, type=float, help="Limit price (omit for market order)")
@click.confirmation_option(prompt="Place REAL order with REAL money?")
def live_buy(query: str, amount: float, outcome: str, limit_price: float | None) -> None:
    """Buy shares with real money."""
    with PolymarketClient() as client:
        markets = client.search_markets(query, limit=3)
        if not markets:
            console.print(f"[red]No markets found for '{query}'.[/red]")
            return

        market = markets[0]
        token = None
        for t in market.tokens:
            if t.get("outcome", "").lower() == outcome.lower():
                token = t
                break
        if not token:
            token = market.tokens[0] if market.tokens else None
        if not token:
            console.print("[red]No tradeable tokens found.[/red]")
            return

        console.print(f"[bold]Market:[/bold] {market.question}")
        console.print(f"[bold]Outcome:[/bold] {token.get('outcome', '?')}")
        console.print(f"[bold]Amount:[/bold] ${amount:.2f}")

        trader = _get_live_trader()

        if limit_price is not None:
            size = amount / limit_price
            console.print(f"[bold]Limit price:[/bold] ${limit_price:.3f} ({size:.1f} shares)")
            result = trader.place_limit_order(
                token_id=token["token_id"],
                price=limit_price,
                size=size,
            )
        else:
            console.print("[bold]Order type:[/bold] Market (best available price)")
            result = trader.place_market_buy(
                token_id=token["token_id"],
                amount=amount,
            )

        if "error" in result:
            console.print(f"[red]Order failed: {result['error']}[/red]")
        else:
            console.print(f"[bold green]Order placed![/bold green]")
            console.print(f"Result: {result}")


@main.command("orders")
def live_orders() -> None:
    """Show open orders."""
    trader = _get_live_trader()
    orders = trader.get_open_orders()
    if not orders:
        console.print("[dim]No open orders.[/dim]")
        return
    for o in orders:
        console.print(o)


@main.command("cancel")
@click.argument("order_id", default="all")
def live_cancel(order_id: str) -> None:
    """Cancel an order (or 'all' to cancel everything)."""
    trader = _get_live_trader()
    if order_id == "all":
        result = trader.cancel_all()
    else:
        result = trader.cancel_order(order_id)
    console.print(result)


@main.command("balance")
def live_balance() -> None:
    """Check on-chain USDC balance."""
    trader = _get_live_trader()
    result = trader.get_balance()
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        console.print(Panel(f"[bold]{result}[/bold]", title="On-Chain Balance", border_style="green"))


@main.command("positions")
def positions() -> None:
    """Show all open paper positions."""
    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=get_bankroll())
        if not trader.positions:
            console.print("[dim]No open positions.[/dim]")
            return

        trader.update_positions()

        table = Table(title="Open Positions", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Market", max_width=45)
        table.add_column("Side", width=5)
        table.add_column("Entry", justify="right", width=8)
        table.add_column("Now", justify="right", width=8)
        table.add_column("Size", justify="right", width=7)
        table.add_column("P&L", justify="right", width=10)
        table.add_column("Token ID", style="dim", max_width=12)

        for i, p in enumerate(trader.positions, 1):
            pc = "green" if p.pnl >= 0 else "red"
            table.add_row(
                str(i),
                p.market_question[:45],
                p.outcome,
                f"${p.entry_price:.4f}",
                f"${p.current_price:.4f}",
                f"{p.size:.1f}",
                f"[{pc}]${p.pnl:+.4f}[/{pc}]",
                p.token_id[:12] + "...",
            )

        console.print(table)
        total_pnl = sum(p.pnl for p in trader.positions)
        pc = "green" if total_pnl >= 0 else "red"
        console.print(
            f"\n  Bankroll: ${trader.bankroll:.2f}  |  "
            f"Positions: {len(trader.positions)}  |  "
            f"Total P&L: [{pc}]${total_pnl:+.4f}[/{pc}]"
        )
        console.print(f"\n[dim]Use 'pb sell <number>' to close a position.[/dim]")


@main.command("sell")
@click.argument("position_num", type=int)
def sell_position(position_num: int) -> None:
    """Sell/close a paper position by its number (from 'pb positions')."""
    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=get_bankroll())
        if not trader.positions:
            console.print("[dim]No open positions.[/dim]")
            return

        if position_num < 1 or position_num > len(trader.positions):
            console.print(f"[red]Invalid position #{position_num}. "
                          f"Valid range: 1-{len(trader.positions)}[/red]")
            return

        pos = trader.positions[position_num - 1]
        console.print(f"[bold]Closing:[/bold] {pos.outcome} \"{pos.market_question[:50]}\"")
        console.print(f"  Entry: ${pos.entry_price:.4f}  |  Size: {pos.size:.1f}")

        ok, msg = trader.sell(pos.token_id)
        if ok:
            console.print(f"[bold green]{msg}[/bold green]")
        else:
            console.print(f"[red]{msg}[/red]")


@main.command("org")
@click.option("--cycles", "-c", default=None, type=int, help="Number of cycles (infinite if omitted)")
@click.option("--interval", "-i", default=300, help="Seconds between cycles (default: 300 pace mode)")
@click.option("--proposals", "-p", default=5, help="Max trade proposals per cycle")
@click.option("--scan-limit", "-n", default=30, help="Markets to scan per cycle")
@click.option(
    "--model", "-m",
    default="sonnet",
    help="Claude model (sonnet, opus, haiku)",
)
@click.option("--fresh", is_flag=True, help="Clear agent sessions (keeps DB/memory)")
@click.option("--reset-db", is_flag=True, help="Also wipe institutional memory DB (use with --fresh)")
@click.option("--simple", is_flag=True, help="Use simple stream output instead of TUI dashboard")
@click.option("--daily/--no-daily", default=True, help="Trade short-term markets (default: on)")
def org(
    cycles: int | None,
    interval: int,
    proposals: int,
    scan_limit: int,
    model: str,
    fresh: bool,
    reset_db: bool,
    simple: bool,
    daily: bool,
) -> None:
    """Run the autonomous zero-human trading organization (paper mode).

    Each agent is a Claude Code session — uses your existing auth,
    no API key needed.  Sessions persist across runs.

    By default shows a full-screen TUI dashboard.  Use --simple for
    plain sequential output (good for piping or logging).
    Use --fresh to wipe agent sessions (new prompts take effect).
    Add --reset-db to also wipe the institutional memory database.
    """
    from .org.runner import Organization

    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=get_bankroll())
        organization = Organization(
            client,
            trader,
            model=model,
            max_proposals=proposals,
            scan_limit=scan_limit,
            simple=simple,
            daily_only=daily,
        )
        if fresh:
            organization.clear_sessions(reset_db=reset_db)
        organization.run(cycles=cycles, interval=interval)


if __name__ == "__main__":
    main()
