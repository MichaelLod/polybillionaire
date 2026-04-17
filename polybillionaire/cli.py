"""CLI for the short-duration crypto trading bot."""

from __future__ import annotations

import os

import click
import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .binance_exec import BinanceFutures
from .client import PolymarketClient
from .crossvenue import scan_opportunities, scan_opportunities_llm
from .futures import USDC_SYMBOLS, FuturesBot, FuturesConfig
from .gamma import fetch_all_open_markets, fetch_updown_markets
from .hourly import HourlyBot, HourlyConfig
from .trader import LiveTrader, PaperTrader

load_dotenv()
console = Console()


def _bankroll() -> float:
    return float(os.getenv("BANKROLL", "5.0"))


def _get_live_trader() -> LiveTrader:
    if not os.getenv("POLY_PRIVATE_KEY") or not os.getenv("POLY_API_KEY"):
        console.print(
            "[red]Live mode needs POLY_PRIVATE_KEY + POLY_API_KEY "
            "(+ _SECRET, _PASSPHRASE) in .env[/red]"
        )
        raise SystemExit(1)
    return LiveTrader.from_env()


@click.group()
def main() -> None:
    """Polybillionaire — short-duration crypto trading on Polymarket."""


@main.command()
@click.option("--live", is_flag=True, help="Trade real USDC (default: paper)")
@click.option(
    "--edge", default=0.07, show_default=True,
    help="Minimum |p_model - book_mid| to open. 0.07 covers Polymarket's "
         "post-2024 dynamic taker fee (~3.15%) + slippage + model error.",
)
@click.option("--dry-run", is_flag=True, help="Log decisions, don't place orders")
@click.option(
    "--horizon", default=3600, show_default=True, type=int,
    help="Max seconds until market end to consider",
)
@click.option("--trade-15m", is_flag=True, help="Also trade 15-minute markets")
@click.option("--trade-5m", is_flag=True, help="Also trade 5-minute markets")
@click.option(
    "--cycle", default=30.0, show_default=True, type=float,
    help="Seconds between cycles",
)
@click.option(
    "--kelly", default=0.5, show_default=True, type=float,
    help="Kelly fraction (0.5 = half-Kelly)",
)
def hourly(
    live: bool,
    edge: float,
    dry_run: bool,
    horizon: int,
    trade_15m: bool,
    trade_5m: bool,
    cycle: float,
    kelly: float,
) -> None:
    """Run the trading loop on up-or-down crypto markets."""
    config = HourlyConfig(
        edge_threshold=edge,
        max_seconds_until_end=horizon,
        trade_15m=trade_15m,
        trade_5m=trade_5m,
        cycle_s=cycle,
        kelly_fraction=kelly,
        dry_run=dry_run,
    )

    with PolymarketClient() as client:
        if live:
            console.print("[bold red]LIVE TRADING — real USDC at risk[/bold red]")
            trader: LiveTrader | PaperTrader = _get_live_trader()
            trader.set_client(client)
            bal = trader.get_balance()
            if "usdc" in bal:
                trader.risk.bankroll = bal["usdc"]
                console.print(f"[dim]USDC balance: ${bal['usdc']:.4f}[/dim]")
            recovered = trader.recover_positions()
            if recovered:
                console.print(f"[yellow]Recovered {recovered} positions[/yellow]")
        else:
            trader = PaperTrader(client, bankroll=_bankroll())
            console.print(
                f"[dim]Paper mode — bankroll ${trader.bankroll:.2f}[/dim]"
            )

        bot = HourlyBot(client, trader, config, print_fn=console.print)
        bot.run()


@main.command()
@click.option(
    "--horizon", default=3600, show_default=True, type=int,
    help="Max seconds until end",
)
def discover(horizon: int) -> None:
    """List active up-or-down crypto markets Gamma sees right now."""
    with httpx.Client(timeout=15.0) as http:
        markets = fetch_updown_markets(
            http, max_seconds_until_end=horizon, min_seconds_until_end=0,
        )

    if not markets:
        console.print("[yellow]No active up-or-down crypto markets.[/yellow]")
        return

    table = Table(title=f"{len(markets)} crypto up-or-down markets", show_lines=True)
    table.add_column("Symbol")
    table.add_column("Dur", width=5)
    table.add_column("Ends in", justify="right")
    table.add_column("p(Up)", justify="right")
    table.add_column("Vol", justify="right", style="cyan")
    table.add_column("Slug", style="dim", max_width=40)

    for m in sorted(markets, key=lambda m: m.seconds_until_end):
        mins = m.seconds_until_end / 60
        table.add_row(
            m.symbol.replace("USDT", ""),
            m.duration_label,
            f"{mins:.1f}m",
            f"{m.up_price:.3f}",
            f"${m.volume:,.0f}",
            m.slug[:40],
        )
    console.print(table)


@main.command()
def positions() -> None:
    """Show open paper positions and P&L."""
    with PolymarketClient() as client:
        trader = PaperTrader(client, bankroll=_bankroll())
        trader.update_positions(auto_stop=False)

        console.print(Panel(
            f"Bankroll: ${trader.bankroll:.4f}\n"
            f"Positions: {len(trader.positions)}\n"
            f"Total value: ${trader.total_value:.4f}",
            title="Paper Portfolio", border_style="green",
        ))

        if not trader.positions:
            return

        table = Table(title="Open Positions", show_lines=True)
        table.add_column("Market", max_width=50)
        table.add_column("Side", width=6)
        table.add_column("Entry", justify="right")
        table.add_column("Now", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("P&L", justify="right")
        for p in trader.positions:
            style = "green" if p.pnl >= 0 else "red"
            table.add_row(
                p.market_question[:50],
                p.outcome,
                f"${p.entry_price:.4f}",
                f"${p.current_price:.4f}",
                f"{p.size:.1f}",
                f"[{style}]${p.pnl:+.4f}[/{style}]",
            )
        console.print(table)


@main.command()
def balance() -> None:
    """Show live USDC balance on Polymarket."""
    trader = _get_live_trader()
    result = trader.get_balance()
    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
    else:
        console.print(Panel(
            f"USDC: ${result['usdc']:.4f}",
            title="On-chain", border_style="green",
        ))


@main.command()
def status() -> None:
    """Show config + API reachability."""
    has_priv = bool(os.getenv("POLY_PRIVATE_KEY"))
    has_api = bool(os.getenv("POLY_API_KEY"))

    with PolymarketClient() as client:
        try:
            client._get("https://clob.polymarket.com", "/ok")
            clob_ok = True
        except Exception:
            clob_ok = False

    with httpx.Client(timeout=5.0) as http:
        try:
            r = http.get("https://api.binance.com/api/v3/ping")
            binance_ok = r.status_code == 200
        except Exception:
            binance_ok = False

    bankroll = _bankroll()
    max_bet_frac = float(os.getenv("MAX_BET_FRACTION", "0.10"))

    lines = [
        f"Polymarket CLOB: {'[green]up[/green]' if clob_ok else '[red]down[/red]'}",
        f"Binance:         {'[green]up[/green]' if binance_ok else '[red]down[/red]'}",
        f"Bankroll:        ${bankroll:.2f}",
        f"Max bet:         ${bankroll * max_bet_frac:.2f}  ({max_bet_frac:.0%})",
        f"Live ready:      {'[green]yes[/green]' if (has_priv and has_api) else '[yellow]paper only[/yellow]'}",
    ]
    console.print(Panel("\n".join(lines), title="Status", border_style="cyan"))


@main.command()
@click.option("--live", is_flag=True, help="Place real orders (default: dry-run)")
@click.option(
    "--leverage", default=5, show_default=True, type=int,
    help="Leverage multiplier per trade (Binance futures: 1-125)",
)
@click.option(
    "--stop-pct", default=0.007, show_default=True, type=float,
    help="Stop-loss distance from entry (0.007 = 0.7%)",
)
@click.option(
    "--edge", default=0.08, show_default=True, type=float,
    help="Minimum |p_up - 0.5| to open (0.08 = trade when p_up>0.58 or <0.42)",
)
@click.option(
    "--margin-frac", default=0.20, show_default=True, type=float,
    help="Fraction of available USDC used as margin per trade",
)
@click.option(
    "--max-margin", default=25.0, show_default=True, type=float,
    help="Hard cap on USD margin per trade",
)
@click.option(
    "--symbols", default="BTC,ETH,SOL", show_default=True,
    help="Comma-separated coin tickers (resolved to *USDC pairs)",
)
@click.option(
    "--cycle", default=20.0, show_default=True, type=float,
    help="Seconds between cycles",
)
@click.option(
    "--take-profit", default=0.02, show_default=True, type=float,
    help="Lock gains in last 10 min if favorable move ≥ this (0 disables)",
)
def futures(
    live: bool,
    leverage: int,
    stop_pct: float,
    edge: float,
    margin_frac: float,
    max_margin: float,
    symbols: str,
    cycle: float,
    take_profit: float,
) -> None:
    """Run leveraged futures trading on Binance USDM.

    Dry-run by default. Use --live to place real orders.
    """
    coin_list = [c.strip().upper() for c in symbols.split(",") if c.strip()]
    pair_list: list[str] = []
    for c in coin_list:
        pair = USDC_SYMBOLS.get(c)
        if pair is None:
            console.print(f"[red]Unknown coin '{c}'. Known: {list(USDC_SYMBOLS)}[/red]")
            raise SystemExit(1)
        pair_list.append(pair)

    if leverage < 1 or leverage > 125:
        console.print("[red]Leverage must be 1-125[/red]")
        raise SystemExit(1)

    cfg = FuturesConfig(
        symbols=pair_list,
        leverage=leverage,
        stop_pct=stop_pct,
        edge_threshold=edge,
        margin_fraction_per_trade=margin_frac,
        max_margin_per_trade_usd=max_margin,
        cycle_s=cycle,
        take_profit_pct=take_profit,
        dry_run=not live,
    )

    try:
        exec = BinanceFutures.from_env()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    if live:
        console.print("[bold red]LIVE FUTURES TRADING — real USDC at risk[/bold red]")
    bot = FuturesBot(exec, cfg, print_fn=console.print)
    bot.run()


@main.command("futures-status")
def futures_status() -> None:
    """Show Binance futures wallet state and open positions."""
    try:
        exec = BinanceFutures.from_env()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    try:
        balances = exec._get("/fapi/v2/balance")  # type: ignore
        acct = exec._get("/fapi/v2/account")  # type: ignore
    except Exception as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    usdc = next((a for a in balances if a.get("asset") == "USDC"), {})
    usdt = next((a for a in balances if a.get("asset") == "USDT"), {})
    lines = [
        f"canTrade:          {acct.get('canTrade')}",
        f"totalWalletUSDT:   ${float(acct.get('totalWalletBalance') or 0):.4f}",
        f"USDC available:    ${float(usdc.get('availableBalance') or 0):.4f}",
        f"USDT available:    ${float(usdt.get('availableBalance') or 0):.4f}",
    ]
    console.print(Panel("\n".join(lines), title="Binance Futures", border_style="cyan"))

    positions = exec.get_positions()
    if not positions:
        console.print("[dim]No open positions.[/dim]")
        return
    table = Table(title="Open Positions", show_lines=True)
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Mark", justify="right")
    table.add_column("uPnL", justify="right")
    table.add_column("Lev", justify="right")
    for p in positions:
        style = "green" if p.unrealized_pnl >= 0 else "red"
        table.add_row(
            p.symbol, p.side, f"{p.qty}",
            f"${p.entry_price:.4f}", f"${p.mark_price:.4f}",
            f"[{style}]${p.unrealized_pnl:+.4f}[/{style}]",
            f"{p.leverage}×",
        )
    console.print(table)


@main.command("kalshi")
@click.option(
    "--status", default="open", show_default=True,
    type=click.Choice(["unopened", "open", "closed", "settled"]),
)
@click.option(
    "--limit", default=20, show_default=True, type=int,
    help="How many markets to list (max 1000)",
)
@click.option("--series", default=None, help="Filter by series ticker (e.g. KXPRES)")
def kalshi_cmd(status: str, limit: int, series: str | None) -> None:
    """Smoke-test the Kalshi client — list open markets + prices."""
    from .kalshi import KalshiClient

    k = KalshiClient()
    try:
        markets = k.get_markets(status=status, series_ticker=series, limit=limit)
    finally:
        k.close()

    if not markets:
        console.print(f"[yellow]No {status} markets found.[/yellow]")
        return

    table = Table(title=f"Kalshi {status} markets ({len(markets)})")
    table.add_column("Ticker")
    table.add_column("Title")
    table.add_column("YES bid", justify="right")
    table.add_column("YES ask", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Vol 24h", justify="right")
    table.add_column("Closes")
    for m in markets:
        table.add_row(
            m.ticker[:40],
            (m.title + " " + m.subtitle)[:50],
            f"{m.yes_bid:.3f}", f"{m.yes_ask:.3f}",
            f"{m.last_price:.3f}",
            f"{m.volume_24h:.0f}",
            m.close_time.strftime("%m-%d %H:%M"),
        )
    console.print(table)


#: Polymarket slug fragments that identify strike-based crypto
#: up/down markets. These match poorly against Kalshi's own crypto
#: strike markets via title-fuzzy alone (bogus hits on "Bitcoin above
#: X vs Bitcoin above Y"), so we exclude them — the hourly bot already
#: handles same-venue edge on these.
_POLY_STRIKE_MARKET_MARKERS = ("up-or-down", "updown")

#: Kalshi series with strike-based crypto markets; excluded from arb
#: scan for the same reason as above. Categorical markets (Trump
#: mentions, sports, elections) match cleanly on titles.
_KALSHI_STRIKE_SERIES = frozenset({"KXBTC", "KXBTCD", "KXETHD", "KXSOL", "KXETH", "KXSP500"})


@main.command("scan-arb")
@click.option(
    "--horizon-days", default=7, show_default=True, type=float,
    help="Only scan markets closing within this many days",
)
@click.option(
    "--min-edge", default=0.07, show_default=True, type=float,
    help="Minimum post-fee edge (0.07 = 7%) — see crossvenue.py for fee math",
)
@click.option(
    "--min-vol", default=100.0, show_default=True, type=float,
    help="Minimum 24h Kalshi contract volume to consider liquid",
)
@click.option(
    "--similarity", default=0.70, show_default=True, type=float,
    help="Title-fuzzy-match threshold (0-1). Lowering below 0.65 lets in "
         "strike-vs-strike false positives.",
)
@click.option(
    "--time-bucket-hours", default=12.0, show_default=True, type=float,
    help="Candidate pairs must close within ±N hours of each other",
)
@click.option("--poly-safe", is_flag=True, help="Only show opportunities where Polymarket is the cheap leg (EU-safe)")
@click.option(
    "--include-strike-markets", is_flag=True,
    help="Don't filter out strike-based crypto markets (off by default — "
         "they generate false positives on title-fuzzy match)",
)
@click.option(
    "--llm-match", is_flag=True,
    help="Use a local LM-Studio LLM to score market pair equivalence "
         "instead of the fuzzy-text matcher. Requires LM Studio running.",
)
@click.option(
    "--llm-threshold", default=0.7, show_default=True, type=float,
    help="Minimum LLM equivalence score (0-1) to treat a pair as the same market",
)
def scan_arb(
    horizon_days: float, min_edge: float, min_vol: float,
    similarity: float, time_bucket_hours: float,
    poly_safe: bool, include_strike_markets: bool,
    llm_match: bool, llm_threshold: float,
) -> None:
    """Scan for Kalshi↔Polymarket arb opportunities above ``min_edge``."""
    import time
    from datetime import datetime, timedelta, timezone

    from .kalshi import KalshiClient

    max_sec = int(horizon_days * 86400)
    max_close_ts = int((datetime.now(timezone.utc) + timedelta(seconds=max_sec)).timestamp())

    console.print(f"[dim]Scanning markets closing within {horizon_days:g} days…[/dim]")

    t0 = time.time()
    with httpx.Client(timeout=20.0) as http:
        poly = fetch_all_open_markets(
            http, max_seconds_until_end=max_sec, min_seconds_until_end=300,
        )
    if not include_strike_markets:
        poly = [p for p in poly if not any(s in p.slug for s in _POLY_STRIKE_MARKET_MARKERS)]
    console.print(f"[dim]  polymarket: {len(poly)} categorical binary markets ({time.time()-t0:.1f}s)[/dim]")

    t1 = time.time()
    k = KalshiClient()
    try:
        kalshi_mkts = k.fetch_liquid_markets(
            min_volume_24h=min_vol, max_close_ts=max_close_ts,
        )
    finally:
        k.close()
    if not include_strike_markets:
        kalshi_mkts = [m for m in kalshi_mkts if m.series_ticker not in _KALSHI_STRIKE_SERIES]
    console.print(f"[dim]  kalshi:     {len(kalshi_mkts)} liquid categorical markets ({time.time()-t1:.1f}s)[/dim]")

    if not poly or not kalshi_mkts:
        console.print("[yellow]Empty side — nothing to scan.[/yellow]")
        return

    t2 = time.time()
    if llm_match:
        from .llm_match import check_server

        ok, msg = check_server()
        if not ok:
            console.print(f"[red]LM Studio unreachable:[/red] {msg}")
            console.print("[red]Start LM Studio and load a model before using --llm-match.[/red]")
            return
        console.print(f"[dim]  LM Studio: {msg}[/dim]")

        def on_progress(stage: str, n: int) -> None:
            if stage == "candidates":
                console.print(f"[dim]  token+time filter → {n} candidate pairs, scoring with LLM…[/dim]")

        opps = scan_opportunities_llm(
            poly, kalshi_mkts,
            min_edge=min_edge, min_llm_equiv=llm_threshold,
            bucket=timedelta(hours=time_bucket_hours),
            progress_fn=on_progress,
        )
    else:
        # scan_opportunities fetches its own Kalshi set by default; we
        # already have one, so pass the markets in via a shim client.
        opps = scan_opportunities(
            poly, kalshi=_StubKalshi(kalshi_mkts),  # type: ignore[arg-type]
            min_edge=min_edge, similarity=similarity,
            bucket=timedelta(hours=time_bucket_hours),
        )
    console.print(f"[dim]  matched in {time.time()-t2:.1f}s — {len(opps)} opps[/dim]")

    if poly_safe:
        opps = [o for o in opps if o.cheap_venue == "polymarket"]

    if not opps:
        console.print(f"[yellow]No opportunities ≥ {min_edge*100:.1f}%.[/yellow]")
        return

    table = Table(title=f"{len(opps)} arb opportunities ≥ {min_edge*100:.1f}%")
    table.add_column("Edge", justify="right", style="green")
    table.add_column("Buy YES", width=10)
    table.add_column("Sell YES", width=10)
    table.add_column("Sim", justify="right")
    table.add_column("Polymarket", max_width=38)
    table.add_column("Kalshi", max_width=38)
    table.add_column("Notes", style="dim", max_width=20)
    for o in opps[:40]:
        poly_px = o.poly.yes_price
        kal_px = o.kalshi.mid
        table.add_row(
            f"{o.edge_after_fees*100:+.1f}%",
            f"{o.cheap_venue[:4]} @{(poly_px if o.cheap_venue=='polymarket' else kal_px):.3f}",
            f"{o.expensive_venue[:4]} @{(kal_px if o.cheap_venue=='polymarket' else poly_px):.3f}",
            f"{o.similarity:.2f}",
            o.poly.question[:38],
            (o.kalshi.title + " " + o.kalshi.subtitle).strip()[:38],
            o.notes,
        )
    console.print(table)


class _StubKalshi:
    """Adapter so ``scan_opportunities`` can accept a pre-fetched list
    of Kalshi markets instead of hitting the API a second time."""

    def __init__(self, markets):
        self._m = markets

    def get_markets(self, **kwargs):
        return self._m

    def close(self):
        pass


if __name__ == "__main__":
    main()
