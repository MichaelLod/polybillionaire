"""Leveraged 1h crypto directional trading on Binance USDM futures.

Each UTC hour:
1. A few minutes into the candle, evaluate ``predict_up_probability``
   on the current 1h bar for each target symbol.
2. If |p - 0.5| exceeds the edge threshold, open LONG / SHORT with
   the configured leverage, attach a STOP_MARKET reduce-only at the
   configured % away from entry.
3. At T-30s before the hour close, cancel any resting stop and
   market-close the position — we take whatever the candle delivered.

One position per symbol per hour. A stop fill counts as the hour's
exit; we don't re-enter after a stop.

Realistic expectation: the signal edge is tiny (model error ≫ any real
directional prior at 1h). Leverage amplifies both edge and noise — so
this is primarily a calibration harness, not a money printer. The
stop-loss and per-trade margin cap are what keep losing fast from
becoming losing everything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from .binance import fetch_snapshot
from .binance_exec import BinanceFutures, BinanceError
from .signals import predict_up_probability

#: Ticker → Binance USDT-quoted perp symbol. USDT pairs are the
#: deepest book on Binance futures. When the account has Multi-Assets
#: Mode enabled, USDC collateral backs USDT-pair margin 1:1 (small
#: haircut). USDC-pair perps exist but sometimes go reduce-only under
#: Binance-wide policies — USDT doesn't.
USDC_SYMBOLS: dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
}

#: Spot feature symbols == futures symbols now that we're on USDT.
SPOT_FEATURE_SYMBOLS: dict[str, str] = {v: v for v in USDC_SYMBOLS.values()}


@dataclass
class FuturesConfig:
    symbols: list[str]            # USDC pair symbols (e.g. BTCUSDC)
    leverage: int = 5
    stop_pct: float = 0.007       # 0.7% of entry
    edge_threshold: float = 0.08  # |p_up - 0.5| must exceed this
    margin_fraction_per_trade: float = 0.20  # % of available USDC used as margin per entry
    max_margin_per_trade_usd: float = 25.0   # hard cap regardless of balance
    cycle_s: float = 20.0
    #: Don't open new trades in the final N seconds of the hour.
    entry_cutoff_s: int = 600     # no new entries in last 10 min
    #: Close any open positions N seconds before the hour rolls over.
    flatten_before_close_s: int = 45
    #: Lock gains: if favorable move ≥ take_profit_pct during the
    #: ``entry_cutoff_s`` window (last 10 min), close early. 0 disables.
    #: Research basis: forced-flatten at close costs slippage; moving the
    #: exit earlier on a solid winner avoids that drag.
    take_profit_pct: float = 0.02
    dry_run: bool = True


class FuturesBot:
    def __init__(
        self,
        exec: BinanceFutures,
        config: FuturesConfig,
        *,
        print_fn=print,
    ) -> None:
        self.exec = exec
        self.config = config
        self._print = print_fn
        self._http = httpx.Client(timeout=15.0)
        #: Track hours for which we've already opened/closed per symbol
        #: so we don't re-enter after a stop fill or post-flatten.
        self._traded_this_hour: dict[str, int] = {}

    def close(self) -> None:
        self._http.close()

    def run(self) -> None:
        cfg = self.config
        mode = "DRY-RUN" if cfg.dry_run else "LIVE"
        tp_desc = (
            f"tp={cfg.take_profit_pct:.2%}" if cfg.take_profit_pct > 0 else "tp=off"
        )
        self._print(
            f"Futures bot [{mode}]: {len(cfg.symbols)} symbols, "
            f"leverage={cfg.leverage}×, stop={cfg.stop_pct:.2%}, "
            f"edge≥{cfg.edge_threshold:.2%}, "
            f"margin={cfg.margin_fraction_per_trade:.0%}/trade "
            f"(cap ${cfg.max_margin_per_trade_usd:.0f}), {tp_desc}"
        )
        try:
            # Preload symbol specs (tick/step/min qty)
            self.exec.load_specs(cfg.symbols)
        except Exception as e:
            self._print(f"load_specs failed: {e}")
            return

        try:
            while True:
                start = time.monotonic()
                try:
                    self._cycle()
                except BinanceError as e:
                    self._print(f"[{_hms()}] binance error: {e}")
                except Exception as e:
                    self._print(f"[{_hms()}] cycle error: {e}")
                elapsed = time.monotonic() - start
                time.sleep(max(cfg.cycle_s - elapsed, 1.0))
        except KeyboardInterrupt:
            self._print("\nStopped.")
        finally:
            self.close()

    # ── Main cycle ─────────────────────────────────

    def _cycle(self) -> None:
        now = datetime.now(timezone.utc)
        hour_end = _next_hour_close(now)
        s_until_close = (hour_end - now).total_seconds()
        hour_tag = int(hour_end.timestamp())

        positions = self.exec.get_positions() if not self.config.dry_run else []

        # 0) Synthetic stop check — exchange-side stops aren't supported
        #    in Credits Trading Mode (-4120), so we poll positions and
        #    flatten any that have moved adversely past stop_pct.
        positions = self._apply_synthetic_stops(positions)

        # 0b) Lock-gains: inside the last-10-min window, close any
        #     position that's already ≥ take_profit_pct favorable.
        if s_until_close <= self.config.entry_cutoff_s:
            positions = self._apply_takeprofit(positions)

        # 1) Flatten positions close to hour end
        if s_until_close <= self.config.flatten_before_close_s:
            for p in positions:
                if p.symbol in self.config.symbols:
                    if self.config.dry_run:
                        self._print(
                            f"[{_hms()}] DRY flatten {p.side} {p.symbol} "
                            f"qty={p.qty} mark=${p.mark_price:.2f} upnl=${p.unrealized_pnl:+.3f}"
                        )
                    else:
                        try:
                            self.exec.cancel_all_orders(p.symbol)
                            self.exec.close_position(p.symbol)
                            self._print(
                                f"[{_hms()}] CLOSED {p.side} {p.symbol} "
                                f"qty={p.qty} pnl=${p.unrealized_pnl:+.3f}"
                            )
                        except BinanceError as e:
                            self._print(f"[{_hms()}] close failed {p.symbol}: {e}")
            return

        # 2) Too early or too late for new entries
        if s_until_close <= self.config.entry_cutoff_s:
            return

        # 3) Evaluate entries
        held_syms = {p.symbol for p in positions}

        # Pull snapshot per symbol (dedup via feature-symbol)
        snapshots = {}
        for sym in self.config.symbols:
            feat_sym = SPOT_FEATURE_SYMBOLS.get(sym, sym)
            if feat_sym in snapshots:
                continue
            snap = fetch_snapshot(self._http, feat_sym)
            if snap.last_price > 0:
                snapshots[feat_sym] = snap

        entered = 0
        for sym in self.config.symbols:
            if sym in held_syms:
                continue
            if self._traded_this_hour.get(sym) == hour_tag:
                continue  # already traded (and exited) this hour

            feat_sym = SPOT_FEATURE_SYMBOLS.get(sym, sym)
            snap = snapshots.get(feat_sym)
            if snap is None:
                continue

            pred = predict_up_probability(
                snap, duration_s=3600, seconds_remaining=s_until_close,
            )
            if pred is None:
                continue

            edge = pred.p_up - 0.5
            side: str | None = None
            if edge > self.config.edge_threshold:
                side = "LONG"
            elif edge < -self.config.edge_threshold:
                side = "SHORT"

            if side is None:
                self._print(
                    f"[{_hms()}] {sym} p={pred.p_up:.3f} edge={edge:+.3f} — hold"
                )
                continue

            if self._enter(sym, side, snap.last_price, pred, hour_tag):
                entered += 1

        if entered == 0 and not positions:
            self._print(
                f"[{_hms()}] no entries. {s_until_close/60:.0f}m to hour close."
            )

    def _enter(
        self,
        symbol: str,
        side: str,
        reference_price: float,
        prediction,
        hour_tag: int,
    ) -> bool:
        cfg = self.config
        try:
            available = (
                self.exec.get_available_collateral()
                if not cfg.dry_run else 50.0
            )
        except BinanceError as e:
            self._print(f"[{_hms()}] balance fetch failed: {e}")
            return False

        margin = min(
            available * cfg.margin_fraction_per_trade,
            cfg.max_margin_per_trade_usd,
        )
        if margin <= 0:
            self._print(f"[{_hms()}] no collateral available")
            return False

        notional = margin * cfg.leverage

        try:
            qty = self.exec.compute_qty_for_notional(
                symbol, notional, reference_price,
            )
        except Exception as e:
            self._print(f"[{_hms()}] spec fetch failed {symbol}: {e}")
            return False

        if qty <= 0:
            self._print(
                f"[{_hms()}] {symbol} notional ${notional:.2f} too small "
                f"(qty rounds to 0 at ${reference_price:.2f})"
            )
            return False

        stop_price = (
            reference_price * (1 - cfg.stop_pct)
            if side == "LONG"
            else reference_price * (1 + cfg.stop_pct)
        )

        if cfg.dry_run:
            self._print(
                f"[{_hms()}] DRY {side} {symbol} p={prediction.p_up:.3f} "
                f"ref=${reference_price:.2f} qty={qty} "
                f"notional=${qty*reference_price:.2f} "
                f"margin=${margin:.2f} stop=${stop_price:.2f}"
            )
            return True

        # Live path ──────────────────────────────
        try:
            self.exec.set_leverage(symbol, cfg.leverage)
            self.exec.set_isolated_margin(symbol)
            entry_side = "BUY" if side == "LONG" else "SELL"
            order = self.exec.open_market(symbol, entry_side, qty)
        except BinanceError as e:
            self._print(f"[{_hms()}] entry failed {symbol}: {e}")
            return False

        entry_price = float(order.get("avgPrice") or 0) or reference_price
        stop_from_entry = (
            entry_price * (1 - cfg.stop_pct)
            if side == "LONG"
            else entry_price * (1 + cfg.stop_pct)
        )
        try:
            self.exec.place_stop_market(symbol, side, stop_from_entry)
        except BinanceError as e:
            self._print(
                f"[{_hms()}] WARN: stop place failed {symbol}: {e} — "
                "will rely on end-of-hour flatten"
            )

        self._traded_this_hour[symbol] = hour_tag
        self._print(
            f"[{_hms()}] OPEN {side} {symbol} @ ${entry_price:.2f} "
            f"qty={qty} stop=${stop_from_entry:.2f} "
            f"margin=${margin:.2f} notional=${qty*entry_price:.2f}"
        )
        return True

    def _apply_takeprofit(self, positions):
        """Close any position with ≥ take_profit_pct favorable move.

        Only called inside the entry-cutoff window (last 10 min of the
        hour). The forced-flatten 45s before close costs market-order
        slippage; if a winner is already at target, exit now instead.
        """
        tp = self.config.take_profit_pct
        if self.config.dry_run or not positions or tp <= 0:
            return positions
        survivors = []
        for p in positions:
            if p.symbol not in self.config.symbols:
                survivors.append(p)
                continue
            if p.entry_price <= 0 or p.mark_price <= 0:
                survivors.append(p)
                continue
            if p.side == "LONG":
                favorable = (p.mark_price - p.entry_price) / p.entry_price
            else:
                favorable = (p.entry_price - p.mark_price) / p.entry_price
            if favorable >= tp:
                try:
                    self.exec.cancel_all_orders(p.symbol)
                    self.exec.close_position(p.symbol)
                    self._print(
                        f"[{_hms()}] TAKEPROFIT {p.side} {p.symbol} "
                        f"entry=${p.entry_price:.4f} mark=${p.mark_price:.4f} "
                        f"favorable={favorable:.2%} pnl=${p.unrealized_pnl:+.3f}"
                    )
                except BinanceError as e:
                    self._print(f"[{_hms()}] take-profit close failed {p.symbol}: {e}")
                    survivors.append(p)
            else:
                survivors.append(p)
        return survivors

    def _apply_synthetic_stops(self, positions):
        """Close any position whose adverse move has hit ``stop_pct``.

        Called every cycle as a fallback for Credits Trading Mode,
        which rejects exchange-side STOP_MARKET orders (-4120).
        """
        if self.config.dry_run or not positions:
            return positions
        survivors = []
        for p in positions:
            if p.symbol not in self.config.symbols:
                survivors.append(p)
                continue
            if p.entry_price <= 0 or p.mark_price <= 0:
                survivors.append(p)
                continue
            if p.side == "LONG":
                adverse = (p.entry_price - p.mark_price) / p.entry_price
            else:
                adverse = (p.mark_price - p.entry_price) / p.entry_price
            if adverse >= self.config.stop_pct:
                try:
                    self.exec.close_position(p.symbol)
                    self._print(
                        f"[{_hms()}] STOPPED {p.side} {p.symbol} "
                        f"entry=${p.entry_price:.4f} mark=${p.mark_price:.4f} "
                        f"adverse={adverse:.2%} pnl=${p.unrealized_pnl:+.3f}"
                    )
                except BinanceError as e:
                    self._print(f"[{_hms()}] synthetic stop close failed {p.symbol}: {e}")
                    survivors.append(p)
            else:
                survivors.append(p)
        return survivors

    def _available_usdc(self) -> float:
        """Fallback balance lookup for USDC-collateralized accounts."""
        try:
            balances = self.exec._get("/fapi/v2/balance")  # type: ignore
        except BinanceError:
            return 0.0
        for a in balances:
            if a.get("asset") == "USDC":
                return float(a.get("availableBalance", 0))
        return 0.0


def _next_hour_close(now: datetime) -> datetime:
    return (now.replace(minute=0, second=0, microsecond=0)
            + timedelta(hours=1))


def _hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
