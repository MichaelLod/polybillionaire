"""Main trading loop for short-duration Polymarket crypto markets.

Every ``cycle_s`` seconds:
1. Settle any positions whose markets resolved (paper or live).
2. Fetch active up-or-down crypto markets from Gamma.
3. Pull a Binance snapshot per unique symbol (price + funding).
4. For each market: compute P(up), compare to Polymarket book.
5. Open a position when ``|p_model - book_mid|`` exceeds the edge
   threshold and the market isn't halted (>= ``min_seconds_until_end``
   remaining).

Positions resolve themselves when the candle closes — no explicit
exit logic. That makes the loop safe to kill at any time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from .binance import Snapshot, fetch_snapshot
from .client import PolymarketClient
from .gamma import UpDownMarket, fetch_updown_markets
from .signals import Prediction, predict_up_probability
from .trader import (
    POLYMARKET_MIN_MARKET_BUY_USD,
    POLYMARKET_MIN_SHARES,
    LiveTrader,
    PaperTrader,
)


@dataclass
class HourlyConfig:
    #: Minimum |p_model - book_mid| to open a position. Polymarket added
    #: a **dynamic taker fee** in late 2024 (~3.15% at 50¢ contracts)
    #: that killed easy latency arb and raised the break-even. Combined
    #: with slippage on thin books (~1–2%) and residual model error,
    #: the post-fee break-even is ≥ 0.07. 0.05 was pre-dynamic-fee.
    edge_threshold: float = 0.07
    #: Only consider markets ending within this window.
    max_seconds_until_end: int = 3600
    #: Skip markets too close to resolution — Polymarket halts trading
    #: ~1–5 min before candle close on short markets.
    min_seconds_until_end: int = 300
    #: Loop period.
    cycle_s: float = 30.0
    #: Kelly fraction multiplier (0.5 = half-Kelly). Our edge estimate
    #: is noisy; full Kelly over-sizes.
    kelly_fraction: float = 0.5
    #: Never bet into a YES token priced above this (tail markets with
    #: asymmetric loss).
    max_entry_price: float = 0.95
    #: Skip 5m markets by default — fees eat the edge and the signal
    #: is near-noise at that horizon.
    trade_5m: bool = False
    #: Skip 15m markets by default for similar reasons, though with
    #: the 15m flag the bot will trade them.
    trade_15m: bool = False
    #: When true, log every would-be trade but don't place orders.
    dry_run: bool = False


class HourlyBot:
    def __init__(
        self,
        client: PolymarketClient,
        trader: PaperTrader | LiveTrader,
        config: HourlyConfig | None = None,
        *,
        print_fn=print,
    ) -> None:
        self.client = client
        self.trader = trader
        self.config = config or HourlyConfig()
        self._print = print_fn
        self._http = httpx.Client(timeout=15.0)

    def close(self) -> None:
        self._http.close()

    def run(self) -> None:
        cfg = self.config
        self._print(
            f"Hourly bot: edge≥{cfg.edge_threshold:.1%}, "
            f"window≤{cfg.max_seconds_until_end}s, "
            f"kelly×{cfg.kelly_fraction}, bankroll=${self.trader.bankroll:.2f}"
        )
        try:
            while True:
                start = time.monotonic()
                try:
                    self._cycle()
                except Exception as e:
                    self._print(f"[{_hms()}] cycle error: {e}")
                elapsed = time.monotonic() - start
                time.sleep(max(cfg.cycle_s - elapsed, 1.0))
        except KeyboardInterrupt:
            self._print("\nStopped.")
        finally:
            self.close()

    def _cycle(self) -> None:
        for s in self.trader.settle_resolved():
            self._print(f"[{_hms()}] {s['message']}")

        markets = fetch_updown_markets(
            self._http,
            max_seconds_until_end=self.config.max_seconds_until_end,
            min_seconds_until_end=self.config.min_seconds_until_end,
        )
        markets = [m for m in markets if self._accept_duration(m)]
        if not markets:
            self._print(f"[{_hms()}] no eligible markets")
            return

        snapshots: dict[str, Snapshot] = {}
        for sym in {m.symbol for m in markets}:
            snap = fetch_snapshot(self._http, sym)
            if snap.last_price > 0:
                snapshots[sym] = snap

        self._print(
            f"[{_hms()}] {len(markets)} markets, {len(snapshots)} symbols"
        )

        for m in markets:
            snap = snapshots.get(m.symbol)
            if snap is None:
                continue
            self._consider(m, snap)

    def _accept_duration(self, m: UpDownMarket) -> bool:
        label = m.duration_label
        if label == "5m" and not self.config.trade_5m:
            return False
        if label == "15m" and not self.config.trade_15m:
            return False
        return True

    def _consider(self, market: UpDownMarket, snapshot: Snapshot) -> None:
        if self._already_in(market):
            return

        # Skip pre-candle markets — our model only adds value once we
        # can anchor on the settlement candle's current-price-vs-open.
        # Before the candle opens we're basically guessing 50% and the
        # market has strictly more information.
        if market.seconds_until_end > market.duration_s:
            return

        pred = predict_up_probability(
            snapshot,
            duration_s=market.duration_s,
            seconds_remaining=market.seconds_until_end,
        )
        if pred is None:
            return

        try:
            up_book = self.client.get_orderbook(market.up_token_id)
        except Exception:
            return

        market_p_up = up_book.midpoint if up_book.midpoint > 0 else market.up_price
        if not 0.02 < market_p_up < 0.98:
            return  # degenerate — orderbook empty or market ending

        edge_up = pred.p_up - market_p_up
        edge_down = -edge_up

        if (
            edge_up >= self.config.edge_threshold
            and up_book.best_ask > 0
            and up_book.best_ask <= self.config.max_entry_price
        ):
            self._enter(market, "Up", market.up_token_id, up_book.best_ask, pred, edge_up)
            return

        if edge_down >= self.config.edge_threshold:
            try:
                down_book = self.client.get_orderbook(market.down_token_id)
            except Exception:
                return
            if (
                down_book.best_ask > 0
                and down_book.best_ask <= self.config.max_entry_price
            ):
                self._enter(
                    market, "Down", market.down_token_id,
                    down_book.best_ask, pred, edge_down,
                )
                return

        self._print(
            f"[{_hms()}] {market.symbol} {market.duration_label} "
            f"p={pred.p_up:.3f} mkt={market_p_up:.3f} "
            f"edge={edge_up:+.3f} — hold"
        )

    def _enter(
        self,
        market: UpDownMarket,
        outcome: str,
        token_id: str,
        entry_price: float,
        prediction: Prediction,
        edge: float,
    ) -> None:
        size = self._position_size(
            p_true=prediction.p_up if outcome == "Up" else 1 - prediction.p_up,
            entry_price=entry_price,
        )
        if size <= 0:
            return

        if self.config.dry_run:
            self._print(
                f"[{_hms()}] DRY {outcome} {market.symbol} "
                f"{market.duration_label} p={prediction.p_up:.3f} "
                f"price={entry_price:.3f} edge={edge:+.3f} size={size:.1f}"
            )
            return

        ok, msg = self.trader.buy(
            token_id=token_id,
            market_question=market.question,
            outcome=outcome,
            size=size,
            end_date=market.end_time.isoformat(),
        )
        tag = "BOUGHT" if ok else "SKIP"
        self._print(
            f"[{_hms()}] {tag} {outcome} {market.symbol} "
            f"{market.duration_label} p={prediction.p_up:.3f} "
            f"price={entry_price:.3f} edge={edge:+.3f} — {msg}"
        )

    def _position_size(self, *, p_true: float, entry_price: float) -> float:
        """Half-Kelly sized, clamped to risk caps and exchange floors."""
        if not 0 < entry_price < 1 or not 0 < p_true < 1:
            return 0.0

        # Kelly for a binary bet at price p that pays $1 on win:
        #   f* = (p_true - p) / (1 - p)     betting YES at price p
        kelly_f = (p_true - entry_price) / (1 - entry_price)
        if kelly_f <= 0:
            return 0.0
        kelly_f *= self.config.kelly_fraction

        bankroll = self.trader.bankroll
        desired_cost = min(kelly_f * bankroll, self.trader.risk.max_bet)

        live = isinstance(self.trader, LiveTrader)
        min_cost = max(
            POLYMARKET_MIN_SHARES * entry_price,
            POLYMARKET_MIN_MARKET_BUY_USD if live else 0.0,
        )
        if desired_cost < min_cost:
            if min_cost > self.trader.risk.max_bet or min_cost > bankroll:
                return 0.0
            desired_cost = min_cost

        size = round(desired_cost / entry_price, 2)
        if size < POLYMARKET_MIN_SHARES:
            return 0.0
        return size

    def _already_in(self, market: UpDownMarket) -> bool:
        tids = {market.up_token_id, market.down_token_id}
        return any(p.token_id in tids for p in self.trader.positions)


def _hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")
