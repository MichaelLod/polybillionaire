# Polybillionaire

Short-duration crypto trading on Polymarket. Trades "BTC/ETH/SOL/XRP/BNB up or
down this hour?" style markets using Binance candle features as the edge source.

## How it works

Polymarket runs up-or-down candle markets for major USDT pairs. They settle
against the Binance spot close vs open on the matching 1h / 15m / 5m candle
(close ≥ open → Up wins). Every cycle, the bot:

1. Queries Gamma for active up-or-down crypto markets.
2. Pulls a Binance snapshot for each symbol (klines + funding rate).
3. Computes `P(up)` via a normal approximation of the remaining-period
   return, anchored at current in-candle price movement.
4. If `|P(up) - Polymarket book mid| ≥ edge threshold`, opens a half-Kelly
   position on the profitable side.
5. Positions resolve themselves when the candle closes.

The `signals.py` model is intentionally tiny: tight `σ` from recent realized
vol, weak momentum + funding drift priors (heavily shrunk), and the
structural math of where current price sits inside the settlement candle.
Coefficients are priors, not backtested — treat live P&L as the training set.

## Honest expectations

1h BTC is near-martingale. After Polymarket fees (~1%) and slippage on
thin books (often $100-ish total volume per hourly event), a well-tuned
model prints ~52% and runs near breakeven. No public backtest gives a
post-cost 1h crypto edge above ~3%. Default edge threshold is 2.5% to
avoid trading on noise.

## Setup

```bash
uv sync
cp .env.example .env   # set BANKROLL, optionally POLY_* for --live
```

## Usage

```bash
pb status              # check Polymarket + Binance reachability
pb discover            # list active up-or-down crypto markets
pb hourly              # run paper trading loop
pb hourly --live       # real USDC (requires POLY_* env)
pb hourly --trade-15m  # also trade 15-min markets (higher fees, less edge)
pb hourly --edge 0.03  # require 3% edge before trading
pb positions           # show open paper positions + P&L
pb balance             # show on-chain USDC balance
```

## Environment

| Var | Default | Notes |
| --- | --- | --- |
| `BANKROLL` | 5.0 | Paper-mode bankroll (USDC). Live mode uses real balance. |
| `MAX_BET_FRACTION` | 0.10 | Max fraction of bankroll per bet. |
| `POLY_PRIVATE_KEY` | — | Required for `--live`. |
| `POLY_API_KEY` / `POLY_API_SECRET` / `POLY_API_PASSPHRASE` | — | Required for `--live`. |
| `POLY_PROXY_ADDRESS` | — | Polymarket proxy address where positions live. |

## License

MIT
