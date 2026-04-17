"""BTC regime-shift watcher — emits a line when BTC moves ≥ 1% in 1 min.

Research basis: BTC-SOL correlation currently ~0.99 and BTC→ETH→alt
spillover is the documented risk-appetite transmission channel. The
ETH/SOL edge model is trained on their own features and is blind to an
exogenous BTC shock — so a sudden BTC move is the one piece of
information a mid-trade LLM orchestrator can usefully act on.

Run under claude-code Monitor so each emitted line wakes the agent:

    python -m polybillionaire.btc_regime_watcher

Emits at most once per forming 1m candle to avoid spamming during a
sustained move. Poll cadence is 10s.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import httpx

THRESHOLD = 0.01   # 1% open-to-current on the forming 1m candle
POLL_S = 10
BINANCE_URL = "https://fapi.binance.com/fapi/v1/klines"


def fetch_forming_1m() -> tuple[int, float, float]:
    """Return (open_time_ms, open_price, latest_close_price) of the currently
    forming 1m BTCUSDT candle."""
    r = httpx.get(
        BINANCE_URL,
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1},
        timeout=5.0,
    )
    r.raise_for_status()
    k = r.json()[0]
    return int(k[0]), float(k[1]), float(k[4])


def main() -> int:
    last_emit_bucket: int = 0
    print(
        f"[{_ts()}] btc-regime watcher up — threshold ±{THRESHOLD:.1%}, "
        f"poll {POLL_S}s",
        flush=True,
    )
    while True:
        try:
            open_ms, open_p, now_p = fetch_forming_1m()
            delta = (now_p - open_p) / open_p if open_p > 0 else 0.0
            if abs(delta) >= THRESHOLD and open_ms != last_emit_bucket:
                direction = "UP" if delta > 0 else "DOWN"
                print(
                    f"[{_ts()}] BTC REGIME {direction} "
                    f"{delta*100:+.2f}% open=${open_p:,.0f} now=${now_p:,.0f}",
                    flush=True,
                )
                last_emit_bucket = open_ms
        except Exception as e:
            print(f"[{_ts()}] btc-regime err: {e}", file=sys.stderr, flush=True)
        time.sleep(POLL_S)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
