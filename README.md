# Polybillionaire

Autonomous prediction market trading swarm. AI agents continuously research edges, cross-reference external odds against Polymarket prices, and execute trades when they find mispricing.

## Architecture

A swarm of research agents runs continuously, each hunting for signals that prediction markets haven't priced in. When an agent finds an angle, it pushes the finding to a reasoning agent that checks feasibility and proposes trades. Approved trades execute automatically with position sizing, stop-losses, and settlement tracking.

- **Research agents** - always-on, configurable model backends (Haiku, Sonnet, Opus, local LM Studio)
- **Reasoning agent** - evaluates findings against live market data, proposes trades with edge calculations
- **Risk manager** - Kelly-criterion sizing, position limits, stop-losses, daily loss caps
- **Paper trading** - full simulation with realistic orderbook interaction before going live

## Quick Start

```bash
uv sync
cp .env.example .env  # configure your keys
uv run poly org        # launch the trading swarm
```

## License

MIT
