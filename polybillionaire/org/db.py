"""SQLite institutional memory for the trading org.

Tables:
  direction          — company thesis, evolves over time
  hypotheses         — tradeable beliefs linked to direction
  leads              — facts/findings that support/weaken hypotheses
  research_log       — append-only findings trail per lead per cycle
  cycles             — per-cycle metadata (tokens, time, bankroll, strategy)
  hints              — human intuition, tracked to outcomes
  positions          — synced from API each cycle, enriched with hypothesis links
  positions_history  — closed trades with P&L linked to hypotheses
  rejections         — risk blocks with reasons
  market_snapshots   — price tracking for watched markets
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path("org_data.db")

SCHEMA = """\
-- Company direction — the overarching thesis that guides everything.
-- Rows are versioned: insert a new row to evolve direction, don't update.
CREATE TABLE IF NOT EXISTS direction (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thesis      TEXT NOT NULL,
    reasoning   TEXT,
    set_by      TEXT NOT NULL DEFAULT 'CEO',
    cycle       INTEGER,
    created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Tradeable beliefs.  A hypothesis links to a direction and can have
-- many leads supporting or weakening it.
CREATE TABLE IF NOT EXISTS hypotheses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    direction_id    INTEGER REFERENCES direction(id),
    title           TEXT NOT NULL,
    thesis          TEXT,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN (
                        'active', 'investigating', 'edge_found',
                        'traded', 'resolved', 'stale', 'dismissed'
                    )),
    category        TEXT,
    market_question TEXT,
    our_probability REAL,
    market_price    REAL,
    edge            REAL,
    side            TEXT CHECK (side IN ('YES', 'NO')),
    created_cycle   INTEGER,
    updated_cycle   INTEGER,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec')),
    updated_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Facts and findings discovered by research.
CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id   INTEGER REFERENCES hypotheses(id),
    title           TEXT NOT NULL,
    source          TEXT,
    signal          TEXT,
    confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    status          TEXT NOT NULL DEFAULT 'new'
                    CHECK (status IN (
                        'new', 'investigating', 'confirmed',
                        'weakened', 'stale', 'dismissed'
                    )),
    found_by        TEXT NOT NULL DEFAULT 'Research',
    found_cycle     INTEGER,
    category        TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec')),
    updated_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Append-only research trail.  Each cycle that touches a lead adds a row.
CREATE TABLE IF NOT EXISTS research_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    lead_id     INTEGER NOT NULL REFERENCES leads(id),
    cycle       INTEGER NOT NULL,
    agent       TEXT NOT NULL DEFAULT 'Research',
    finding     TEXT NOT NULL,
    confidence  TEXT CHECK (confidence IN ('high', 'medium', 'low')),
    created_at  REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Per-cycle metadata.
CREATE TABLE IF NOT EXISTS cycles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle           INTEGER NOT NULL UNIQUE,
    strategy        TEXT,
    bankroll_start  REAL,
    bankroll_end    REAL,
    positions_count INTEGER,
    trades_executed INTEGER DEFAULT 0,
    trades_rejected INTEGER DEFAULT 0,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    duration_s      REAL,
    summary         TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Human intuition — hints from the operator.
CREATE TABLE IF NOT EXISTS hints (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    text            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN (
                        'pending', 'seen', 'investigating',
                        'lead_created', 'dismissed'
                    )),
    lead_id         INTEGER REFERENCES leads(id),
    cycle_submitted INTEGER,
    cycle_seen      INTEGER,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Positions synced from the Polymarket API, enriched with our metadata.
CREATE TABLE IF NOT EXISTS positions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id        TEXT NOT NULL UNIQUE,
    market_question TEXT,
    outcome         TEXT,
    side            TEXT,
    entry_price     REAL,
    current_price   REAL,
    size            REAL,
    cost            REAL,
    pnl             REAL,
    hypothesis_id   INTEGER REFERENCES hypotheses(id),
    entry_cycle     INTEGER,
    end_date        TEXT,
    synced_at       REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Closed trades with P&L linked back to hypotheses.
CREATE TABLE IF NOT EXISTS positions_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    token_id        TEXT NOT NULL,
    market_question TEXT,
    outcome         TEXT,
    side            TEXT,
    entry_price     REAL,
    exit_price      REAL,
    size            REAL,
    cost            REAL,
    pnl             REAL,
    hypothesis_id   INTEGER REFERENCES hypotheses(id),
    entry_cycle     INTEGER,
    exit_cycle      INTEGER,
    reason          TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Risk rejections — why a trade was blocked.
CREATE TABLE IF NOT EXISTS rejections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle           INTEGER NOT NULL,
    market_question TEXT,
    side            TEXT,
    cost            REAL,
    reason          TEXT NOT NULL,
    hypothesis_id   INTEGER REFERENCES hypotheses(id),
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Price snapshots for markets we're watching.
CREATE TABLE IF NOT EXISTS market_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    market_question TEXT NOT NULL,
    token_id        TEXT,
    price           REAL NOT NULL,
    volume_24h      REAL,
    cycle           INTEGER,
    created_at      REAL NOT NULL DEFAULT (unixepoch('subsec'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_leads_status ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_hypothesis ON leads(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_research_log_lead ON research_log(lead_id);
CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token_id);
CREATE INDEX IF NOT EXISTS idx_market_snapshots_question ON market_snapshots(market_question);
CREATE INDEX IF NOT EXISTS idx_hints_status ON hints(status);
"""


class OrgDB:
    """Institutional memory for the trading org."""

    def __init__(self, path: Path | str = DB_PATH) -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that may be missing from older databases."""
        cols = {
            r["name"]
            for r in self._fetchall("PRAGMA table_info(positions)")
        }
        if "end_date" not in cols:
            self._conn.execute("ALTER TABLE positions ADD COLUMN end_date TEXT")
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def reset(self) -> None:
        """Delete and recreate the database."""
        self._conn.close()
        self.path.unlink(missing_ok=True)
        Path(str(self.path) + "-shm").unlink(missing_ok=True)
        Path(str(self.path) + "-wal").unlink(missing_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None

    # ── Direction ──────────────────────────────────────────────

    def set_direction(self, thesis: str, reasoning: str = "",
                      cycle: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO direction (thesis, reasoning, cycle) VALUES (?, ?, ?)",
                (thesis, reasoning, cycle),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_direction(self) -> dict | None:
        return self._fetchone(
            "SELECT * FROM direction ORDER BY id DESC LIMIT 1"
        )

    def get_direction_history(self, limit: int = 10) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM direction ORDER BY id DESC LIMIT ?", (limit,)
        )

    # ── Hypotheses ─────────────────────────────────────────────

    def add_hypothesis(self, title: str, thesis: str = "",
                       category: str = "", direction_id: int | None = None,
                       cycle: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO hypotheses
                   (title, thesis, category, direction_id, created_cycle, updated_cycle)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (title, thesis, category, direction_id, cycle, cycle),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_hypothesis(self, hyp_id: int, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [hyp_id]
        with self._tx() as cur:
            cur.execute(f"UPDATE hypotheses SET {sets} WHERE id = ?", vals)

    def get_hypothesis(self, hyp_id: int) -> dict | None:
        return self._fetchone("SELECT * FROM hypotheses WHERE id = ?", (hyp_id,))

    def get_active_hypotheses(self) -> list[dict]:
        return self._fetchall(
            """SELECT * FROM hypotheses
               WHERE status NOT IN ('resolved', 'dismissed', 'stale')
               ORDER BY updated_at DESC"""
        )

    def get_hypotheses_by_status(self, status: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM hypotheses WHERE status = ? ORDER BY updated_at DESC",
            (status,),
        )

    # ── Leads ──────────────────────────────────────────────────

    def add_lead(self, title: str, source: str = "", signal: str = "",
                 confidence: str = "medium", found_by: str = "Research",
                 hypothesis_id: int | None = None, category: str = "",
                 cycle: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO leads
                   (title, source, signal, confidence, found_by,
                    hypothesis_id, category, found_cycle)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (title, source, signal, confidence, found_by,
                 hypothesis_id, category, cycle),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def update_lead(self, lead_id: int, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [lead_id]
        with self._tx() as cur:
            cur.execute(f"UPDATE leads SET {sets} WHERE id = ?", vals)

    def get_active_leads(self) -> list[dict]:
        return self._fetchall(
            """SELECT * FROM leads
               WHERE status NOT IN ('stale', 'dismissed')
               ORDER BY updated_at DESC"""
        )

    def get_leads_for_hypothesis(self, hyp_id: int) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM leads WHERE hypothesis_id = ? ORDER BY created_at DESC",
            (hyp_id,),
        )

    def cascade_lead_status(self, hypothesis_id: int, new_status: str) -> int:
        """When a hypothesis is dismissed/resolved/etc, cascade to its leads."""
        lead_status = {
            "dismissed": "dismissed",
            "stale": "stale",
            "won": "dismissed",
            "lost": "dismissed",
            "stopped_out": "dismissed",
            "resolved": "dismissed",
        }.get(new_status)
        if not lead_status:
            return 0
        with self._tx() as cur:
            cur.execute(
                """UPDATE leads SET status = ?, updated_at = ?
                   WHERE hypothesis_id = ?
                   AND status NOT IN ('stale', 'dismissed')""",
                (lead_status, time.time(), hypothesis_id),
            )
            return cur.rowcount

    def expire_stale_leads(self, max_age_hours: float = 24,
                           max_active: int = 30) -> int:
        """Auto-expire old leads and enforce a cap on active leads.

        Runs during free monitoring phase — 0 tokens.
        """
        now = time.time()
        cutoff = now - (max_age_hours * 3600)
        expired = 0

        with self._tx() as cur:
            # 1. Stale leads older than max_age_hours
            cur.execute(
                """UPDATE leads SET status = 'stale', updated_at = ?
                   WHERE status NOT IN ('stale', 'dismissed')
                   AND created_at < ?""",
                (now, cutoff),
            )
            expired += cur.rowcount

            # 2. Orphaned leads: linked to dead hypotheses
            cur.execute(
                """UPDATE leads SET status = 'dismissed', updated_at = ?
                   WHERE status NOT IN ('stale', 'dismissed')
                   AND hypothesis_id IS NOT NULL
                   AND hypothesis_id IN (
                       SELECT id FROM hypotheses
                       WHERE status IN ('dismissed', 'stale', 'won',
                                        'lost', 'stopped_out', 'resolved')
                   )""",
                (now,),
            )
            expired += cur.rowcount

            # 3. Cap: if still over max_active, stale the oldest
            cur.execute(
                """SELECT COUNT(*) as c FROM leads
                   WHERE status NOT IN ('stale', 'dismissed')""",
            )
            active_count = cur.fetchone()["c"]
            if active_count > max_active:
                excess = active_count - max_active
                cur.execute(
                    """UPDATE leads SET status = 'stale', updated_at = ?
                       WHERE id IN (
                           SELECT id FROM leads
                           WHERE status NOT IN ('stale', 'dismissed')
                           ORDER BY created_at ASC
                           LIMIT ?
                       )""",
                    (now, excess),
                )
                expired += cur.rowcount

        return expired

    # ── Research Log ───────────────────────────────────────────

    def log_research(self, lead_id: int, cycle: int, finding: str,
                     confidence: str = "medium", agent: str = "Research") -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO research_log (lead_id, cycle, agent, finding, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (lead_id, cycle, agent, finding, confidence),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_research_trail(self, lead_id: int) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM research_log WHERE lead_id = ? ORDER BY cycle ASC",
            (lead_id,),
        )

    # ── Cycles ─────────────────────────────────────────────────

    def start_cycle(self, cycle: int, bankroll: float,
                    positions_count: int) -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT OR REPLACE INTO cycles (cycle, bankroll_start, positions_count)
                   VALUES (?, ?, ?)""",
                (cycle, bankroll, positions_count),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def end_cycle(self, cycle: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [cycle]
        with self._tx() as cur:
            cur.execute(f"UPDATE cycles SET {sets} WHERE cycle = ?", vals)

    def get_cycle(self, cycle: int) -> dict | None:
        return self._fetchone("SELECT * FROM cycles WHERE cycle = ?", (cycle,))

    def get_recent_cycles(self, limit: int = 10) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM cycles ORDER BY cycle DESC LIMIT ?", (limit,)
        )

    # ── Hints ──────────────────────────────────────────────────

    def add_hint(self, text: str, cycle: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                "INSERT INTO hints (text, cycle_submitted) VALUES (?, ?)",
                (text, cycle),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_pending_hints(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM hints WHERE status = 'pending' ORDER BY created_at ASC"
        )

    def update_hint(self, hint_id: int, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = list(fields.values()) + [hint_id]
        with self._tx() as cur:
            cur.execute(f"UPDATE hints SET {sets} WHERE id = ?", vals)

    # ── Positions (synced cache) ───────────────────────────────

    def sync_positions(self, positions: list[dict],
                       cycle: int | None = None) -> list[dict]:
        """Replace cached positions with fresh data from API.

        Returns list of closed positions (disappeared since last sync).
        Closed positions are automatically logged to positions_history.
        """
        closed: list[dict] = []
        # Deduplicate incoming positions by token_id (keep last)
        deduped: dict[str, dict] = {}
        for p in positions:
            deduped[p["token_id"]] = p
        positions = list(deduped.values())

        with self._tx() as cur:
            # Fetch full existing position data
            existing = {
                r["token_id"]: r for r in
                self._fetchall("SELECT * FROM positions")
            }
            incoming_ids = {p["token_id"] for p in positions}

            # Detect closed positions (in DB but not in new data)
            for token_id, old in existing.items():
                if token_id not in incoming_ids:
                    cur.execute(
                        """INSERT INTO positions_history
                           (token_id, market_question, outcome, side,
                            entry_price, exit_price, size, cost, pnl,
                            hypothesis_id, entry_cycle, exit_cycle, reason)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            token_id, old["market_question"],
                            old["outcome"], old["side"],
                            old["entry_price"], old["current_price"],
                            old["size"], old["cost"], old["pnl"],
                            old["hypothesis_id"], old["entry_cycle"],
                            cycle, "closed",
                        ),
                    )
                    closed.append(old)

            # Replace all positions
            cur.execute("DELETE FROM positions")
            for p in positions:
                meta = existing.get(p["token_id"], {})
                cur.execute(
                    """INSERT INTO positions
                       (token_id, market_question, outcome, side,
                        entry_price, current_price, size, cost, pnl,
                        hypothesis_id, entry_cycle, end_date)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        p["token_id"], p.get("market_question", ""),
                        p.get("outcome", ""), p.get("side", ""),
                        p.get("entry_price", 0), p.get("current_price", 0),
                        p.get("size", 0), p.get("cost", 0), p.get("pnl", 0),
                        meta.get("hypothesis_id"), meta.get("entry_cycle"),
                        p.get("end_date") or meta.get("end_date", ""),
                    ),
                )
        return closed

    def get_positions(self) -> list[dict]:
        return self._fetchall("SELECT * FROM positions ORDER BY pnl DESC")

    def link_position_hypothesis(self, token_id: str, hypothesis_id: int,
                                 entry_cycle: int | None = None) -> None:
        with self._tx() as cur:
            cur.execute(
                """UPDATE positions
                   SET hypothesis_id = ?, entry_cycle = COALESCE(?, entry_cycle)
                   WHERE token_id = ?""",
                (hypothesis_id, entry_cycle, token_id),
            )

    # ── Positions History ──────────────────────────────────────

    def close_position(self, token_id: str, exit_price: float,
                       pnl: float, exit_cycle: int | None = None,
                       reason: str = "") -> int:
        pos = self._fetchone("SELECT * FROM positions WHERE token_id = ?", (token_id,))
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO positions_history
                   (token_id, market_question, outcome, side,
                    entry_price, exit_price, size, cost, pnl,
                    hypothesis_id, entry_cycle, exit_cycle, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    token_id,
                    pos["market_question"] if pos else "",
                    pos["outcome"] if pos else "",
                    pos["side"] if pos else "",
                    pos["entry_price"] if pos else 0,
                    exit_price,
                    pos["size"] if pos else 0,
                    pos["cost"] if pos else 0,
                    pnl,
                    pos["hypothesis_id"] if pos else None,
                    pos["entry_cycle"] if pos else None,
                    exit_cycle,
                    reason,
                ),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_positions_history(self, limit: int = 50) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM positions_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    def get_hypothesis_pnl(self, hypothesis_id: int) -> dict:
        row = self._fetchone(
            """SELECT COUNT(*) as trades, COALESCE(SUM(pnl), 0) as total_pnl,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                      SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses
               FROM positions_history WHERE hypothesis_id = ?""",
            (hypothesis_id,),
        )
        return row or {"trades": 0, "total_pnl": 0, "wins": 0, "losses": 0}

    # ── Rejections ─────────────────────────────────────────────

    def log_rejection(self, cycle: int, market_question: str, side: str,
                      cost: float, reason: str,
                      hypothesis_id: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO rejections
                   (cycle, market_question, side, cost, reason, hypothesis_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (cycle, market_question, side, cost, reason, hypothesis_id),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_recent_rejections(self, limit: int = 20) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM rejections ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )

    # ── Market Snapshots ───────────────────────────────────────

    def snapshot_market(self, market_question: str, price: float,
                        volume_24h: float = 0, token_id: str = "",
                        cycle: int | None = None) -> int:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO market_snapshots
                   (market_question, token_id, price, volume_24h, cycle)
                   VALUES (?, ?, ?, ?, ?)""",
                (market_question, token_id, price, volume_24h, cycle),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def get_market_history(self, market_question: str,
                           limit: int = 50) -> list[dict]:
        return self._fetchall(
            """SELECT * FROM market_snapshots
               WHERE market_question = ?
               ORDER BY created_at DESC LIMIT ?""",
            (market_question, limit),
        )

    # ── Context builders for agent prompts ─────────────────────

    def build_ceo_context(self) -> str:
        """Build a context summary for the CEO agent."""
        parts: list[str] = []

        # Current direction
        d = self.get_direction()
        if d:
            parts.append(f"== CURRENT DIRECTION ==\n{d['thesis']}")

        # Active hypotheses
        hyps = self.get_active_hypotheses()
        if hyps:
            lines = []
            for h in hyps[:10]:
                edge_str = f" | edge {h['edge']:+.0%}" if h["edge"] else ""
                lines.append(
                    f"  [{h['status']}] {h['title']}{edge_str}"
                )
            parts.append("== ACTIVE HYPOTHESES ==\n" + "\n".join(lines))

        # Active leads summary
        leads = self.get_active_leads()
        if leads:
            parts.append(f"== ACTIVE LEADS ==\n  {len(leads)} leads being tracked")

        # Recent rejections
        rejs = self.get_recent_rejections(limit=5)
        if rejs:
            lines = [f"  {r['market_question'][:50]}: {r['reason']}" for r in rejs]
            parts.append("== RECENT REJECTIONS ==\n" + "\n".join(lines))

        # Positions
        positions = self.get_positions()
        if positions:
            lines = []
            for p in positions[:10]:
                lines.append(
                    f"  {p['outcome']} \"{p['market_question'][:50]}\" "
                    f"@ ${p['entry_price']:.3f} → ${p['current_price']:.3f} "
                    f"(PnL: ${p['pnl']:.4f})"
                )
            parts.append("== CURRENT POSITIONS ==\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "(no institutional memory yet)"

    def build_research_context(self) -> str:
        """Build context for Research — what's already known, what to dig into."""
        parts: list[str] = []

        # Direction
        d = self.get_direction()
        if d:
            parts.append(f"== COMPANY DIRECTION ==\n{d['thesis']}")

        # Recent leads — tell Research what's already been found so it
        # doesn't waste tokens re-discovering the same signals.
        active = self.get_active_leads()
        if active:
            lines = [
                f"  - [{l['confidence']}] {l['title']}"
                for l in active[:12]
            ]
            parts.append(
                "== ALREADY FOUND (don't re-research these) ==\n"
                + "\n".join(lines)
            )

        # Hypotheses needing evidence
        hyps = self.get_hypotheses_by_status("investigating")
        if hyps:
            lines = [f"  - {h['title']}: {h['thesis'][:100]}" for h in hyps[:5]]
            parts.append("== HYPOTHESES NEEDING EVIDENCE ==\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else ""

    def build_reasoning_context(self) -> str:
        """Build context for Reasoning — hypotheses, positions, rejections."""
        parts: list[str] = []

        # Active hypotheses with edge estimates
        hyps = self.get_active_hypotheses()
        if hyps:
            lines = []
            for h in hyps[:10]:
                price_str = f" | mkt ${h['market_price']:.3f}" if h["market_price"] else ""
                lines.append(f"  [{h['status']}] {h['title']}{price_str}")
            parts.append("== HYPOTHESES ==\n" + "\n".join(lines))

        # Current positions (avoid duplicates)
        positions = self.get_positions()
        if positions:
            lines = [
                f"  HOLDING: {p['outcome']} \"{p['market_question'][:50]}\" "
                f"@ ${p['entry_price']:.3f}"
                for p in positions[:10]
            ]
            parts.append("== CURRENT POSITIONS (avoid duplicates) ==\n" + "\n".join(lines))

        # Recent rejections (don't re-propose)
        rejs = self.get_recent_rejections(limit=5)
        if rejs:
            lines = [f"  BLOCKED: \"{r['market_question'][:50]}\" — {r['reason']}" for r in rejs]
            parts.append("== RECENT REJECTIONS (don't re-propose) ==\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else ""
