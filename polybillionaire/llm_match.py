"""Semantic market-pair matcher using a local LM-Studio model.

Title-fuzzy matching (``crossvenue._title_similarity``) can't bridge the
phrasing gap between Kalshi and Polymarket for the same underlying
event (observed 2026-04-17: zero real opportunities, bogus template
overlaps). This module asks gpt-oss-20b via LM Studio's OpenAI-
compatible API whether two market questions resolve on the same
event, returning a 0–1 equivalence score.

Requires LM Studio running at ``$LMSTUDIO_URL`` (default
``http://localhost:1234/v1``) with ``$LLM_MATCH_MODEL`` loaded
(default ``openai/gpt-oss-20b``). See memory ``models_gpt_oss.md``
for the chosen model stack.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"

#: Max pairs per LLM call. 20 fits well under the 20b context and keeps
#: individual calls fast (~5-10s each on an M-class Mac).
DEFAULT_BATCH = 20

_SYSTEM_PROMPT = (
    "You are a prediction-market equivalence checker. For each numbered "
    "pair of market questions, decide whether they resolve on the exact "
    "same real-world event (same asset, same strike/outcome, same expiry "
    "window). Phrasing will differ — judge the underlying resolution "
    "condition, not the wording.\n\n"
    "Return a JSON array. Each element has:\n"
    '  {"i": <int>, "equiv": <0.0-1.0>, "reason": "<<=15 words>"}\n\n'
    "Scoring guide:\n"
    "  1.0  — identical resolution (same asset, strike, expiry)\n"
    "  0.7  — same underlying event but slightly different strike/date\n"
    "  0.3  — related topic, different resolution condition\n"
    "  0.0  — unrelated\n\n"
    "Output ONLY the JSON array, no prose, no code fences."
)


@dataclass
class PairScore:
    index: int
    equiv: float
    reason: str


def score_pairs(
    pairs: list[tuple[str, str]],
    *,
    batch: int = DEFAULT_BATCH,
    base_url: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    reasoning_effort: str = "low",
) -> list[PairScore]:
    """Score each ``(question_a, question_b)`` pair in [0,1].

    Results align with ``pairs`` index. Pairs that fail to parse
    receive ``equiv=0.0, reason="parse-error"`` so the caller can
    filter on a threshold without worrying about gaps.
    """
    base_url = (base_url or os.getenv("LMSTUDIO_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = model or os.getenv("LLM_MATCH_MODEL") or DEFAULT_MODEL

    out: list[PairScore] = [
        PairScore(i, 0.0, "unscored") for i in range(len(pairs))
    ]
    if not pairs:
        return out

    with httpx.Client(timeout=timeout) as http:
        for start in range(0, len(pairs), batch):
            chunk = pairs[start : start + batch]
            scored = _score_chunk(
                http, chunk, base_url=base_url, model=model,
                reasoning_effort=reasoning_effort,
            )
            for local_i, ps in scored.items():
                gi = start + local_i
                if 0 <= gi < len(out):
                    out[gi] = PairScore(gi, ps[0], ps[1])
    return out


def _score_chunk(
    http: httpx.Client,
    chunk: list[tuple[str, str]],
    *,
    base_url: str,
    model: str,
    reasoning_effort: str,
) -> dict[int, tuple[float, str]]:
    numbered = "\n".join(
        f"{i}. A: {a!r}\n   B: {b!r}" for i, (a, b) in enumerate(chunk)
    )
    system = f"Reasoning: {reasoning_effort}\n\n{_SYSTEM_PROMPT}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    r = http.post(f"{base_url}/chat/completions", json=body)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    arr = _extract_json_array(content)
    results: dict[int, tuple[float, str]] = {}
    if arr is None:
        return results
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("i"))
            equiv = float(item.get("equiv", 0.0))
        except (TypeError, ValueError):
            continue
        equiv = max(0.0, min(1.0, equiv))
        reason = str(item.get("reason", ""))[:120]
        results[i] = (equiv, reason)
    return results


def _extract_json_array(text: str) -> list | None:
    """Pull a JSON array out of the model response. Some models prefix
    with whitespace or stray prose even when told not to; scan for
    the first ``[`` and matching ``]``."""
    if not text:
        return None
    # Fast path — already valid JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    # Slow path — slice from first '[' to last ']'.
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def check_server(base_url: str | None = None) -> tuple[bool, str]:
    """Return (ok, message) for the configured LM Studio endpoint."""
    base_url = (base_url or os.getenv("LMSTUDIO_URL") or DEFAULT_BASE_URL).rstrip("/")
    try:
        r = httpx.get(f"{base_url}/models", timeout=3.0)
        r.raise_for_status()
        data = r.json()
        ids = [m.get("id", "") for m in data.get("data", [])]
        return True, f"{len(ids)} models: {', '.join(ids[:5])}"
    except (httpx.HTTPError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"
