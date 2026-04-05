"""Agent backend abstraction — swap LLM providers without changing agent logic.

Backends:
  ClaudeCliBackend      — wraps ``claude`` CLI subprocess (existing approach)
  AnthropicApiBackend   — direct Anthropic API via httpx
  OpenAiCompatBackend   — LM Studio, Ollama, vLLM (OpenAI-compatible API)
"""

from __future__ import annotations

import abc
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SESSION_DIR = Path("org_sessions")

# ── Response ───────────────────────────────────────────────────

@dataclass
class AgentResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    session_id: str | None = None
    duration_s: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


# ── ABC ────────────────────────────────────────────────────────

class AgentBackend(abc.ABC):
    """Abstract base for all LLM backends."""

    @abc.abstractmethod
    def send(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
    ) -> AgentResponse: ...

    @abc.abstractmethod
    def supports_tools(self) -> bool: ...

    @abc.abstractmethod
    def backend_name(self) -> str: ...

    @abc.abstractmethod
    def cost_per_1k_input(self) -> float: ...

    @abc.abstractmethod
    def cost_per_1k_output(self) -> float: ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.cost_per_1k_input()
            + output_tokens / 1000 * self.cost_per_1k_output()
        )


# ── Claude CLI Backend ─────────────────────────────────────────

CLAUDE_COSTS = {
    "haiku": (0.00025, 0.00125),
    "sonnet": (0.003, 0.015),
    "opus": (0.015, 0.075),
}


class ClaudeCliBackend(AgentBackend):
    """Wraps the ``claude`` CLI subprocess. Supports tools + sessions."""

    def __init__(
        self,
        model: str = "sonnet",
        tools: list[str] | None = None,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.timeout = timeout or (600 if tools and "WebSearch" in tools else 300)
        self._sessions_file = SESSION_DIR / "sessions.json"

    def send(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
    ) -> AgentResponse:
        cmd = [
            "claude", "-p",
            "--output-format", "json",
            "--model", self.model,
        ]
        if self.tools is not None:
            cmd.extend(["--tools", ",".join(self.tools) if self.tools else ""])
        if session_id:
            cmd.extend(["--resume", session_id])
        elif system_prompt:
            cmd.extend(["--system-prompt", system_prompt])

        start = time.time()
        result = subprocess.run(
            cmd, input=prompt, capture_output=True, text=True,
            timeout=self.timeout,
        )
        duration = time.time() - start

        if result.returncode != 0:
            err = (result.stderr or result.stdout).strip()[:200]
            return AgentResponse(
                text=f"[error: {err}]", model=self.model, duration_s=duration,
            )

        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        if json_start < 0:
            return AgentResponse(text=stdout, model=self.model, duration_s=duration)

        try:
            envelope = json.loads(stdout[json_start:])
        except json.JSONDecodeError:
            return AgentResponse(text=stdout, model=self.model, duration_s=duration)

        usage = envelope.get("usage", {})
        input_tok = int(usage.get("input_tokens", 0))
        output_tok = int(usage.get("output_tokens", 0))
        sid = envelope.get("session_id")
        text = str(envelope.get("result", ""))

        return AgentResponse(
            text=text,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            model=self.model,
            session_id=sid,
            duration_s=duration,
            raw=envelope,
        )

    def supports_tools(self) -> bool:
        return True

    def backend_name(self) -> str:
        return f"claude-cli:{self.model}"

    def cost_per_1k_input(self) -> float:
        return CLAUDE_COSTS.get(self.model, (0.003, 0.015))[0]

    def cost_per_1k_output(self) -> float:
        return CLAUDE_COSTS.get(self.model, (0.003, 0.015))[1]


# ── Anthropic API Backend ──────────────────────────────────────

MODEL_IDS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250514",
    "opus": "claude-opus-4-6-20250514",
}


class AnthropicApiBackend(AgentBackend):
    """Direct Anthropic Messages API via httpx."""

    def __init__(
        self,
        model: str = "sonnet",
        api_key: str = "",
        max_tokens: int = 4096,
        timeout: float = 300,
    ) -> None:
        self.model = model
        self.model_id = MODEL_IDS.get(model, model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def send(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
    ) -> AgentResponse:
        import httpx

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        start = time.time()
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return AgentResponse(
                text=f"[api error: {e}]",
                model=self.model,
                duration_s=time.time() - start,
            )

        data = resp.json()
        duration = time.time() - start

        text_blocks = [
            b["text"] for b in data.get("content", []) if b.get("type") == "text"
        ]
        text = "\n".join(text_blocks)
        usage = data.get("usage", {})
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)

        return AgentResponse(
            text=text,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=self.estimate_cost(input_tok, output_tok),
            model=self.model,
            duration_s=duration,
            raw=data,
        )

    def supports_tools(self) -> bool:
        return True

    def backend_name(self) -> str:
        return f"anthropic-api:{self.model}"

    def cost_per_1k_input(self) -> float:
        return CLAUDE_COSTS.get(self.model, (0.003, 0.015))[0]

    def cost_per_1k_output(self) -> float:
        return CLAUDE_COSTS.get(self.model, (0.003, 0.015))[1]


# ── OpenAI-Compatible Backend (LM Studio, Ollama, vLLM) ───────

class OpenAiCompatBackend(AgentBackend):
    """OpenAI-compatible chat completions API for local models."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "local-model",
        api_key: str = "lm-studio",
        max_tokens: int = 4096,
        timeout: float = 300,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout = timeout

    def send(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
    ) -> AgentResponse:
        import httpx

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            return AgentResponse(
                text=f"[local error: {e}]",
                model=self.model,
                duration_s=time.time() - start,
            )

        data = resp.json()
        duration = time.time() - start

        text = ""
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")

        usage = data.get("usage", {})
        input_tok = usage.get("prompt_tokens", 0)
        output_tok = usage.get("completion_tokens", 0)

        return AgentResponse(
            text=text,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=0.0,
            model=self.model,
            duration_s=duration,
            raw=data,
        )

    def supports_tools(self) -> bool:
        return False

    def backend_name(self) -> str:
        return f"openai-compat:{self.model}"

    def cost_per_1k_input(self) -> float:
        return 0.0

    def cost_per_1k_output(self) -> float:
        return 0.0


# ── Web Search / Fetch helpers ────────────────────────────────
#
# Multi-backend: tries Google Custom Search API first (if key set),
# then duckduckgo-search package, then curl fallback.
# Uses subprocess+curl for HTTP to avoid Python 3.14 SSL issues.

def _curl_get(url: str, params: dict[str, str] | None = None,
              timeout: int = 15) -> str | None:
    """HTTP GET via curl subprocess — bypasses Python SSL issues."""
    if params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(params)}"
    try:
        result = subprocess.run(
            ["curl", "-s", "-L", "--max-time", str(timeout),
             "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
             "-H", "Accept-Language: en-US,en;q=0.9",
             url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _google_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Google Custom Search JSON API."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    cx = os.environ.get("GOOGLE_SEARCH_CX", "")
    if not api_key or not cx:
        return []

    raw = _curl_get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": str(max_results)},
    )
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if "error" in data:
        return []
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in data.get("items", [])[:max_results]
    ]


def _ddg_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo via the duckduckgo-search package."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results, region="us-en"))
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except Exception:
        return []


def _curl_ddg_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """DuckDuckGo HTML scrape via curl subprocess."""
    import re as _re
    from urllib.parse import quote_plus, unquote

    raw = _curl_get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
    )
    if not raw:
        return []

    results: list[dict[str, str]] = []
    for m in _re.finditer(
        r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
        raw, _re.DOTALL,
    ):
        url = m.group(1).strip()
        url_match = _re.search(r"uddg=([^&]+)", url)
        if url_match:
            url = unquote(url_match.group(1))
        title = _re.sub(r"<[^>]+>", "", m.group(2)).strip()
        snippet = _re.sub(r"<[^>]+>", "", m.group(3)).strip()
        results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _chrome_bridge_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search via the Chrome extension bridge on localhost:3456."""
    raw = _curl_get(
        "http://localhost:3456/search",
        params={"q": query, "n": str(max_results)},
        timeout=35,
    )
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if "error" in data:
            return []
        return []
    except json.JSONDecodeError:
        return []


def _chrome_bridge_fetch(url: str, max_chars: int = 8000) -> str | None:
    """Fetch via the Chrome extension bridge."""
    raw = _curl_get(
        "http://localhost:3456/fetch",
        params={"url": url},
        timeout=35,
    )
    if not raw:
        return None
    try:
        data = json.loads(raw)
        text = data.get("text", "")
        return text[:max_chars] if text else None
    except json.JSONDecodeError:
        return None


_last_chrome_search: float = 0.0
_chrome_search_lock = threading.Lock()
_CHROME_SEARCH_MIN_INTERVAL = 5.0  # seconds between Chrome searches


def _rate_limited_chrome_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Chrome bridge search with rate limiting to avoid Google CAPTCHAs."""
    global _last_chrome_search
    with _chrome_search_lock:
        elapsed = time.time() - _last_chrome_search
        if elapsed < _CHROME_SEARCH_MIN_INTERVAL:
            return []  # skip, let next backend handle it
        _last_chrome_search = time.time()
    return _chrome_bridge_search(query, max_results)


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web — Google API first, then rate-limited Chrome, then DDG."""
    for search_fn in (_google_search, _rate_limited_chrome_search, _ddg_search, _curl_ddg_search):
        results = search_fn(query, max_results)
        if results:
            return results
    return [{"title": "All search backends failed", "url": "", "snippet": ""}]


def web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL — tries Chrome bridge first, then curl."""
    import re as _re

    # Try Chrome bridge first
    bridge_result = _chrome_bridge_fetch(url, max_chars)
    if bridge_result:
        return bridge_result

    # Fallback to curl
    raw = _curl_get(url, timeout=15)
    if not raw:
        return "[fetch failed]"

    text = _re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=_re.DOTALL)
    text = _re.sub(r"<style[^>]*>.*?</style>", "", text, flags=_re.DOTALL)
    text = _re.sub(r"<[^>]+>", " ", text)
    text = _re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# Tool definitions sent to LM Studio for function calling
_LM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. IMPORTANT: Only call this ONCE per turn with your best query. Do NOT make multiple search calls. NEVER include a year or date in your query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short search query — NO dates, NO years, just keywords",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch the text content of a specific URL. Only call ONCE per turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# ── LM Studio Backend (with tool-calling loop) ───────────────

class LMStudioBackend(AgentBackend):
    """LM Studio via OpenAI-compat endpoint with web search tool loop.

    The model can call ``web_search`` and ``web_fetch`` tools. We execute
    the searches locally (DuckDuckGo + httpx) and feed results back until
    the model produces a final text response.
    """

    MAX_TOOL_ROUNDS = 5
    MAX_TOOL_CALLS_PER_ROUND = 1  # Force one tool call at a time
    TOOL_ROUND_MAX_TOKENS = 2048  # Limit tokens during tool rounds to prevent spam

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "google/gemma-4-26b-a4b",
        timeout: float = 300,
        max_tokens: int = 4096,
        tools: list | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._use_tools = tools is None or len(tools) > 0

    def send(
        self,
        prompt: str,
        system_prompt: str = "",
        session_id: str | None = None,
    ) -> AgentResponse:
        import httpx
        from datetime import datetime, timezone

        # Inject current date — omit year to stop Gemma thinking it's "future"
        today = datetime.now(timezone.utc).strftime("%A, %B %d")
        date_prefix = f"[Today is {today}]\n\n"

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": date_prefix + prompt})

        total_input = 0
        total_output = 0
        start = time.time()

        max_rounds = self.MAX_TOOL_ROUNDS if self._use_tools else 1
        for _round in range(max_rounds):
            last_round = _round == max_rounds - 1

            # On last round, drop tools and ask model to summarize
            if last_round and self._use_tools and _round > 0:
                messages.append({
                    "role": "user",
                    "content": "Now summarize all the information you found. "
                    "Do NOT search again. Write your findings as text. "
                    "ONLY include facts from the search results above — "
                    "do NOT invent or assume any details.",
                })

            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens if last_round else self.TOOL_ROUND_MAX_TOKENS,
            }
            if self._use_tools and not last_round:
                body["tools"] = _LM_TOOLS
                body["parallel_tool_calls"] = False

            try:
                resp = httpx.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers={
                        "Authorization": "Bearer lm-studio",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except httpx.HTTPError as e:
                return AgentResponse(
                    text=f"[lm-studio error: {e}]",
                    model=self.model,
                    duration_s=time.time() - start,
                )

            data = resp.json()
            usage = data.get("usage", {})
            total_input += usage.get("prompt_tokens", 0)
            total_output += usage.get("completion_tokens", 0)

            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            finish = choice.get("finish_reason", "")

            # If model produced tool calls, execute them
            tool_calls = msg.get("tool_calls", [])
            if finish == "tool_calls" and tool_calls:
                # Cap tool calls — Gemma can emit 100+ identical ones
                tool_calls = tool_calls[:self.MAX_TOOL_CALLS_PER_ROUND]
                # Add assistant message with only the capped calls
                trimmed_msg = dict(msg)
                trimmed_msg["tool_calls"] = tool_calls
                messages.append(trimmed_msg)

                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    if name == "web_search":
                        results = web_search(args.get("query", ""))
                        content = "\n\n".join(
                            f"**{r['title']}**\n{r['url']}\n{r['snippet']}"
                            for r in results
                        )
                        content += (
                            "\n\n[GROUND RULE: Only report facts that appear "
                            "in these results. Do NOT invent details.]"
                        )
                    elif name == "web_fetch":
                        content = web_fetch(args.get("url", ""))
                        content += (
                            "\n\n[GROUND RULE: Only report facts from this page. "
                            "Do NOT invent details.]"
                        )
                    else:
                        content = f"[unknown tool: {name}]"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": content,
                    })
                continue  # next round with tool results

            # No tool calls — final response
            text = msg.get("content", "") or ""
            return AgentResponse(
                text=text,
                input_tokens=total_input,
                output_tokens=total_output,
                cost_usd=0.0,
                model=self.model,
                duration_s=time.time() - start,
                raw=data,
            )

        # Exhausted tool rounds — return whatever we have
        return AgentResponse(
            text="[max tool rounds exceeded]",
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=0.0,
            model=self.model,
            duration_s=time.time() - start,
        )

    def supports_tools(self) -> bool:
        return True

    def backend_name(self) -> str:
        return f"lm-studio:{self.model}"

    def cost_per_1k_input(self) -> float:
        return 0.0

    def cost_per_1k_output(self) -> float:
        return 0.0


# ── Factory ────────────────────────────────────────────────────

def create_backend(
    backend_type: str,
    model: str = "sonnet",
    tools: list[str] | None = None,
    **kwargs: Any,
) -> AgentBackend:
    """Create a backend from a type string + config.

    Extra kwargs are filtered to match each backend's __init__ signature.
    """
    import inspect

    def _filter(cls: type, kw: dict) -> dict:
        sig = inspect.signature(cls.__init__)
        valid = set(sig.parameters.keys()) - {"self"}
        return {k: v for k, v in kw.items() if k in valid and v}

    if backend_type == "claude-cli":
        return ClaudeCliBackend(model=model, tools=tools, **_filter(ClaudeCliBackend, kwargs))
    if backend_type == "anthropic-api":
        return AnthropicApiBackend(model=model, **_filter(AnthropicApiBackend, kwargs))
    if backend_type == "openai-compat":
        return OpenAiCompatBackend(model=model, **_filter(OpenAiCompatBackend, kwargs))
    if backend_type == "lm-studio":
        return LMStudioBackend(model=model, tools=tools, **_filter(LMStudioBackend, kwargs))
    raise ValueError(f"Unknown backend type: {backend_type!r}")
