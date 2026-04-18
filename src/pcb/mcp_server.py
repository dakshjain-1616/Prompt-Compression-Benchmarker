"""MCP server — pcb compression tools for Claude Code, Codex, and any MCP client.

Add to Claude Code:
    claude mcp add pcb -s project -- python -m pcb.mcp_server

Or drop .mcp.json in your project root:
    {"mcpServers": {"pcb": {"type": "stdio", "command": "python", "args": ["-m", "pcb.mcp_server"]}}}
"""

import asyncio
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from pcb.compressors import ALL_COMPRESSORS, BaseCompressor
from pcb.utils.token_counter import count_tokens

mcp = FastMCP(
    "pcb-compression",
    instructions=(
        "Compress long LLM prompts before sending them to expensive models. "
        "Default to compress_text with compressor='smart'. "
        "When you have a specific question the context will answer, pass it as 'query' "
        "for query-aware compression — this gets 60–70% token reduction safely. "
        "Use estimate_savings to quantify the cost impact. "
        "Use recommend when unsure which compressor to pick."
    ),
)

_DEFAULT_MIN_TOKENS = 100

_COMPRESSOR_DESCRIPTIONS = {
    "smart":            "Two-pass pipeline: sentence selection + word pruning. 60–70% reduction. Default choice.",
    "tfidf":            "TF-IDF sentence scoring. Keeps highest-scoring sentences. ~40% reduction.",
    "selective_context":"Greedy token-budget selection. Good when early sentences are most important. ~55% reduction.",
    "llmlingua":        "Sentence-level coarse pruning. Good for long narratives and transcripts. ~50% reduction.",
    "llmlingua2":       "Word-level stopword pruning. Good for code and technical docs. ~45% reduction.",
    "no_compression":   "Passthrough baseline (0% reduction, ceiling quality).",
}

_HEAVY_COMPRESSORS = {"llmlingua", "llmlingua2", "selective_context"}

_TASK_RECOMMENDATIONS = {
    "rag": {
        "best":      "smart",
        "rate":      0.55,
        "reasoning": "smart's query-aware mode keeps only sentences relevant to the question — pass the query param for best results.",
    },
    "summarization": {
        "best":      "smart",
        "rate":      0.45,
        "reasoning": "Two-pass pipeline preserves structural coverage while removing filler and redundant sentences.",
    },
    "coding": {
        "best":      "llmlingua2",
        "rate":      0.40,
        "reasoning": "Word-level pruning removes stopwords while preserving identifiers, types, and operators. Falls back to a heuristic if the llmlingua library isn't installed.",
    },
    "chat": {
        "best":      "smart",
        "rate":      0.45,
        "reasoning": "Two-pass pipeline handles mixed content well. Dedup pass removes repeated context from long threads.",
    },
    "general": {
        "best":      "smart",
        "rate":      0.45,
        "reasoning": "smart is the safe default for any content type.",
    },
}

_FALLBACK_PRICES: dict[str, float] = {
    "claude-opus-4-7":    15.00,
    "claude-opus-4-6":    15.00,
    "claude-sonnet-4-6":   3.00,
    "claude-sonnet-4-5":   3.00,
    "claude-haiku-4-5":    0.80,
    "gpt-5":               5.00,
    "gpt-5.4":             5.00,
    "gpt-4.1":             2.00,
    "gpt-4.1-mini":        0.40,
    "gpt-4o":              2.50,
    "o3":                 10.00,
    "o4-mini":             1.10,
    "codex-mini-latest":   1.50,
    "gemini-2.5-pro":      1.25,
    "gemini-2.5-flash":    0.15,
    "deepseek-v3.2":       0.27,
}


def _load_prices() -> tuple[dict[str, float], Optional[str]]:
    """Load the model price table, preferring the JSON data file.

    Falls back to the hardcoded dict if the JSON is missing or malformed so a
    bad deploy can't take the MCP offline.
    """
    try:
        path = Path(__file__).parent / "data" / "model_prices.json"
        data = json.loads(path.read_text())
        prices = {k.lower(): float(v) for k, v in data.get("prices", {}).items()}
        if not prices:
            return _FALLBACK_PRICES, None
        return prices, data.get("_last_updated")
    except Exception:
        return _FALLBACK_PRICES, None


_MODEL_PRICES, _PRICES_UPDATED = _load_prices()

_DATE_SUFFIX = re.compile(r"-\d{8}$")


def _normalize_model_name(model: str) -> str:
    """Normalize a model name for price lookup.

    Handles three common input shapes:
      - ``claude-opus-4-7``              → ``claude-opus-4-7``
      - ``anthropic/claude-opus-4-7``    → ``claude-opus-4-7``     (strip vendor prefix)
      - ``claude-opus-4-7-20260101``     → ``claude-opus-4-7``     (strip date snapshot)
    """
    name = model.lower().strip()
    if "/" in name:
        name = name.split("/", 1)[1]
    name = _DATE_SUFFIX.sub("", name)
    return name


_COMPRESSOR_CACHE: dict[str, BaseCompressor] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class _SessionStats:
    """Cumulative compression stats for the lifetime of the MCP server process."""
    started_at:             float = field(default_factory=time.time)
    calls:                  int = 0
    original_tokens_total:  int = 0
    compressed_tokens_total: int = 0
    by_compressor:          dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def tokens_saved_total(self) -> int:
        return self.original_tokens_total - self.compressed_tokens_total

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens_total == 0:
            return 0.0
        return 100.0 * self.tokens_saved_total / self.original_tokens_total

    def snapshot(self) -> dict:
        return {
            "calls":                  self.calls,
            "original_tokens_total":  self.original_tokens_total,
            "compressed_tokens_total": self.compressed_tokens_total,
            "tokens_saved_total":     self.tokens_saved_total,
            "reduction_pct":          round(self.reduction_pct, 2),
            "uptime_seconds":         round(time.time() - self.started_at, 1),
            "by_compressor":          {k: dict(v) for k, v in self.by_compressor.items()},
        }


_SESSION = _SessionStats()
_SESSION_LOCK = threading.Lock()


def _record_call(compressor: str, original_tokens: int, compressed_tokens: int) -> dict:
    """Update the running session stats and return the latest snapshot."""
    with _SESSION_LOCK:
        _SESSION.calls += 1
        _SESSION.original_tokens_total += original_tokens
        _SESSION.compressed_tokens_total += compressed_tokens
        bucket = _SESSION.by_compressor.setdefault(
            compressor, {"calls": 0, "original_tokens": 0, "compressed_tokens": 0}
        )
        bucket["calls"] += 1
        bucket["original_tokens"] += original_tokens
        bucket["compressed_tokens"] += compressed_tokens
        return _SESSION.snapshot()


def _reset_session() -> dict:
    """Clear cumulative stats and return the fresh (zeroed) snapshot."""
    global _SESSION
    with _SESSION_LOCK:
        _SESSION = _SessionStats()
        return _SESSION.snapshot()


def _get_compressor(name: str) -> BaseCompressor:
    """Return a cached, initialized compressor instance.

    Initializing llmlingua/selective_context loads transformer models from disk,
    so we amortize that cost across all future calls by keeping a single instance
    per compressor name for the lifetime of the MCP server process.
    """
    with _CACHE_LOCK:
        c = _COMPRESSOR_CACHE.get(name)
        if c is None:
            c = ALL_COMPRESSORS[name]()
            c.initialize()
            _COMPRESSOR_CACHE[name] = c
        return c


def _short_text_passthrough(text: str, min_tokens: int, compressor: str) -> Optional[dict]:
    """If the text is shorter than min_tokens, return a passthrough payload.

    Compressing very short text has negative ROI: the overhead of scoring
    dominates and the risk of dropping meaningful tokens is high.
    """
    n = count_tokens(text)
    if n < min_tokens:
        session = _record_call(compressor, n, n)
        return {
            "compressed_text":      text,
            "original_tokens":      n,
            "compressed_tokens":    n,
            "tokens_saved":         0,
            "reduction_pct":        0.0,
            "skipped_short_text":   True,
            "min_tokens_threshold": min_tokens,
            "session":              session,
        }
    return None


def _validate_compress_args(compressor: str, rate: float) -> Optional[dict]:
    """Return an error payload if args are invalid, else None.

    Shared by the short-text passthrough and the full compression path so bogus
    compressor names or out-of-range rates can't sneak through a small input.
    """
    if compressor not in ALL_COMPRESSORS:
        return {"error": f"Unknown compressor '{compressor}'. Valid: {list(ALL_COMPRESSORS)}"}
    if not (0.0 <= rate < 1.0):
        return {"error": "rate must be in [0.0, 1.0) — 1.0 would remove all tokens."}
    return None


def _run_compress(
    text: str,
    compressor: str,
    rate: float,
    query: str,
) -> dict:
    """Synchronous core for compress_text. Safe to call from asyncio.to_thread."""
    err = _validate_compress_args(compressor, rate)
    if err is not None:
        return err

    if rate == 0.0:
        n = count_tokens(text)
        session = _record_call(compressor, n, n)
        return {
            "compressed_text":   text,
            "original_tokens":   n,
            "compressed_tokens": n,
            "tokens_saved":      0,
            "reduction_pct":     0.0,
            "compressor":        compressor,
            "rate":              0.0,
            "query_aware":       bool(query),
            "noop":              True,
            "session":           session,
        }

    c = _get_compressor(compressor)

    kwargs: dict = {"rate": rate}
    if query:
        kwargs["query"] = query

    result = c.compress(text, **kwargs)
    fallback = bool(result.metadata.get("fallback_active", False))
    session = _record_call(compressor, result.original_tokens, result.compressed_tokens)

    return {
        "compressed_text":           result.compressed_text,
        "original_tokens":           result.original_tokens,
        "compressed_tokens":         result.compressed_tokens,
        "tokens_saved":              result.original_tokens - result.compressed_tokens,
        "reduction_pct":             round(result.compression_ratio * 100, 1),
        "compressor":                compressor,
        "rate":                      rate,
        "query_aware":               bool(query),
        "used_heuristic_fallback":   fallback,
        "session":                   session,
    }


@mcp.tool()
async def compress_text(
    text: str,
    compressor: str = "smart",
    rate: float = 0.50,
    query: str = "",
    min_tokens: int = _DEFAULT_MIN_TOKENS,
) -> str:
    """Compress text to cut LLM token usage by 50–70% before sending to a model.

    Args:
        text:       The text to compress — document, RAG context, codebase, chat history.
        compressor: Algorithm. Default 'smart' (two-pass pipeline, best all-around).
                    Others: tfidf, selective_context, llmlingua, llmlingua2.
        rate:       Fraction of tokens to REMOVE in [0.0, 1.0). 0.0 is a no-op; 1.0 is rejected.
        query:      Optional. If provided, sentences are scored by relevance to this query
                    instead of by corpus importance. Use for RAG contexts — gets 60–70%
                    reduction safely because irrelevant sentences go first.
        min_tokens: Skip compression when the input is shorter than this (default 100).
                    Short prompts gain little from compression and risk losing content.

    Returns:
        JSON: compressed_text, original_tokens, compressed_tokens, tokens_saved, reduction_pct,
        plus used_heuristic_fallback (True if the real model library wasn't available),
        and skipped_short_text when the input was under the min_tokens threshold.
    """
    err = _validate_compress_args(compressor, rate)
    if err is not None:
        return json.dumps(err)

    passthrough = _short_text_passthrough(text, min_tokens, compressor)
    if passthrough is not None:
        passthrough.update({"compressor": compressor, "rate": rate, "query_aware": bool(query)})
        return json.dumps(passthrough)

    payload = await asyncio.to_thread(_run_compress, text, compressor, rate, query)
    return json.dumps(payload)


def _run_estimate(
    text: str,
    model: str,
    daily_calls: int,
    compressor: str,
    rate: float,
    query: str,
) -> dict:
    """Synchronous core for estimate_savings."""
    compressed = _run_compress(text, compressor, rate, query)
    if "error" in compressed:
        return compressed

    tokens_saved = compressed["tokens_saved"]

    normalized = _normalize_model_name(model)
    price = _MODEL_PRICES.get(normalized)
    if price is None:
        price_source = "fallback"
        price_note = (
            f"Model '{model}' not in price table — using $3/1M as fallback. "
            f"Add it to src/pcb/data/model_prices.json for accurate estimates."
        )
        price = 3.0
    else:
        price_source = "table"
        price_note = f"${price:.2f}/1M input tokens (table updated {_PRICES_UPDATED or 'n/a'})."

    tokens_saved_daily = tokens_saved * daily_calls
    tokens_saved_monthly = tokens_saved_daily * 30
    tokens_saved_annually = tokens_saved_monthly * 12

    monthly = tokens_saved_monthly / 1_000_000 * price
    annual = monthly * 12

    session = compressed.get("session", _SESSION.snapshot())
    lifetime_saved = session.get("tokens_saved_total", 0)
    lifetime_savings_usd = round(lifetime_saved / 1_000_000 * price, 4)

    return {
        "model":                    model,
        "normalized_model":         normalized,
        "price_per_million_tokens": price,
        "price_source":             price_source,
        "price_note":               price_note,
        "sample_original_tokens":   compressed["original_tokens"],
        "sample_compressed_tokens": compressed["compressed_tokens"],
        "tokens_saved_per_call":    tokens_saved,
        "reduction_pct":            compressed["reduction_pct"],
        "daily_calls":              daily_calls,
        "tokens_saved_daily":       tokens_saved_daily,
        "tokens_saved_monthly":     tokens_saved_monthly,
        "tokens_saved_annually":    tokens_saved_annually,
        "monthly_savings_usd":      round(monthly, 2),
        "annual_savings_usd":       round(annual, 2),
        "compressor":               compressor,
        "rate":                     rate,
        "query_aware":              bool(query),
        "used_heuristic_fallback":  compressed.get("used_heuristic_fallback", False),
        "session":                  session,
        "session_savings_usd":      lifetime_savings_usd,
    }


@mcp.tool()
async def estimate_savings(
    text: str,
    model: str = "claude-sonnet-4-6",
    daily_calls: int = 1000,
    compressor: str = "smart",
    rate: float = 0.50,
    query: str = "",
) -> str:
    """Estimate monthly API cost savings from compressing prompts like this one.

    Args:
        text:        A representative sample of text you'd send to the model.
        model:       Model name for price lookup. Accepts bare (``claude-opus-4-7``),
                     vendor-prefixed (``anthropic/claude-opus-4-7``), or dated
                     (``claude-opus-4-7-20260101``) forms.
        daily_calls: How many API calls per day use this kind of context.
        compressor:  Compressor to evaluate. Default 'smart'.
        rate:        Target token removal rate in [0.0, 1.0).
        query:       Optional query for query-aware compression.

    Returns:
        JSON with USD savings (monthly_savings_usd, annual_savings_usd) AND raw token
        savings (tokens_saved_per_call, tokens_saved_daily, tokens_saved_monthly,
        tokens_saved_annually) — the token axis matters when pricing changes or you're
        capacity-planning a context window. Plus a price_source field indicating
        whether the model was found in the price table or fell back to $3/1M.
    """
    payload = await asyncio.to_thread(
        _run_estimate, text, model, daily_calls, compressor, rate, query,
    )
    return json.dumps(payload)


def _run_recommend(task_type: str, quality_floor: float, latency_sensitive: bool) -> dict:
    """Synchronous core for recommend."""
    task_type = task_type.lower().strip()
    rec = _TASK_RECOMMENDATIONS.get(task_type, _TASK_RECOMMENDATIONS["general"])
    best = rec["best"]
    rate = rec["rate"]
    reasoning = rec["reasoning"]

    if latency_sensitive and best in _HEAVY_COMPRESSORS:
        best = "tfidf" if quality_floor >= 0.95 else "smart"
        reasoning = (
            f"{reasoning} Switched to '{best}' because latency_sensitive=True "
            f"and the original pick loads a transformer model on first call."
        )

    if not latency_sensitive and task_type == "general" and quality_floor >= 0.95:
        best = "tfidf"
        rate = min(rate, 0.35)
        reasoning = "For quality-floor ≥ 0.95 on general content, tfidf is the most predictable choice — deterministic scoring, no model dependency."

    if quality_floor >= 0.95:
        rate = max(0.20, rate - 0.10)
    elif quality_floor <= 0.80:
        rate = min(0.65, rate + 0.10)

    return {
        "task_type":               task_type,
        "recommended_compressor":  best,
        "recommended_rate":        round(rate, 3),
        "expected_reduction_pct":  round(rate * 100, 0),
        "quality_floor_requested": quality_floor,
        "latency_sensitive":       latency_sensitive,
        "reasoning":               reasoning,
        "query_aware_tip": (
            "For RAG tasks, pass the user's question as 'query' to compress_text "
            "for 60–70% reduction — only query-relevant sentences are kept."
        ),
        "usage_example": (
            f"Call compress_text with compressor='{best}', rate={round(rate, 3)}"
            + (", query='<the user question>'" if task_type == "rag" else "")
        ),
    }


@mcp.tool()
async def recommend(
    task_type: str = "general",
    quality_floor: float = 0.90,
    latency_sensitive: bool = False,
) -> str:
    """Recommend the best compressor and rate for a task type and quality target.

    Args:
        task_type:          One of: rag, summarization, coding, chat, general.
        quality_floor:      Minimum quality retention (0.0–1.0). 0.90 = accept ≤10% drop.
        latency_sensitive:  If true, avoid compressors that load transformer models on
                            first call (llmlingua/llmlingua2/selective_context).

    Returns:
        JSON with recommended compressor, rate, expected reduction, and reasoning.
    """
    payload = await asyncio.to_thread(_run_recommend, task_type, quality_floor, latency_sensitive)
    return json.dumps(payload)


@mcp.tool()
async def list_compressors() -> str:
    """List all available compression algorithms with descriptions and typical reduction rates.

    Returns:
        JSON array of compressor info objects, ordered from most to least powerful.
    """
    order = ["smart", "selective_context", "llmlingua", "llmlingua2", "tfidf", "no_compression"]
    return json.dumps([
        {"name": name, "description": _COMPRESSOR_DESCRIPTIONS.get(name, "")}
        for name in order
        if name in ALL_COMPRESSORS
    ])


@mcp.tool()
async def session_stats(
    reset: bool = False,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Show cumulative tokens saved across every compression call in this MCP session.

    Every successful compress_text / estimate_savings call feeds a running counter.
    This tool returns the current totals plus a per-compressor breakdown so you can
    see at a glance how much value the MCP has delivered.

    Args:
        reset: If true, zero the counters after returning the snapshot.
        model: Model whose price is used to convert tokens_saved → USD. Accepts the
               same formats as estimate_savings (vendor-prefixed / dated).

    Returns:
        JSON with calls, original_tokens_total, compressed_tokens_total,
        tokens_saved_total, reduction_pct, uptime_seconds, per-compressor breakdown,
        and a USD conversion for the chosen model.
    """
    with _SESSION_LOCK:
        snapshot = _SESSION.snapshot()

    normalized = _normalize_model_name(model)
    price = _MODEL_PRICES.get(normalized)
    if price is None:
        price = 3.0
        price_source = "fallback"
    else:
        price_source = "table"

    snapshot["model"] = model
    snapshot["normalized_model"] = normalized
    snapshot["price_per_million_tokens"] = price
    snapshot["price_source"] = price_source
    snapshot["session_savings_usd"] = round(
        snapshot["tokens_saved_total"] / 1_000_000 * price, 4
    )

    if reset:
        _reset_session()
        snapshot["reset"] = True

    return json.dumps(snapshot)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
