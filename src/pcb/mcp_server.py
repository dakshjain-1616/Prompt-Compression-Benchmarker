"""MCP server — pcb compression tools for Claude Code, Codex, and any MCP client.

Add to Claude Code:
    claude mcp add pcb -s project -- python -m pcb.mcp_server

Or drop .mcp.json in your project root:
    {"mcpServers": {"pcb": {"type": "stdio", "command": "python", "args": ["-m", "pcb.mcp_server"]}}}
"""

import json

from mcp.server.fastmcp import FastMCP

from pcb.compressors import ALL_COMPRESSORS

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

_COMPRESSOR_DESCRIPTIONS = {
    "smart":            "Two-pass pipeline: sentence selection + word pruning. 60–70% reduction. Default choice.",
    "tfidf":            "TF-IDF sentence scoring. Keeps highest-scoring sentences. ~40% reduction.",
    "selective_context":"Greedy token-budget selection. Good when early sentences are most important. ~55% reduction.",
    "llmlingua":        "Sentence-level coarse pruning. Good for long narratives and transcripts. ~50% reduction.",
    "llmlingua2":       "Word-level stopword pruning. Good for code and technical docs. ~45% reduction.",
    "no_compression":   "Passthrough baseline (0% reduction, ceiling quality).",
}

_TASK_RECOMMENDATIONS = {
    "rag": {
        "best":      "smart",
        "rate":      0.50,
        "reasoning": "smart's query-aware mode keeps only sentences relevant to the question — pass the query param for best results.",
    },
    "summarization": {
        "best":      "smart",
        "rate":      0.45,
        "reasoning": "Two-pass pipeline preserves structural coverage while removing filler and redundant sentences.",
    },
    "coding": {
        "best":      "smart",
        "rate":      0.40,
        "reasoning": "Sentence selection removes boilerplate; word pruning removes stopwords while keeping identifiers and types.",
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

_MODEL_PRICES: dict[str, float] = {
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


@mcp.tool()
def compress_text(
    text: str,
    compressor: str = "smart",
    rate: float = 0.50,
    query: str = "",
) -> str:
    """Compress text to cut LLM token usage by 50–70% before sending to a model.

    Args:
        text:       The text to compress — document, RAG context, codebase, chat history.
        compressor: Algorithm. Default 'smart' (two-pass pipeline, best all-around).
                    Others: tfidf, selective_context, llmlingua, llmlingua2.
        rate:       Fraction of tokens to REMOVE (0.0–1.0). Default 0.50 removes half.
        query:      Optional. If provided, sentences are scored by relevance to this query
                    instead of by corpus importance. Use for RAG contexts — gets 60–70%
                    reduction safely because irrelevant sentences go first.

    Returns:
        JSON: compressed_text, original_tokens, compressed_tokens, tokens_saved, reduction_pct.
    """
    if compressor not in ALL_COMPRESSORS:
        return json.dumps({"error": f"Unknown compressor '{compressor}'. Valid: {list(ALL_COMPRESSORS)}"})
    if not (0.0 < rate < 1.0):
        return json.dumps({"error": "rate must be between 0.0 and 1.0 (exclusive)"})

    c = ALL_COMPRESSORS[compressor]()
    c.initialize()

    kwargs: dict = {"rate": rate}
    if query:
        kwargs["query"] = query

    result = c.compress(text, **kwargs)

    return json.dumps({
        "compressed_text":   result.compressed_text,
        "original_tokens":   result.original_tokens,
        "compressed_tokens": result.compressed_tokens,
        "tokens_saved":      result.original_tokens - result.compressed_tokens,
        "reduction_pct":     round(result.compression_ratio * 100, 1),
        "compressor":        compressor,
        "rate":              rate,
        "query_aware":       bool(query),
    })


@mcp.tool()
def estimate_savings(
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
        model:       Model name for price lookup (e.g. claude-opus-4-7, gpt-4.1).
        daily_calls: How many API calls per day use this kind of context.
        compressor:  Compressor to evaluate. Default 'smart'.
        rate:        Target token removal rate.
        query:       Optional query for query-aware compression.

    Returns:
        JSON with monthly/annual savings, token stats, and breakeven note.
    """
    if compressor not in ALL_COMPRESSORS:
        return json.dumps({"error": f"Unknown compressor '{compressor}'"})

    c = ALL_COMPRESSORS[compressor]()
    c.initialize()

    kwargs: dict = {"rate": rate}
    if query:
        kwargs["query"] = query

    result = c.compress(text, **kwargs)

    tokens_saved = result.original_tokens - result.compressed_tokens
    price = _MODEL_PRICES.get(model.lower().replace("/", "-"), None)
    if price is None:
        price_note = f"Model '{model}' not in price table — using $3/1M as fallback."
        price = 3.0
    else:
        price_note = f"${price:.2f}/1M input tokens."

    monthly = tokens_saved / 1_000_000 * price * daily_calls * 30
    annual = monthly * 12

    return json.dumps({
        "model":                    model,
        "price_per_million_tokens": price,
        "price_note":               price_note,
        "sample_original_tokens":   result.original_tokens,
        "sample_compressed_tokens": result.compressed_tokens,
        "tokens_saved_per_call":    tokens_saved,
        "reduction_pct":            round(result.compression_ratio * 100, 1),
        "daily_calls":              daily_calls,
        "monthly_savings_usd":      round(monthly, 2),
        "annual_savings_usd":       round(annual, 2),
        "compressor":               compressor,
        "rate":                     rate,
        "query_aware":              bool(query),
    })


@mcp.tool()
def recommend(
    task_type: str = "general",
    quality_floor: float = 0.90,
) -> str:
    """Recommend the best compressor and rate for a task type and quality target.

    Args:
        task_type:     One of: rag, summarization, coding, chat, general.
        quality_floor: Minimum quality retention (0.0–1.0). 0.90 = accept ≤10% drop.

    Returns:
        JSON with recommended compressor, rate, expected reduction, and reasoning.
    """
    task_type = task_type.lower().strip()
    rec = _TASK_RECOMMENDATIONS.get(task_type, _TASK_RECOMMENDATIONS["general"])

    rate = rec["rate"]
    if quality_floor >= 0.95:
        rate = max(0.20, rate - 0.10)
    elif quality_floor <= 0.80:
        rate = min(0.65, rate + 0.10)

    return json.dumps({
        "task_type":               task_type,
        "recommended_compressor":  rec["best"],
        "recommended_rate":        rate,
        "expected_reduction_pct":  round(rate * 100, 0),
        "quality_floor_requested": quality_floor,
        "reasoning":               rec["reasoning"],
        "query_aware_tip": (
            "For RAG tasks, pass the user's question as 'query' to compress_text "
            "for 60–70% reduction — only query-relevant sentences are kept."
        ),
        "usage_example": (
            f"Use the pcb compress_text tool with:\n"
            f"  compressor='{rec['best']}', rate={rate}"
            + (", query='<the user question>'" if task_type == "rag" else "")
        ),
    })


@mcp.tool()
def list_compressors() -> str:
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


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
