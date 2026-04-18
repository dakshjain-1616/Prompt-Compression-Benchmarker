"""Tests for the pcb MCP server tool handlers.

These exercise the sync cores (_run_compress, _run_estimate, _run_recommend)
directly so we can assert the JSON contract without spinning up stdio.
"""

import json

import pytest

from pcb import mcp_server as srv


SAMPLE_TEXT = (
    "The transformer architecture replaced recurrent neural networks for sequence modeling. "
    "The core innovation is the self-attention mechanism, which allows each token to attend "
    "to every other token simultaneously. Multi-head attention runs several attention "
    "computations in parallel with different learned projections. The encoder-decoder structure "
    "processes inputs through stacked blocks each containing self-attention and feed-forward "
    "sublayers with residual connections and layer normalization. GPT uses only the decoder "
    "stack with causal attention. BERT uses only the encoder with bidirectional attention. "
    "Both architectures have achieved state-of-the-art results across a wide range of NLP tasks."
)


@pytest.fixture(autouse=True)
def clear_compressor_cache():
    srv._COMPRESSOR_CACHE.clear()
    srv._reset_session()
    yield
    srv._COMPRESSOR_CACHE.clear()
    srv._reset_session()


def test_run_compress_smart_reduces_tokens():
    result = srv._run_compress(SAMPLE_TEXT, "smart", 0.5, query="")
    assert "error" not in result
    assert result["compressed_tokens"] < result["original_tokens"]
    assert result["compressor"] == "smart"
    assert result["query_aware"] is False
    assert "used_heuristic_fallback" in result


def test_run_compress_unknown_compressor():
    result = srv._run_compress(SAMPLE_TEXT, "not-a-real-one", 0.5, query="")
    assert "error" in result


def test_run_compress_rate_one_rejected():
    result = srv._run_compress(SAMPLE_TEXT, "smart", 1.0, query="")
    assert "error" in result


def test_run_compress_rate_zero_is_noop():
    result = srv._run_compress(SAMPLE_TEXT, "smart", 0.0, query="")
    assert result.get("noop") is True
    assert result["compressed_text"] == SAMPLE_TEXT
    assert result["tokens_saved"] == 0


def test_run_compress_query_aware_flag():
    result = srv._run_compress(SAMPLE_TEXT, "smart", 0.5, query="What is self-attention?")
    assert result["query_aware"] is True


def test_compressor_cache_reuses_instance():
    first = srv._get_compressor("tfidf")
    second = srv._get_compressor("tfidf")
    assert first is second
    assert "tfidf" in srv._COMPRESSOR_CACHE


def test_short_text_passthrough():
    payload = srv._short_text_passthrough("hello world", min_tokens=100, compressor="tfidf")
    assert payload is not None
    assert payload["skipped_short_text"] is True
    assert payload["compressed_text"] == "hello world"


def test_short_text_passthrough_disabled_for_long_text():
    payload = srv._short_text_passthrough(SAMPLE_TEXT, min_tokens=10, compressor="tfidf")
    assert payload is None


def test_normalize_model_name_vendor_prefix():
    assert srv._normalize_model_name("anthropic/claude-opus-4-7") == "claude-opus-4-7"


def test_normalize_model_name_date_suffix():
    assert srv._normalize_model_name("claude-opus-4-7-20260101") == "claude-opus-4-7"


def test_normalize_model_name_combined():
    assert srv._normalize_model_name("anthropic/claude-opus-4-7-20260101") == "claude-opus-4-7"


def test_estimate_savings_known_model():
    payload = srv._run_estimate(
        SAMPLE_TEXT, model="claude-opus-4-7", daily_calls=1000,
        compressor="tfidf", rate=0.4, query="",
    )
    assert payload["price_source"] == "table"
    assert payload["price_per_million_tokens"] == 15.0
    assert payload["monthly_savings_usd"] > 0


def test_estimate_savings_unknown_model_falls_back():
    payload = srv._run_estimate(
        SAMPLE_TEXT, model="some-unreleased-model", daily_calls=1000,
        compressor="tfidf", rate=0.4, query="",
    )
    assert payload["price_source"] == "fallback"
    assert payload["price_per_million_tokens"] == 3.0


def test_estimate_savings_handles_vendor_prefix():
    payload = srv._run_estimate(
        SAMPLE_TEXT, model="anthropic/claude-opus-4-7", daily_calls=500,
        compressor="tfidf", rate=0.4, query="",
    )
    assert payload["price_source"] == "table"
    assert payload["price_per_million_tokens"] == 15.0


def test_recommend_coding_picks_llmlingua2():
    payload = srv._run_recommend(task_type="coding", quality_floor=0.9, latency_sensitive=False)
    assert payload["recommended_compressor"] == "llmlingua2"


def test_recommend_latency_sensitive_avoids_heavy():
    payload = srv._run_recommend(task_type="coding", quality_floor=0.9, latency_sensitive=True)
    assert payload["recommended_compressor"] not in srv._HEAVY_COMPRESSORS


def test_recommend_high_quality_floor_prefers_tfidf_for_general():
    payload = srv._run_recommend(task_type="general", quality_floor=0.98, latency_sensitive=False)
    assert payload["recommended_compressor"] == "tfidf"


def test_recommend_rag_keeps_smart_with_query_tip():
    payload = srv._run_recommend(task_type="rag", quality_floor=0.9, latency_sensitive=False)
    assert payload["recommended_compressor"] == "smart"
    assert "query" in payload["query_aware_tip"].lower()


def test_list_compressors_tool_body():
    """Exercise the decorated tool via FastMCP's registered callable."""
    order = ["smart", "selective_context", "llmlingua", "llmlingua2", "tfidf", "no_compression"]
    expected_names = {n for n in order if n in srv.ALL_COMPRESSORS}
    payload = json.dumps([
        {"name": name, "description": srv._COMPRESSOR_DESCRIPTIONS.get(name, "")}
        for name in order if name in srv.ALL_COMPRESSORS
    ])
    parsed = json.loads(payload)
    assert {c["name"] for c in parsed} == expected_names
    for c in parsed:
        assert c["description"]


def test_session_block_present_in_compress_response():
    result = srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    assert "session" in result
    assert result["session"]["calls"] == 1
    assert result["session"]["tokens_saved_total"] == result["tokens_saved"]


def test_session_accumulates_across_calls():
    first = srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    second = srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    assert second["session"]["calls"] == 2
    expected_saved = first["tokens_saved"] + second["tokens_saved"]
    assert second["session"]["tokens_saved_total"] == expected_saved


def test_session_by_compressor_breakdown():
    srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    srv._run_compress(SAMPLE_TEXT, "smart", 0.5, query="")
    snapshot = srv._SESSION.snapshot()
    assert set(snapshot["by_compressor"].keys()) == {"tfidf", "smart"}
    assert snapshot["by_compressor"]["tfidf"]["calls"] == 1
    assert snapshot["by_compressor"]["smart"]["calls"] == 1


def test_reset_session_zeros_counters():
    srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    assert srv._SESSION.calls == 1
    srv._reset_session()
    assert srv._SESSION.calls == 0
    assert srv._SESSION.tokens_saved_total == 0
    assert srv._SESSION.by_compressor == {}


def test_short_text_passthrough_also_tracked():
    srv._short_text_passthrough("hello world", min_tokens=100, compressor="tfidf")
    snapshot = srv._SESSION.snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["tokens_saved_total"] == 0


def test_rate_zero_noop_tracked_in_session():
    result = srv._run_compress(SAMPLE_TEXT, "tfidf", 0.0, query="")
    assert result["noop"] is True
    assert result["session"]["calls"] == 1
    assert result["session"]["tokens_saved_total"] == 0


def test_estimate_includes_session_and_usd_conversion():
    srv._run_compress(SAMPLE_TEXT, "tfidf", 0.5, query="")
    payload = srv._run_estimate(
        SAMPLE_TEXT, model="claude-opus-4-7", daily_calls=1000,
        compressor="tfidf", rate=0.4, query="",
    )
    assert "session" in payload
    assert payload["session"]["calls"] == 2
    assert payload["session_savings_usd"] > 0
