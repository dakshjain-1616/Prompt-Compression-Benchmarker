"""Core unit tests — no API keys required."""

import pytest
from pcb.utils.token_counter import count_tokens
from pcb.compressors import ALL_COMPRESSORS
from pcb.middleware.anthropic_client import _compress_content, CompressionStats


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


def test_token_counter():
    n = count_tokens(SAMPLE_TEXT)
    assert n > 50


@pytest.mark.parametrize("name", ["no_compression", "tfidf"])
def test_compressor_reduces_tokens(name):
    c = ALL_COMPRESSORS[name]()
    c.initialize()
    result = c.compress(SAMPLE_TEXT, rate=0.4)
    assert hasattr(result, "compressed_text")
    assert len(result.compressed_text) > 0
    if name != "no_compression":
        assert count_tokens(result.compressed_text) < count_tokens(SAMPLE_TEXT)


def test_compression_stats():
    stats = CompressionStats()
    stats.calls = 100
    stats.original_tokens = 45000
    stats.compressed_tokens = 25000
    assert stats.tokens_saved == 20000
    assert 40 < stats.reduction_pct < 50
    monthly = stats.monthly_savings_usd(price_per_million=15.0, daily_calls_estimate=2000)
    assert monthly > 0


def test_compress_content_string():
    c = ALL_COMPRESSORS["tfidf"]()
    c.initialize()
    compressed, orig, comp = _compress_content(SAMPLE_TEXT, c, rate=0.4)
    assert isinstance(compressed, str)
    assert orig > comp


def test_compress_content_short_text_passthrough():
    c = ALL_COMPRESSORS["tfidf"]()
    c.initialize()
    short = "Hello world."
    compressed, orig, comp = _compress_content(short, c, rate=0.4)
    assert compressed == short
    assert orig == comp
