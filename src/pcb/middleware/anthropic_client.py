"""Drop-in Anthropic client wrapper with automatic prompt compression."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import tiktoken

from pcb.compressors import ALL_COMPRESSORS, BaseCompressor

_enc = tiktoken.get_encoding("cl100k_base")
_MIN_TOKENS = 100  # skip compression for very short content


@dataclass
class CompressionStats:
    calls: int = 0
    original_tokens: int = 0
    compressed_tokens: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.compressed_tokens

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.tokens_saved / self.original_tokens

    def monthly_savings_usd(self, price_per_million: float, daily_calls_estimate: int = 0) -> float:
        if self.calls == 0 or daily_calls_estimate == 0:
            return 0.0
        avg_saved = self.tokens_saved / self.calls
        return avg_saved / 1_000_000 * price_per_million * daily_calls_estimate * 30

    def __repr__(self) -> str:
        return (
            f"CompressionStats(calls={self.calls}, "
            f"tokens_saved={self.tokens_saved:,}, "
            f"reduction={self.reduction_pct:.1f}%)"
        )


def _compress_content(content: Any, compressor: BaseCompressor, rate: float) -> tuple[Any, int, int]:
    """Compress a message's content field. Returns (new_content, orig_tokens, comp_tokens)."""
    if isinstance(content, str):
        n = len(_enc.encode(content))
        if n < _MIN_TOKENS:
            return content, n, n
        result = compressor.compress(content, rate=rate)
        return result.compressed_text, result.original_tokens, result.compressed_tokens

    if isinstance(content, list):
        new_blocks, orig_total, comp_total = [], 0, 0
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block["text"]
                n = len(_enc.encode(text))
                if n >= _MIN_TOKENS:
                    result = compressor.compress(text, rate=rate)
                    new_blocks.append({**block, "text": result.compressed_text})
                    orig_total += result.original_tokens
                    comp_total += result.compressed_tokens
                else:
                    new_blocks.append(block)
                    orig_total += n
                    comp_total += n
            else:
                new_blocks.append(block)
        return new_blocks, orig_total, comp_total

    return content, 0, 0


class _CompressingMessages:
    """Wraps client.messages to intercept .create() calls."""

    def __init__(self, messages_ns: Any, compressor: BaseCompressor, rate: float,
                 stats: CompressionStats, compress_roles: tuple, verbose: bool) -> None:
        self._ns = messages_ns
        self._compressor = compressor
        self._rate = rate
        self._stats = stats
        self._compress_roles = compress_roles
        self._verbose = verbose

    def create(self, *, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        compressed_messages = []
        call_orig = call_comp = 0

        for msg in messages:
            role = msg.get("role", "")
            if role in self._compress_roles:
                new_content, orig, comp = _compress_content(
                    msg.get("content", ""), self._compressor, self._rate
                )
                compressed_messages.append({**msg, "content": new_content})
                call_orig += orig
                call_comp += comp
            else:
                compressed_messages.append(msg)

        self._stats.calls += 1
        self._stats.original_tokens += call_orig
        self._stats.compressed_tokens += call_comp

        if self._verbose and call_orig > 0:
            saved = call_orig - call_comp
            print(
                f"[pcb] compressed {call_orig}→{call_comp} tokens "
                f"({saved/call_orig*100:.1f}% reduction)",
                file=sys.stderr,
            )

        return self._ns.create(messages=compressed_messages, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)


class CompressingAnthropic:
    """Drop-in replacement for anthropic.Anthropic() that compresses messages before sending.

    Args:
        compressor: Name of compressor — "tfidf", "selective_context", "llmlingua", "llmlingua2".
        rate: Fraction of tokens to remove (0.0–1.0). Default 0.45.
        compress_roles: Which message roles to compress. Default ("user",).
        verbose: Print token savings to stderr on each call.
        **kwargs: Passed directly to anthropic.Anthropic().

    Example:
        client = CompressingAnthropic(compressor="llmlingua2", rate=0.45)
        response = client.messages.create(
            model="claude-opus-4-7",
            messages=[{"role": "user", "content": "long document..."}],
            max_tokens=1024,
        )
        print(client.stats)
    """

    def __init__(
        self,
        compressor: str = "tfidf",
        rate: float = 0.45,
        compress_roles: tuple = ("user",),
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        if compressor not in ALL_COMPRESSORS:
            raise ValueError(f"Unknown compressor '{compressor}'. Valid: {list(ALL_COMPRESSORS)}")

        self._client = anthropic.Anthropic(**kwargs)
        self._compressor_instance = ALL_COMPRESSORS[compressor]()
        self._compressor_instance.initialize()
        self._rate = rate
        self.stats = CompressionStats()
        self.compressor_name = compressor

        self.messages = _CompressingMessages(
            self._client.messages,
            self._compressor_instance,
            rate,
            self.stats,
            compress_roles,
            verbose,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
