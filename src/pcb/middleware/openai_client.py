"""Drop-in OpenAI client wrapper with automatic prompt compression.

Works with both Chat Completions API and the Responses API (Codex / o-series).
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import tiktoken

from pcb.compressors import ALL_COMPRESSORS, BaseCompressor
from pcb.middleware.anthropic_client import CompressionStats, _compress_content

_enc = tiktoken.get_encoding("cl100k_base")
_MIN_TOKENS = 100


class _CompressingCompletions:
    """Wraps client.chat.completions to intercept .create() calls."""

    def __init__(self, ns: Any, compressor: BaseCompressor, rate: float,
                 stats: CompressionStats, compress_roles: tuple, verbose: bool) -> None:
        self._ns = ns
        self._compressor = compressor
        self._rate = rate
        self._stats = stats
        self._compress_roles = compress_roles
        self._verbose = verbose

    def create(self, *, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        compressed, call_orig, call_comp = [], 0, 0

        for msg in messages:
            if msg.get("role") in self._compress_roles:
                new_content, orig, comp = _compress_content(
                    msg.get("content", ""), self._compressor, self._rate
                )
                compressed.append({**msg, "content": new_content})
                call_orig += orig
                call_comp += comp
            else:
                compressed.append(msg)

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

        return self._ns.create(messages=compressed, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)


class _CompressingChat:
    def __init__(self, chat_ns: Any, compressor: BaseCompressor, rate: float,
                 stats: CompressionStats, compress_roles: tuple, verbose: bool) -> None:
        self.completions = _CompressingCompletions(
            chat_ns.completions, compressor, rate, stats, compress_roles, verbose
        )
        self._ns = chat_ns

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)


class _CompressingResponses:
    """Wraps client.responses for the Responses API (Codex / o-series)."""

    def __init__(self, ns: Any, compressor: BaseCompressor, rate: float,
                 stats: CompressionStats, verbose: bool) -> None:
        self._ns = ns
        self._compressor = compressor
        self._rate = rate
        self._stats = stats
        self._verbose = verbose

    def create(self, *, input: Any = None, **kwargs: Any) -> Any:
        call_orig = call_comp = 0

        if isinstance(input, str):
            n = len(_enc.encode(input))
            if n >= _MIN_TOKENS:
                result = self._compressor.compress(input, rate=self._rate)
                call_orig, call_comp = result.original_tokens, result.compressed_tokens
                input = result.compressed_text
        elif isinstance(input, list):
            new_input = []
            for item in input:
                if isinstance(item, dict) and item.get("role") == "user":
                    new_content, orig, comp = _compress_content(
                        item.get("content", ""), self._compressor, self._rate
                    )
                    new_input.append({**item, "content": new_content})
                    call_orig += orig
                    call_comp += comp
                else:
                    new_input.append(item)
            input = new_input

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

        return self._ns.create(input=input, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ns, name)


class CompressingOpenAI:
    """Drop-in replacement for openai.OpenAI() that compresses messages before sending.

    Works with Chat Completions and the Responses API (for Codex / o-series models).

    Args:
        compressor: "tfidf", "selective_context", "llmlingua", "llmlingua2".
        rate: Fraction of tokens to remove (0.0–1.0). Default 0.45.
        compress_roles: Which roles to compress in Chat Completions. Default ("user",).
        verbose: Print token savings to stderr on each call.
        **kwargs: Passed to openai.OpenAI().

    Example:
        client = CompressingOpenAI(compressor="tfidf", rate=0.4)
        # Chat Completions
        response = client.chat.completions.create(
            model="gpt-4.1", messages=[{"role": "user", "content": "long context..."}]
        )
        # Responses API (Codex)
        response = client.responses.create(
            model="codex-mini-latest", input="long context..."
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
            import openai
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        if compressor not in ALL_COMPRESSORS:
            raise ValueError(f"Unknown compressor '{compressor}'. Valid: {list(ALL_COMPRESSORS)}")

        self._client = openai.OpenAI(**kwargs)
        self._compressor_instance = ALL_COMPRESSORS[compressor]()
        self._compressor_instance.initialize()
        self._rate = rate
        self.stats = CompressionStats()
        self.compressor_name = compressor

        self.chat = _CompressingChat(
            self._client.chat, self._compressor_instance, rate, self.stats, compress_roles, verbose
        )
        self.responses = _CompressingResponses(
            self._client.responses, self._compressor_instance, rate, self.stats, verbose
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
