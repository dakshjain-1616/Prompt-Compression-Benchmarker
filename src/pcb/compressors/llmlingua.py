"""LLMLingua compressor for prompt compression."""

import warnings
from typing import Any, Optional

from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.utils.token_counter import TokenCounter


class LLMLinguaCompressor(BaseCompressor):
    """Compressor using LLMLingua algorithm."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the LLMLingua compressor.
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: Model to use for compression (default: "lgaalves/gpt2-dolly-v2")
                - rate: Target compression rate (default: 0.5)
        """
        super().__init__(name="llmlingua", config=config)
        self.token_counter = TokenCounter()
        self.model_name = self.config.get("model_name", "lgaalves/gpt2-dolly-v2")
        self.rate = self.config.get("rate", 0.5)
        self._compressor = None
        self._fallback_active = False

    def initialize(self) -> None:
        """Initialize the LLMLingua compressor."""
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name=self.model_name,
                device_map="cpu"
            )
            self._is_initialized = True
        except ImportError:
            # Expected when the optional [llmlingua] extras aren't installed.
            self._fallback_active = True
            self._is_initialized = True
        except Exception as e:
            # Something else went wrong (CUDA OOM, disk full, broken checkpoint).
            # Surface it — silent fallback hides real bugs as "heuristic mode".
            warnings.warn(f"LLMLingua init failed, falling back to heuristic: {e!r}")
            self._fallback_active = True
            self._is_initialized = True

    def _simple_compress(self, text: str, rate: float) -> str:
        """Fallback: coarse sentence-level pruning (keep first+last+evenly-spaced middle).

        Mimics LLMLingua's coarse-to-fine strategy at sentence granularity.
        """
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        num_to_keep = max(1, int(len(sentences) * (1.0 - rate)))
        if num_to_keep >= len(sentences):
            return text

        if num_to_keep == 1:
            selected = [sentences[len(sentences) // 2]]  # keep middle
        elif num_to_keep == 2:
            selected = [sentences[0], sentences[-1]]
        else:
            # Keep first, last, and evenly-distributed middle sentences
            indices = {0, len(sentences) - 1}
            middle_count = num_to_keep - 2
            step = (len(sentences) - 2) / (middle_count + 1)
            for i in range(1, middle_count + 1):
                indices.add(max(1, min(len(sentences) - 2, int(i * step))))
            selected = [sentences[i] for i in sorted(indices)]

        return " ".join(selected)
    
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Compress text using LLMLingua.
        
        Args:
            text: The text to compress.
            **kwargs: Additional parameters.
                - rate: Override target compression rate.
            
        Returns:
            CompressionResult with compressed text.
        """
        self._ensure_initialized()
        self._validate_kwargs(kwargs)

        rate = kwargs.get("rate", self.rate)
        
        original_tokens = self.token_counter.count_tokens(text)

        used_fallback = self._fallback_active
        try:
            if self._compressor is not None:
                result = self._compressor.compress_prompt(
                    context=text,
                    rate=rate
                )
                compressed_text = result.get("compressed_prompt", text)
            else:
                compressed_text = self._simple_compress(text, rate)
                used_fallback = True
        except Exception:
            compressed_text = self._simple_compress(text, rate)
            used_fallback = True

        compressed_tokens = self.token_counter.count_tokens(compressed_text)

        compression_ratio = 0.0
        if original_tokens > 0:
            compression_ratio = 1.0 - (compressed_tokens / original_tokens)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            metadata={
                "method": "llmlingua",
                "model_name": self.model_name,
                "target_rate": rate,
                "actual_rate": compression_ratio,
                "fallback_active": used_fallback,
            }
        )


class LLMLingua2Compressor(BaseCompressor):
    """Compressor using LLMLingua2 algorithm."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the LLMLingua2 compressor.
        
        Args:
            config: Configuration dictionary with keys:
                - rate: Target compression rate (default: 0.5)
        """
        super().__init__(name="llmlingua2", config=config)
        self.token_counter = TokenCounter()
        self.rate = self.config.get("rate", 0.5)
        self._compressor = None
        self._fallback_active = False

    def initialize(self) -> None:
        """Initialize the LLMLingua2 compressor."""
        try:
            from llmlingua import PromptCompressor
            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-xlm-roberta-large",
                device_map="cpu"
            )
            self._is_initialized = True
        except ImportError:
            self._fallback_active = True
            self._is_initialized = True
        except Exception as e:
            warnings.warn(f"LLMLingua2 init failed, falling back to heuristic: {e!r}")
            self._fallback_active = True
            self._is_initialized = True
    
    def _simple_compress(self, text: str, rate: float) -> str:
        """Fallback: word-level token pruning (removes stopwords and low-content tokens).

        Mimics LLMLingua-2's fine-grained token-level classification approach.
        """
        import re

        from pcb.utils.stopwords import STOPWORDS as _STOPWORDS

        words = text.split()
        if not words:
            return text

        target_words = max(1, int(len(words) * (1.0 - rate)))

        # Score each word: low score = more likely to remove
        scored = []
        word_freq: dict = {}
        for w in words:
            clean = re.sub(r'[^a-z0-9]', '', w.lower())
            word_freq[clean] = word_freq.get(clean, 0) + 1

        for i, w in enumerate(words):
            clean = re.sub(r'[^a-z0-9]', '', w.lower())
            is_stop = clean in _STOPWORDS
            freq_score = 1.0 / (word_freq.get(clean, 1) + 1)  # rare = higher score
            len_score = min(1.0, len(clean) / 8)               # longer = more content
            score = (0 if is_stop else 0.5) + 0.3 * len_score + 0.2 * freq_score
            scored.append((i, w, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        keep_indices = {idx for idx, _, _ in scored[:target_words]}
        result = " ".join(w for i, w, _ in sorted(scored, key=lambda x: x[0]) if i in keep_indices)
        return result if result.strip() else words[0]
    
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Compress text using LLMLingua2.
        
        Args:
            text: The text to compress.
            **kwargs: Additional parameters.
                - rate: Override target compression rate.
            
        Returns:
            CompressionResult with compressed text.
        """
        self._ensure_initialized()
        self._validate_kwargs(kwargs)

        rate = kwargs.get("rate", self.rate)
        
        original_tokens = self.token_counter.count_tokens(text)

        used_fallback = self._fallback_active
        try:
            if self._compressor is not None:
                result = self._compressor.compress_prompt(
                    context=text,
                    rate=rate,
                    force_tokens=["\n", "."]
                )
                compressed_text = result.get("compressed_prompt", text)
            else:
                compressed_text = self._simple_compress(text, rate)
                used_fallback = True
        except Exception:
            compressed_text = self._simple_compress(text, rate)
            used_fallback = True

        compressed_tokens = self.token_counter.count_tokens(compressed_text)

        compression_ratio = 0.0
        if original_tokens > 0:
            compression_ratio = 1.0 - (compressed_tokens / original_tokens)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            metadata={
                "method": "llmlingua2",
                "target_rate": rate,
                "actual_rate": compression_ratio,
                "fallback_active": used_fallback,
            }
        )
