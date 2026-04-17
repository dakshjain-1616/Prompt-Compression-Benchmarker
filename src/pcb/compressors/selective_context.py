"""Selective Context compressor for prompt compression."""

from typing import Any, Optional

from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.utils.token_counter import TokenCounter


class SelectiveContextCompressor(BaseCompressor):
    """Compressor using Selective Context algorithm."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the Selective Context compressor.
        
        Args:
            config: Configuration dictionary with keys:
                - token_budget: Target token budget (default: 1000)
                - context_level: Level of context selection (default: "sentence")
        """
        super().__init__(name="selective_context", config=config)
        self.token_counter = TokenCounter()
        self.token_budget = self.config.get("token_budget", 1000)
        self.context_level = self.config.get("context_level", "sentence")
        self._compressor = None
    
    def initialize(self) -> None:
        """Initialize the Selective Context compressor."""
        try:
            from selective_context import SelectiveContext
            self._compressor = SelectiveContext(
                model_type="gpt2",
                lang="en"
            )
            self._is_initialized = True
        except ImportError:
            # Fallback to simple implementation if selective-context not available
            self._is_initialized = True
    
    def _simple_compress(self, text: str, token_budget: int) -> str:
        """Simple fallback compression when selective-context is not available."""
        import re

        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        compressed_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= token_budget:
                compressed_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break

        return " ".join(compressed_sentences) if compressed_sentences else sentences[0]
    
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Compress text using Selective Context.
        
        Args:
            text: The text to compress.
            **kwargs: Additional parameters.
                - token_budget: Override target token budget.
            
        Returns:
            CompressionResult with compressed text.
        """
        self._ensure_initialized()
        
        rate = kwargs.get("rate", None)
        original_tokens = self.token_counter.count_tokens(text)
        if rate is not None:
            token_budget = max(10, int(original_tokens * (1.0 - rate)))
        else:
            token_budget = kwargs.get("token_budget", min(self.token_budget, max(10, int(original_tokens * 0.5))))
        
        try:
            if self._compressor is not None:
                # Use selective-context library
                compressed_text = self._compressor(
                    context=text,
                    reduce_ratio=1.0 - (token_budget / max(original_tokens, 1))
                )
            else:
                # Use fallback implementation
                compressed_text = self._simple_compress(text, token_budget)
        except Exception:
            # Fallback on any error
            compressed_text = self._simple_compress(text, token_budget)
        
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
                "method": "selective_context",
                "token_budget": token_budget,
                "context_level": self.context_level
            }
        )
