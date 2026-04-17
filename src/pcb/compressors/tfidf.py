"""TF-IDF based compressor for prompt compression."""

import re
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.utils.token_counter import TokenCounter


class TFIDFCompressor(BaseCompressor):
    """Compressor that uses TF-IDF to extract most important sentences."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the TF-IDF compressor.
        
        Args:
            config: Configuration dictionary with keys:
                - top_k: Number of top sentences to keep (default: 10)
                - min_df: Minimum document frequency (default: 1)
                - max_df: Maximum document frequency (default: 0.95)
        """
        super().__init__(name="tfidf", config=config)
        self.token_counter = TokenCounter()
        self.top_k = self.config.get("top_k", 10)
        self.min_df = self.config.get("min_df", 1)
        self.max_df = self.config.get("max_df", 0.95)
        self.vectorizer: Optional[TfidfVectorizer] = None
    
    def initialize(self) -> None:
        """Initialize the TF-IDF vectorizer."""
        self.vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True
        )
        self._is_initialized = True
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting on periods, question marks, exclamation marks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Compress text using TF-IDF sentence selection.

        Args:
            text: The text to compress.
            **kwargs: Additional parameters.
                - top_k: Override number of sentences to keep (absolute).
                - rate: Target compression rate 0-1 (overrides top_k when present).

        Returns:
            CompressionResult with compressed text.
        """
        self._ensure_initialized()

        # Split into sentences first so we can compute ratio-based top_k
        sentences = self._split_into_sentences(text)

        rate = kwargs.get("rate", None)
        if rate is not None:
            # ratio-based: keep (1 - rate) fraction of sentences
            top_k = max(1, int(len(sentences) * (1.0 - rate)))
        else:
            top_k = kwargs.get("top_k", self.top_k)

        if len(sentences) <= 1:
            # Nothing to compress
            original_tokens = self.token_counter.count_tokens(text)
            return CompressionResult(
                original_text=text,
                compressed_text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                compression_ratio=0.0,
                metadata={"method": "tfidf", "reason": "single_sentence"}
            )

        if top_k >= len(sentences):
            top_k = max(1, len(sentences) - 1)
        
        # Calculate TF-IDF scores
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            # Sum TF-IDF scores for each sentence
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top-k sentence indices
            top_indices = np.argsort(sentence_scores)[-top_k:]
            top_indices = sorted(top_indices)  # Keep original order
            
            # Reconstruct text with selected sentences
            selected_sentences = [sentences[i] for i in top_indices]
            compressed_text = " ".join(selected_sentences)
            
        except ValueError:
            # TF-IDF failed (e.g., all words are stop words)
            compressed_text = text
        
        original_tokens = self.token_counter.count_tokens(text)
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
                "method": "tfidf",
                "top_k": top_k,
                "sentences_kept": len(selected_sentences) if 'selected_sentences' in locals() else len(sentences),
                "total_sentences": len(sentences)
            }
        )
