"""No-op compressor that returns text unchanged."""

from typing import Any, Optional

from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.utils.token_counter import TokenCounter


class NoCompressionCompressor(BaseCompressor):
    """Compressor that performs no compression - returns text unchanged."""
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the no-compression compressor.
        
        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(name="no_compression", config=config)
        self.token_counter = TokenCounter()
    
    def initialize(self) -> None:
        """No initialization needed for no-compression."""
        self._is_initialized = True
    
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Return text unchanged (no compression).
        
        Args:
            text: The text to "compress".
            **kwargs: Additional parameters (ignored).
            
        Returns:
            CompressionResult with original text and 0% compression.
        """
        self._ensure_initialized()
        
        original_tokens = self.token_counter.count_tokens(text)
        
        return CompressionResult(
            original_text=text,
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=0.0,
            metadata={"method": "no_compression"}
        )
