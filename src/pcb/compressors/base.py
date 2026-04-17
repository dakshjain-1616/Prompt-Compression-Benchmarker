"""Base compressor interface for prompt compression."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float  # (1 - compressed/original)
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseCompressor(ABC):
    """Abstract base class for prompt compressors."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compressor.
        
        Args:
            name: Name of the compressor.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config = config or {}
        self._is_initialized = False
    
    @abstractmethod
    def compress(self, text: str, **kwargs: Any) -> CompressionResult:
        """
        Compress the given text.
        
        Args:
            text: The text to compress.
            **kwargs: Additional compression parameters.
            
        Returns:
            CompressionResult with compression details.
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize any required models or resources."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if the compressor is initialized."""
        return self._is_initialized
    
    def _ensure_initialized(self) -> None:
        """Ensure the compressor is initialized before use."""
        if not self._is_initialized:
            self.initialize()
            self._is_initialized = True
