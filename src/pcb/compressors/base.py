"""Base compressor interface for prompt compression."""

import warnings
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

    # Subclasses set True if their compress() uses the `query` kwarg.
    supports_query: bool = False

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

    def _validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate common kwargs. Call at the top of compress().

        - `rate` must be in [0.0, 1.0). 1.0 would remove all tokens.
        - `query` warns (not raises) when the compressor doesn't use it, so
          callers don't silently think they're doing query-aware compression.
        """
        rate = kwargs.get("rate")
        if rate is not None and not (0.0 <= rate < 1.0):
            raise ValueError(f"rate must be in [0.0, 1.0), got {rate}")
        if kwargs.get("query") and not self.supports_query:
            warnings.warn(
                f"{self.name} ignores the 'query' kwarg — only 'smart' supports "
                "query-aware compression. Pass compressor='smart' for query scoring.",
                stacklevel=3,
            )
