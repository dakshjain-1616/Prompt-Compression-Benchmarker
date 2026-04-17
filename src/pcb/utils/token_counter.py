"""Token counting utility using tiktoken."""

from typing import Optional

import tiktoken


class TokenCounter:
    """Utility class for counting tokens in text using tiktoken."""
    
    # Default encoding to use
    DEFAULT_ENCODING = "cl100k_base"  # Used by GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    
    def __init__(self, encoding_name: Optional[str] = None):
        """
        Initialize the token counter with a specific encoding.
        
        Args:
            encoding_name: The tiktoken encoding to use. Defaults to cl100k_base.
        """
        self.encoding_name = encoding_name or self.DEFAULT_ENCODING
        try:
            self.encoding = tiktoken.get_encoding(self.encoding_name)
        except KeyError:
            # Fallback to default if encoding not found
            self.encoding = tiktoken.get_encoding(self.DEFAULT_ENCODING)
            self.encoding_name = self.DEFAULT_ENCODING
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for.
            
        Returns:
            The number of tokens in the text.
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_tokens_batch(self, texts: list[str]) -> list[int]:
        """
        Count tokens for a batch of texts.
        
        Args:
            texts: List of texts to count tokens for.
            
        Returns:
            List of token counts corresponding to each text.
        """
        return [self.count_tokens(text) for text in texts]
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to a maximum number of tokens.
        
        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens allowed.
            
        Returns:
            The truncated text.
        """
        if not text:
            return ""
        
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def get_token_ratio(self, original: str, compressed: str) -> float:
        """
        Calculate the compression ratio between original and compressed text.
        
        Args:
            original: The original text.
            compressed: The compressed text.
            
        Returns:
            The compression ratio (compressed / original).
        """
        original_tokens = self.count_tokens(original)
        compressed_tokens = self.count_tokens(compressed)
        
        if original_tokens == 0:
            return 1.0
        
        return compressed_tokens / original_tokens
    
    def get_compression_ratio(self, original: str, compressed: str) -> float:
        """
        Calculate the compression ratio as a percentage reduction.
        
        Args:
            original: The original text.
            compressed: The compressed text.
            
        Returns:
            The compression ratio (1 - compressed/original).
        """
        token_ratio = self.get_token_ratio(original, compressed)
        return 1.0 - token_ratio


# Global instance for convenience
default_counter = TokenCounter()


def count_tokens(text: str, encoding_name: Optional[str] = None) -> int:
    """
    Convenience function to count tokens in text.
    
    Args:
        text: The text to count tokens for.
        encoding_name: Optional encoding name to use.
        
    Returns:
        The number of tokens.
    """
    if encoding_name:
        counter = TokenCounter(encoding_name)
        return counter.count_tokens(text)
    return default_counter.count_tokens(text)


def get_compression_ratio(original: str, compressed: str) -> float:
    """
    Convenience function to get compression ratio.
    
    Args:
        original: The original text.
        compressed: The compressed text.
        
    Returns:
        The compression ratio.
    """
    return default_counter.get_compression_ratio(original, compressed)
