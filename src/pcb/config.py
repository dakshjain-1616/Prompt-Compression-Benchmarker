"""Configuration module for Prompt Compression Benchmarker."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class TaskType(str, Enum):
    """Supported task types for benchmarking."""
    RAG = "rag"
    SUMMARIZATION = "summarization"
    CODING = "coding"


class CompressorType(str, Enum):
    """Supported compressor types."""
    NO_COMPRESSION = "no_compression"
    TFIDF = "tfidf"
    SELECTIVE_CONTEXT = "selective_context"
    LLMLINGUA = "llmlingua"
    LLMLINGUA2 = "llmlingua2"


class OutputFormat(str, Enum):
    """Supported output formats."""
    TERMINAL = "terminal"
    JSON = "json"
    CSV = "csv"
    HTML = "html"


@dataclass
class CompressionConfig:
    """Configuration for compression algorithms."""
    target_ratio: float = 0.5
    min_tokens: int = 50
    max_tokens: Optional[int] = None
    preserve_context: bool = True
    
    # TF-IDF specific
    tfidf_top_k: int = 10
    
    # Selective Context specific
    sc_token_budget: int = 1000
    sc_context_level: str = "sentence"  # "token", "phrase", "sentence"
    
    # LLMLingua specific
    llmlingua_model: str = "lgaalves/gpt2-dolly-v2"
    llmlingua_rate: float = 0.5
    
    # LLMLingua2 specific
    llmlingua2_rate: float = 0.5


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    tasks: List[TaskType] = field(default_factory=lambda: [TaskType.RAG])
    compressors: List[CompressorType] = field(
        default_factory=lambda: [CompressorType.NO_COMPRESSION]
    )
    output_formats: List[OutputFormat] = field(
        default_factory=lambda: [OutputFormat.TERMINAL]
    )
    compression_config: CompressionConfig = field(default_factory=CompressionConfig)
    
    # Dataset paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("results"))
    output_filename: Optional[str] = None
    
    # Evaluation settings
    max_samples: Optional[int] = None
    random_seed: int = 42
    
    # Model settings for evaluation
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)


# Default configurations
DEFAULT_CONFIG = BenchmarkConfig()

# Task-specific configurations
TASK_CONFIGS: Dict[TaskType, Dict[str, Union[int, float, str]]] = {
    TaskType.RAG: {
        "max_context_length": 4096,
        "evaluation_metric": "retrieval_accuracy",
    },
    TaskType.SUMMARIZATION: {
        "max_context_length": 2048,
        "evaluation_metric": "rouge_score",
    },
    TaskType.CODING: {
        "max_context_length": 8192,
        "evaluation_metric": "code_similarity",
    },
}

# Compressor-specific default settings
COMPRESSOR_DEFAULTS: Dict[CompressorType, Dict[str, Union[int, float, str]]] = {
    CompressorType.TFIDF: {
        "top_k": 10,
        "min_df": 1,
        "max_df": 0.95,
    },
    CompressorType.SELECTIVE_CONTEXT: {
        "token_budget": 1000,
        "context_level": "sentence",
    },
    CompressorType.LLMLINGUA: {
        "model_name": "lgaalves/gpt2-dolly-v2",
        "rate": 0.5,
    },
    CompressorType.LLMLINGUA2: {
        "rate": 0.5,
    },
}
