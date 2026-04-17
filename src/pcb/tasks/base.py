"""Abstract base class for benchmark tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pcb.compressors.base import BaseCompressor, CompressionResult


@dataclass
class TaskResult:
    """Result of evaluating one sample with one compressor."""
    compressor_name: str
    task_type: str
    sample_id: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float       # 1 - compressed/original
    token_reduction_pct: float     # compression_ratio * 100
    quality_score: float           # task-specific proxy score 0-1
    baseline_quality: float        # no_compression quality for this sample
    quality_drop_pct: float        # (baseline - quality) / baseline * 100
    latency_ms: float
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    # LLM-as-judge fields (populated when --llm-judge is enabled)
    llm_score: Optional[float] = None           # judge score 0-1
    llm_baseline_score: Optional[float] = None  # baseline judge score
    llm_drop_pct: Optional[float] = None        # judge quality drop %
    llm_model: Optional[str] = None             # which model judged


class BaseTask(ABC):
    """Abstract task: knows how to load data and evaluate compressed output."""

    task_type: str = "base"

    @abstractmethod
    def load_samples(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load JSONL samples from path."""
        ...

    @abstractmethod
    def evaluate(
        self,
        sample: Dict[str, Any],
        compressor: BaseCompressor,
        baseline_quality: Optional[float] = None,
    ) -> TaskResult:
        """Run compressor on sample and return scored TaskResult."""
        ...

    def _load_jsonl(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        import json
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        if max_samples:
            samples = samples[:max_samples]
        return samples
