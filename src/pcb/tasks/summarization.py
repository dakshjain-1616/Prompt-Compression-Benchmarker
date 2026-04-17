"""Summarization task — compresses article, scores ROUGE vs reference summary."""

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pcb.compressors.base import BaseCompressor
from pcb.tasks.base import BaseTask, TaskResult

if TYPE_CHECKING:
    from pcb.evaluators.llm_judge import LLMJudge


def _extractive_summary(text: str, num_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:num_sentences])


def _rouge_scores(hypothesis: str, reference: str) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception:
        hyp = set(hypothesis.lower().split())
        ref = set(reference.lower().split())
        overlap = len(hyp & ref) / max(len(ref), 1)
        return {"rouge1": overlap, "rouge2": overlap * 0.5, "rougeL": overlap}


class SummarizationTask(BaseTask):
    task_type = "summarization"

    def load_samples(self, path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        return self._load_jsonl(path, max_samples)

    def evaluate(
        self,
        sample: Dict[str, Any],
        compressor: BaseCompressor,
        baseline_quality: Optional[float] = None,
        judge: Optional["LLMJudge"] = None,
        llm_baseline_score: Optional[float] = None,
    ) -> TaskResult:
        article: str = sample["article"]
        reference_summary: str = sample.get("summary", "")
        sample_id: str = sample.get("id", "unknown")

        t0 = time.perf_counter()
        try:
            rate = getattr(compressor, "rate", 0.5)
            result = compressor.compress(article, rate=rate)
            compressed_article = result.compressed_text
            original_tokens = result.original_tokens
            compressed_tokens = result.compressed_tokens
            compression_ratio = result.compression_ratio
            error = None
        except Exception as e:
            compressed_article = article
            from pcb.utils.token_counter import TokenCounter
            tc = TokenCounter()
            original_tokens = tc.count_tokens(article)
            compressed_tokens = original_tokens
            compression_ratio = 0.0
            error = str(e)
        latency_ms = (time.perf_counter() - t0) * 1000

        hyp_summary = _extractive_summary(compressed_article, num_sentences=3)
        scores = _rouge_scores(hyp_summary, reference_summary)
        quality_score = scores["rougeL"]

        baseline = baseline_quality if baseline_quality is not None else quality_score
        quality_drop = 0.0
        if baseline > 0:
            quality_drop = (baseline - quality_score) / baseline * 100

        # LLM judge (optional)
        llm_score: Optional[float] = None
        llm_drop: Optional[float] = None
        if judge is not None:
            llm_score = judge.score_summarization(hyp_summary, reference_summary, sample_id)
            if llm_score is not None and llm_baseline_score is not None and llm_baseline_score > 0:
                llm_drop = (llm_baseline_score - llm_score) / llm_baseline_score * 100

        return TaskResult(
            compressor_name=compressor.name,
            task_type=self.task_type,
            sample_id=sample_id,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            token_reduction_pct=compression_ratio * 100,
            quality_score=quality_score,
            baseline_quality=baseline,
            quality_drop_pct=quality_drop,
            latency_ms=latency_ms,
            metrics=scores,
            error=error,
            llm_score=llm_score,
            llm_baseline_score=llm_baseline_score,
            llm_drop_pct=llm_drop,
            llm_model=judge.name if judge else None,
        )
