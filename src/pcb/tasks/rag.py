"""RAG task evaluator — compresses context, measures F1/EM against gold answer."""

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pcb.compressors.base import BaseCompressor
from pcb.tasks.base import BaseTask, TaskResult

if TYPE_CHECKING:
    from pcb.evaluators.llm_judge import LLMJudge


def _normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def _f1(pred_tokens: List[str], gold_tokens: List[str]) -> float:
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return 2 * prec * rec / (prec + rec)


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if " ".join(_normalize(pred)) == " ".join(_normalize(gold)) else 0.0


def _context_recall(compressed: str, answer: str) -> float:
    ans_tokens = set(_normalize(answer))
    ctx_tokens = set(_normalize(compressed))
    if not ans_tokens:
        return 1.0
    return len(ans_tokens & ctx_tokens) / len(ans_tokens)


class RAGTask(BaseTask):
    task_type = "rag"

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
        context: str = sample["context"]
        question: str = sample.get("question", "")
        answer: str = sample.get("answer", "")
        sample_id: str = sample.get("id", "unknown")

        t0 = time.perf_counter()
        try:
            rate = getattr(compressor, "rate", 0.5)
            result = compressor.compress(context, rate=rate)
            compressed_ctx = result.compressed_text
            original_tokens = result.original_tokens
            compressed_tokens = result.compressed_tokens
            compression_ratio = result.compression_ratio
            error = None
        except Exception as e:
            compressed_ctx = context
            from pcb.utils.token_counter import TokenCounter
            tc = TokenCounter()
            original_tokens = tc.count_tokens(context)
            compressed_tokens = original_tokens
            compression_ratio = 0.0
            error = str(e)
        latency_ms = (time.perf_counter() - t0) * 1000

        f1 = _f1(_normalize(compressed_ctx + " " + question), _normalize(answer))
        em = _exact_match(compressed_ctx + " " + question, answer)
        recall = _context_recall(compressed_ctx, answer)
        quality_score = 0.5 * f1 + 0.3 * recall + 0.2 * em

        baseline = baseline_quality if baseline_quality is not None else quality_score
        quality_drop = 0.0
        if baseline > 0:
            quality_drop = (baseline - quality_score) / baseline * 100

        # LLM judge (optional)
        llm_score: Optional[float] = None
        llm_drop: Optional[float] = None
        if judge is not None:
            llm_score = judge.score_rag(compressed_ctx, question, answer, sample_id)
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
            metrics={"f1": f1, "exact_match": em, "context_recall": recall},
            error=error,
            llm_score=llm_score,
            llm_baseline_score=llm_baseline_score,
            llm_drop_pct=llm_drop,
            llm_model=judge.name if judge else None,
        )
