"""Coding task — compresses docstring/context, scores identifier preservation + BM25.

Code is split into logical blocks (imports, function defs, docstrings) rather than
prose sentences, since code rarely uses `.!?` as sentence terminators.
"""

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from pcb.compressors.base import BaseCompressor
from pcb.tasks.base import BaseTask, TaskResult

if TYPE_CHECKING:
    from pcb.evaluators.llm_judge import LLMJudge


def _extract_identifiers(code: str) -> Set[str]:
    tokens = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", code)
    stopwords = {
        "def", "class", "return", "import", "from", "if", "else", "elif",
        "for", "while", "try", "except", "with", "as", "pass", "None",
        "True", "False", "and", "or", "not", "in", "is", "lambda", "self",
        "str", "int", "float", "list", "dict", "set", "tuple", "bool",
        "print", "range", "len", "type", "isinstance", "hasattr", "getattr",
    }
    return {t for t in tokens if t not in stopwords}


def _bm25_score(query_tokens: List[str], corpus_tokens: List[str]) -> float:
    try:
        from rank_bm25 import BM25Okapi
        if not corpus_tokens or not query_tokens:
            return 0.0
        bm25 = BM25Okapi([corpus_tokens])
        scores = bm25.get_scores(query_tokens)
        raw = float(scores[0])
        return min(1.0, raw / max(len(query_tokens), 1))
    except Exception:
        q = set(query_tokens)
        c = set(corpus_tokens)
        if not q:
            return 0.0
        return len(q & c) / len(q)


def _identifier_preservation(compressed: str, solution: str) -> float:
    solution_ids = _extract_identifiers(solution)
    if not solution_ids:
        return 1.0
    compressed_ids = _extract_identifiers(compressed)
    return len(solution_ids & compressed_ids) / len(solution_ids)


class CodingTask(BaseTask):
    task_type = "coding"

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
        raw_ctx = sample.get("context", "")
        docstring = sample.get("docstring", "")
        blocks = [b.strip() for b in re.split(r'\n{2,}', raw_ctx) if b.strip()]
        context = ". ".join(blocks)
        if docstring:
            context = context + ". " + docstring.strip()
        solution: str = sample.get("solution", "")
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

        compressed_tokens_list = compressed_ctx.lower().split()
        solution_tokens = solution.lower().split()
        bm25 = _bm25_score(solution_tokens, compressed_tokens_list)
        id_preservation = _identifier_preservation(compressed_ctx, solution)
        quality_score = 0.6 * id_preservation + 0.4 * bm25

        baseline = baseline_quality if baseline_quality is not None else quality_score
        quality_drop = 0.0
        if baseline > 0:
            quality_drop = (baseline - quality_score) / baseline * 100

        # LLM judge (optional)
        llm_score: Optional[float] = None
        llm_drop: Optional[float] = None
        if judge is not None:
            llm_score = judge.score_coding(
                compressed_ctx, docstring, solution, sample_id
            )
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
            metrics={"identifier_preservation": id_preservation, "bm25_similarity": bm25},
            error=error,
            llm_score=llm_score,
            llm_baseline_score=llm_baseline_score,
            llm_drop_pct=llm_drop,
            llm_model=judge.name if judge else None,
        )
