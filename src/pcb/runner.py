"""Benchmark runner — orchestrates compressors × tasks × samples."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pcb.compressors import ALL_COMPRESSORS
from pcb.compressors.base import BaseCompressor
from pcb.tasks import ALL_TASKS
from pcb.tasks.base import TaskResult


@dataclass
class BenchmarkSummary:
    compressor_name: str
    task_type: str
    avg_compression_ratio: float
    avg_token_reduction_pct: float
    avg_quality_score: float
    avg_quality_drop_pct: float
    avg_latency_ms: float
    num_samples: int
    num_errors: int
    raw_results: List[TaskResult] = field(default_factory=list)
    # LLM judge averages (None when judge not used)
    avg_llm_score: Optional[float] = None
    avg_llm_drop_pct: Optional[float] = None
    llm_model: Optional[str] = None


@dataclass
class BenchmarkReport:
    summaries: List[BenchmarkSummary] = field(default_factory=list)
    compressor_names: List[str] = field(default_factory=list)
    task_types: List[str] = field(default_factory=list)
    judge_model: Optional[str] = None


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_benchmark(
    compressor_names: List[str],
    task_names: List[str],
    data_dir: Path,
    max_samples: Optional[int] = None,
    rate: float = 0.5,
    judge=None,  # Optional[LLMJudge]
) -> BenchmarkReport:
    report = BenchmarkReport(
        compressor_names=compressor_names,
        task_types=task_names,
        judge_model=judge.name if judge else None,
    )

    compressors: List[BaseCompressor] = []
    for name in compressor_names:
        cls = ALL_COMPRESSORS.get(name)
        if cls is None:
            print(f"  [warn] Unknown compressor '{name}', skipping.")
            continue
        cfg = {"rate": rate, "token_budget": int(1000 * (1 - rate)), "top_k": 5}
        compressors.append(cls(config=cfg))

    for task_name in task_names:
        task_cls = ALL_TASKS.get(task_name)
        if task_cls is None:
            print(f"  [warn] Unknown task '{task_name}', skipping.")
            continue
        task = task_cls()

        data_file = data_dir / f"{task_name}_samples.jsonl"
        if not data_file.exists():
            alt = data_dir / f"sample_{task_name}.jsonl"
            if alt.exists():
                data_file = alt
            else:
                print(f"  [warn] No data file for task '{task_name}' at {data_file}, skipping.")
                continue

        samples = task.load_samples(str(data_file), max_samples)
        if not samples:
            print(f"  [warn] No samples loaded for task '{task_name}'.")
            continue

        # Identify baseline compressor
        baseline_compressor = next(
            (c for c in compressors if c.name == "no_compression"),
            compressors[0] if compressors else None,
        )

        # First pass: proxy baselines + optional LLM baselines
        proxy_baselines: Dict[str, float] = {}
        llm_baselines: Dict[str, float] = {}

        if baseline_compressor is not None:
            for sample in samples:
                sid = sample.get("id", str(samples.index(sample)))
                try:
                    r = task.evaluate(
                        sample, baseline_compressor,
                        baseline_quality=None,
                        judge=judge,
                        llm_baseline_score=None,
                    )
                    proxy_baselines[sid] = r.quality_score
                    if r.llm_score is not None:
                        llm_baselines[sid] = r.llm_score
                except Exception as e:
                    # Leave baseline unset so downstream quality_drop is skipped rather than
                    # computed against a bogus 0.0. Surfacing the error matters — a silent
                    # baseline failure makes every compressor's drop % misleading.
                    print(f"  [warn] baseline eval failed for sample {sid}: {e}")

        # Second pass: all compressors
        for compressor in compressors:
            results: List[TaskResult] = []
            for sample in samples:
                sid = sample.get("id", str(samples.index(sample)))
                bq = proxy_baselines.get(sid)
                llm_bq = llm_baselines.get(sid) if judge else None
                try:
                    r = task.evaluate(
                        sample, compressor,
                        baseline_quality=bq,
                        judge=judge,
                        llm_baseline_score=llm_bq,
                    )
                    results.append(r)
                except Exception as e:
                    from pcb.utils.token_counter import TokenCounter
                    tc = TokenCounter()
                    raw = sample.get("context", sample.get("article", ""))
                    tok = tc.count_tokens(raw)
                    results.append(TaskResult(
                        compressor_name=compressor.name,
                        task_type=task_name,
                        sample_id=sid,
                        original_tokens=tok,
                        compressed_tokens=tok,
                        compression_ratio=0.0,
                        token_reduction_pct=0.0,
                        quality_score=0.0,
                        baseline_quality=bq or 0.0,
                        quality_drop_pct=0.0,
                        latency_ms=0.0,
                        error=str(e),
                    ))

            valid = [r for r in results if r.error is None]

            # LLM averages
            llm_scores = [r.llm_score for r in valid if r.llm_score is not None]
            llm_drops = [r.llm_drop_pct for r in valid if r.llm_drop_pct is not None]

            summary = BenchmarkSummary(
                compressor_name=compressor.name,
                task_type=task_name,
                avg_compression_ratio=_mean([r.compression_ratio for r in valid]),
                avg_token_reduction_pct=_mean([r.token_reduction_pct for r in valid]),
                avg_quality_score=_mean([r.quality_score for r in valid]),
                avg_quality_drop_pct=_mean([r.quality_drop_pct for r in valid]),
                avg_latency_ms=_mean([r.latency_ms for r in results]),
                num_samples=len(results),
                num_errors=len([r for r in results if r.error]),
                raw_results=results,
                avg_llm_score=_mean(llm_scores) if llm_scores else None,
                avg_llm_drop_pct=_mean(llm_drops) if llm_drops else None,
                llm_model=results[0].llm_model if results and results[0].llm_model else None,
            )
            report.summaries.append(summary)

    return report
