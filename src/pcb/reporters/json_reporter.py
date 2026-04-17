"""JSON reporter."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from pcb.runner import BenchmarkReport


class JSONReporter:
    def render(self, report: BenchmarkReport, path: Path) -> None:
        data = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "compressors": report.compressor_names,
            "tasks": report.task_types,
            "summaries": [],
        }
        for s in report.summaries:
            entry = {
                "compressor": s.compressor_name,
                "task": s.task_type,
                "avg_compression_ratio": round(s.avg_compression_ratio, 4),
                "avg_token_reduction_pct": round(s.avg_token_reduction_pct, 2),
                "avg_quality_score": round(s.avg_quality_score, 4),
                "avg_quality_drop_pct": round(s.avg_quality_drop_pct, 2),
                "avg_llm_score": round(s.avg_llm_score, 4) if s.avg_llm_score is not None else None,
                "avg_llm_drop_pct": round(s.avg_llm_drop_pct, 2) if s.avg_llm_drop_pct is not None else None,
                "llm_model": s.llm_model,
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "num_samples": s.num_samples,
                "num_errors": s.num_errors,
                "samples": [
                    {
                        "id": r.sample_id,
                        "original_tokens": r.original_tokens,
                        "compressed_tokens": r.compressed_tokens,
                        "compression_ratio": round(r.compression_ratio, 4),
                        "quality_score": round(r.quality_score, 4),
                        "quality_drop_pct": round(r.quality_drop_pct, 2),
                        "llm_score": round(r.llm_score, 4) if r.llm_score is not None else None,
                        "llm_drop_pct": round(r.llm_drop_pct, 2) if r.llm_drop_pct is not None else None,
                        "latency_ms": round(r.latency_ms, 2),
                        "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                        "error": r.error,
                    }
                    for r in s.raw_results
                ],
            }
            data["summaries"].append(entry)

        path.write_text(json.dumps(data, indent=2))
        print(f"  JSON report written to {path}")
