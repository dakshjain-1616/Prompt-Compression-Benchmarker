"""CSV reporter — flat row per compressor × task."""

import csv
from pathlib import Path

from pcb.runner import BenchmarkReport


class CSVReporter:
    def render(self, report: BenchmarkReport, path: Path) -> None:
        rows = []
        for s in report.summaries:
            rows.append({
                "compressor": s.compressor_name,
                "task": s.task_type,
                "avg_token_reduction_pct": round(s.avg_token_reduction_pct, 2),
                "avg_quality_score": round(s.avg_quality_score, 4),
                "avg_quality_drop_pct": round(s.avg_quality_drop_pct, 2),
                "avg_llm_score": round(s.avg_llm_score, 4) if s.avg_llm_score is not None else "",
                "avg_llm_drop_pct": round(s.avg_llm_drop_pct, 2) if s.avg_llm_drop_pct is not None else "",
                "llm_model": s.llm_model or "",
                "avg_latency_ms": round(s.avg_latency_ms, 2),
                "num_samples": s.num_samples,
                "num_errors": s.num_errors,
            })

        if not rows:
            path.write_text("")
            return

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        print(f"  CSV report written to {path}")
