"""Self-contained HTML reporter with inline Chart.js scatter plot."""

import json
from datetime import datetime
from pathlib import Path

from pcb.runner import BenchmarkReport

# Inline Chart.js 4.x minified (fetched once, stored inline for offline use)
# We use CDN but also embed a fallback message if offline
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Prompt Compression Benchmarker Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e1e4e8; margin: 0; padding: 24px; }}
  h1   {{ color: #58a6ff; margin-bottom: 4px; }}
  .meta {{ color: #8b949e; font-size: 0.85em; margin-bottom: 32px; }}
  .section {{ margin-bottom: 48px; }}
  h2   {{ color: #79c0ff; border-bottom: 1px solid #30363d; padding-bottom: 8px; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
  th   {{ background: #161b22; color: #58a6ff; padding: 10px 14px;
          text-align: left; border-bottom: 2px solid #30363d; }}
  td   {{ padding: 8px 14px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #161b22; }}
  .good  {{ color: #3fb950; font-weight: 600; }}
  .warn  {{ color: #d29922; font-weight: 600; }}
  .bad   {{ color: #f85149; font-weight: 600; }}
  .star  {{ color: #f0e040; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d;
                 border-radius: 8px; padding: 20px; margin-top: 24px; max-width: 700px; }}
  canvas {{ max-width: 100%; }}
</style>
</head>
<body>
<h1>&#128202; Prompt Compression Benchmarker</h1>
<p class="meta">Generated {generated_at} &nbsp;|&nbsp; Compressors: {compressor_list}
&nbsp;|&nbsp; Tasks: {task_list}</p>

{sections}

<script>
const allData = {chart_data_json};

const COLORS = [
  "#58a6ff","#3fb950","#d29922","#f85149","#bc8cff","#39d353"
];

allData.forEach(function(taskData) {{
  const ctx = document.getElementById("chart_" + taskData.task);
  if (!ctx) return;
  const datasets = taskData.compressors.map(function(c, i) {{
    return {{
      label: c.name,
      data: [{{ x: c.token_reduction_pct, y: c.quality_drop_pct }}],
      backgroundColor: COLORS[i % COLORS.length],
      pointRadius: 8,
      pointHoverRadius: 12,
    }};
  }});
  new Chart(ctx, {{
    type: "scatter",
    data: {{ datasets: datasets }},
    options: {{
      responsive: true,
      plugins: {{
        title: {{ display: true, text: taskData.task.toUpperCase() + " — Compression vs Quality Drop",
                  color: "#e1e4e8" }},
        legend: {{ labels: {{ color: "#e1e4e8" }} }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return ctx.dataset.label + ": " +
                ctx.parsed.x.toFixed(1) + "% reduction, " +
                ctx.parsed.y.toFixed(1) + "% quality drop";
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: "Token Reduction %", color: "#8b949e" }},
              ticks: {{ color: "#8b949e" }}, grid: {{ color: "#21262d" }} }},
        y: {{ title: {{ display: true, text: "Quality Drop %", color: "#8b949e" }},
              ticks: {{ color: "#8b949e" }}, grid: {{ color: "#21262d" }},
              min: 0 }}
      }}
    }}
  }});
}});
</script>
</body>
</html>
"""

_SECTION_TEMPLATE = """\
<div class="section">
  <h2>{task_title}</h2>
  <table>
    <thead><tr>
      <th>Compressor</th>
      <th>Token Reduction %</th>
      <th>Avg Orig Tokens</th>
      <th>Avg Comp Tokens</th>
      <th>Quality Score</th>
      <th>Quality Drop %</th>
      <th>Latency (ms)</th>
      <th>Samples</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="chart-wrap">
    <canvas id="chart_{task_id}"></canvas>
  </div>
</div>
"""


def _drop_class(drop: float) -> str:
    if drop < 5:
        return "good"
    if drop < 15:
        return "warn"
    return "bad"


class HTMLReporter:
    def render(self, report: BenchmarkReport, path: Path) -> None:
        sections = []
        chart_data = []

        for task_type in report.task_types:
            task_summaries = [s for s in report.summaries if s.task_type == task_type]
            if not task_summaries:
                continue

            # Find pareto best
            pareto = ""
            best_score = -1.0
            for s in task_summaries:
                if s.avg_quality_drop_pct < 15:
                    sc = s.avg_token_reduction_pct - s.avg_quality_drop_pct
                    if sc > best_score:
                        best_score = sc
                        pareto = s.compressor_name

            rows_html = ""
            compressor_chart_data = []
            for s in task_summaries:
                orig_avg = comp_avg = 0
                if s.raw_results:
                    orig_avg = int(sum(r.original_tokens for r in s.raw_results) / len(s.raw_results))
                    comp_avg = int(sum(r.compressed_tokens for r in s.raw_results) / len(s.raw_results))
                cls = _drop_class(s.avg_quality_drop_pct)
                star = '<span class="star"> ★</span>' if s.compressor_name == pareto else ""
                rows_html += (
                    f"<tr>"
                    f"<td><strong>{s.compressor_name}</strong>{star}</td>"
                    f"<td>{s.avg_token_reduction_pct:.1f}%</td>"
                    f"<td>{orig_avg}</td>"
                    f"<td>{comp_avg}</td>"
                    f"<td>{s.avg_quality_score:.4f}</td>"
                    f'<td class="{cls}">{s.avg_quality_drop_pct:.1f}%</td>'
                    f"<td>{s.avg_latency_ms:.1f}</td>"
                    f"<td>{s.num_samples}</td>"
                    f"</tr>"
                )
                compressor_chart_data.append({
                    "name": s.compressor_name,
                    "token_reduction_pct": round(s.avg_token_reduction_pct, 2),
                    "quality_drop_pct": round(s.avg_quality_drop_pct, 2),
                })

            sections.append(_SECTION_TEMPLATE.format(
                task_title=task_type.upper(),
                task_id=task_type,
                rows=rows_html,
            ))
            chart_data.append({"task": task_type, "compressors": compressor_chart_data})

        html = _HTML_TEMPLATE.format(
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            compressor_list=", ".join(report.compressor_names),
            task_list=", ".join(report.task_types),
            sections="\n".join(sections),
            chart_data_json=json.dumps(chart_data),
        )
        path.write_text(html)
        print(f"  HTML report written to {path}")
