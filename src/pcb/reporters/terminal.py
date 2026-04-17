"""Rich terminal reporter."""

from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from pcb.runner import BenchmarkReport, BenchmarkSummary

console = Console()


def _drop_color(drop_pct: Optional[float]) -> str:
    if drop_pct is None:
        return "white"
    if drop_pct < 0:
        return "cyan"    # improvement
    if drop_pct < 5:
        return "green"
    if drop_pct < 15:
        return "yellow"
    return "red"


def _fmt_drop(drop_pct: Optional[float]) -> str:
    if drop_pct is None:
        return "N/A"
    sign = "+" if drop_pct > 0 else ""
    return f"{sign}{drop_pct:.1f}%"


def _pareto_best(summaries: List[BenchmarkSummary], use_llm: bool = False) -> str:
    best = None
    best_score = -999.0
    for s in summaries:
        if s.compressor_name == "no_compression":
            continue
        drop = s.avg_llm_drop_pct if (use_llm and s.avg_llm_drop_pct is not None) else s.avg_quality_drop_pct
        if drop < 20:
            score = s.avg_token_reduction_pct - max(0.0, drop)
            if score > best_score:
                best_score = score
                best = s.compressor_name
    return best or ""


class TerminalReporter:
    def render(self, report: BenchmarkReport) -> None:
        use_llm = report.judge_model is not None

        console.print()
        judge_note = f"  [dim]LLM Judge: {report.judge_model}[/dim]" if use_llm else ""
        console.print(Panel.fit(
            "[bold cyan]Prompt Compression Benchmarker[/bold cyan]\n"
            "[dim]Token reduction % vs quality drop across task types[/dim]"
            + ("\n" + judge_note if judge_note else ""),
            border_style="cyan",
        ))
        console.print()

        for task_type in report.task_types:
            task_summaries = [s for s in report.summaries if s.task_type == task_type]
            if not task_summaries:
                continue

            pareto = _pareto_best(task_summaries, use_llm)

            table = Table(
                title=f"[bold]{task_type.upper()}[/bold]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Compressor", style="bold white", min_width=20)
            table.add_column("Token Reduc %", justify="right")
            table.add_column("Orig Tok", justify="right")
            table.add_column("Comp Tok", justify="right")
            table.add_column("Proxy Score", justify="right")
            table.add_column("Proxy Drop %", justify="right")
            if use_llm:
                table.add_column("LLM Score", justify="right")
                table.add_column("LLM Drop %", justify="right")
            table.add_column("ms", justify="right")

            for s in task_summaries:
                orig_avg = comp_avg = 0
                if s.raw_results:
                    orig_avg = int(sum(r.original_tokens for r in s.raw_results) / len(s.raw_results))
                    comp_avg = int(sum(r.compressed_tokens for r in s.raw_results) / len(s.raw_results))

                proxy_color = _drop_color(s.avg_quality_drop_pct)
                star = " ★" if s.compressor_name == pareto else ""

                row = [
                    f"{s.compressor_name}{star}",
                    f"{s.avg_token_reduction_pct:.1f}%",
                    str(orig_avg),
                    str(comp_avg),
                    f"{s.avg_quality_score:.4f}",
                    f"[{proxy_color}]{_fmt_drop(s.avg_quality_drop_pct)}[/{proxy_color}]",
                ]
                if use_llm:
                    llm_color = _drop_color(s.avg_llm_drop_pct)
                    llm_score_str = f"{s.avg_llm_score:.4f}" if s.avg_llm_score is not None else "N/A"
                    row.append(f"[bold]{llm_score_str}[/bold]")
                    row.append(f"[{llm_color}]{_fmt_drop(s.avg_llm_drop_pct)}[/{llm_color}]")
                row.append(f"{s.avg_latency_ms:.1f}")

                table.add_row(*row)

            console.print(table)

            legend = "  [dim]★ = best tradeoff (Pareto optimal)"
            if use_llm:
                legend += " | [cyan]cyan[/cyan]=improvement [green]green[/green]=<5% [yellow]yellow[/yellow]=<15% [red]red[/red]=≥15%"
            legend += "[/dim]"
            console.print(legend)
            console.print()
