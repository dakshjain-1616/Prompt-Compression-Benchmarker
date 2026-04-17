"""Typer CLI for Prompt Compression Benchmarker."""

import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from pcb.compressors import ALL_COMPRESSORS
from pcb.tasks import ALL_TASKS
from pcb.runner import run_benchmark
from pcb.reporters import TerminalReporter, JSONReporter, CSVReporter, HTMLReporter
from pcb.evaluators.llm_judge import SUPPORTED_MODELS, DEFAULT_MODEL

app = typer.Typer(
    name="pcb",
    help=(
        "Prompt Compression Benchmarker — compare prompt compressors on "
        "accuracy vs token reduction. Supports optional LLM-as-judge scoring "
        "via OpenRouter (set OPENROUTER_API_KEY)."
    ),
    add_completion=False,
)
console = Console()

_MODEL_PRICES = {
    "claude-opus-4-7":   15.00,
    "claude-opus-4-6":   15.00,
    "claude-sonnet-4-6":  3.00,
    "claude-haiku-4-5":   0.80,
    "gpt-5":              5.00,
    "gpt-5.4":            5.00,
    "gpt-4.1":            2.00,
    "gpt-4.1-mini":       0.40,
    "gpt-4o":             2.50,
    "o3":                10.00,
    "o4-mini":            1.10,
    "codex-mini-latest":  1.50,
    "gemini-2.5-pro":     1.25,
    "gemini-2.5-flash":   0.15,
    "deepseek-v3.2":      0.27,
}

_DEFAULT_COMPRESSORS = ["no_compression", "tfidf", "selective_context", "llmlingua", "llmlingua2"]
_DEFAULT_TASKS = ["rag", "summarization", "coding"]
_BUNDLED_DATA_DIR = Path(__file__).parent / "data"          # installed package: src/pcb/data/
_DEV_DATA_DIR = Path(__file__).parent.parent.parent / "data"  # dev checkout: repo root/data/


def _resolve_data_dir(data_dir: Optional[Path]) -> Path:
    if data_dir and data_dir.exists():
        return data_dir
    if _BUNDLED_DATA_DIR.exists():
        return _BUNDLED_DATA_DIR
    if _DEV_DATA_DIR.exists():
        return _DEV_DATA_DIR
    return Path("data")


def _render_cost_section(report, daily_tokens: int, price_per_million: float, model_label: str) -> None:
    from pcb.runner import BenchmarkReport
    monthly_base = daily_tokens / 1_000_000 * price_per_million * 30

    table = Table(
        title=f"[bold]Monthly Cost Projection[/bold]  "
              f"[dim]{model_label} · ${price_per_million}/1M tokens · "
              f"{daily_tokens/1e6:.1f}M tokens/day[/dim]",
        box=box.ROUNDED,
        header_style="bold magenta",
    )
    table.add_column("Compressor", style="bold white", min_width=20)
    table.add_column("Avg Reduction", justify="right")
    table.add_column("Tokens Saved/Day", justify="right")
    table.add_column("Monthly Savings", justify="right")
    table.add_column("Annual Savings", justify="right")
    table.add_column("Quality Drop", justify="right")

    # Average reduction across all tasks per compressor
    from collections import defaultdict
    reductions: dict = defaultdict(list)
    quality_drops: dict = defaultdict(list)
    for s in report.summaries:
        reductions[s.compressor_name].append(s.avg_token_reduction_pct)
        if s.avg_llm_drop_pct is not None:
            quality_drops[s.compressor_name].append(s.avg_llm_drop_pct)
        else:
            quality_drops[s.compressor_name].append(s.avg_quality_drop_pct)

    baseline_monthly = monthly_base
    for cname in [s.compressor_name for s in report.summaries if s.task_type == report.task_types[0]]:
        if cname == "no_compression":
            table.add_row(
                cname, "0.0%", "0",
                f"${baseline_monthly:,.0f}/mo (baseline)",
                "—", "—",
            )
            continue

        avg_reduction = sum(reductions[cname]) / len(reductions[cname])
        avg_drop = sum(quality_drops[cname]) / len(quality_drops[cname])
        tokens_saved_daily = int(daily_tokens * avg_reduction / 100)
        monthly_savings = tokens_saved_daily / 1_000_000 * price_per_million * 30
        annual_savings = monthly_savings * 12

        drop_color = "green" if avg_drop < 5 else ("yellow" if avg_drop < 15 else "red")
        savings_color = "bold green" if monthly_savings > 0 else "white"
        drop_str = f"[{drop_color}]{avg_drop:+.1f}%[/{drop_color}]"
        savings_str = f"[{savings_color}]${monthly_savings:,.0f}[/{savings_color}]"

        table.add_row(
            cname,
            f"{avg_reduction:.1f}%",
            f"{tokens_saved_daily:,}",
            savings_str,
            f"${annual_savings:,.0f}",
            drop_str,
        )

    console.print()
    console.print(table)
    console.print(
        "  [dim]Savings = tokens reduced × price/token × 30 days. "
        "Quality drop averaged across all task types.[/dim]"
    )
    console.print()


@app.command()
def compress(
    input_file: Optional[Path] = typer.Argument(
        None, help="File to compress. Reads from stdin if not provided."
    ),
    compressor: str = typer.Option(
        "tfidf", "--compressor", "-c",
        help="Compressor: tfidf, selective_context, llmlingua, llmlingua2.",
    ),
    rate: float = typer.Option(
        0.45, "--rate", "-r",
        help="Fraction of tokens to remove (0.0–1.0). Default: 0.45.",
    ),
    output_file: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Write compressed text to file instead of stdout.",
    ),
    stats: bool = typer.Option(
        False, "--stats", "-s",
        help="Print token stats to stderr.",
    ),
) -> None:
    """Compress text from a file or stdin and write to stdout.

    Examples:

        cat context.txt | pcb compress

        pcb compress context.txt --compressor llmlingua2 --rate 0.4

        pcb compress context.txt -o compressed.txt --stats

        cat prompt.txt | pcb compress | python my_api_script.py
    """
    if compressor not in ALL_COMPRESSORS or compressor == "no_compression":
        console.print(f"[red]Unknown compressor '{compressor}'. "
                      f"Valid: {[k for k in ALL_COMPRESSORS if k != 'no_compression']}[/red]")
        raise typer.Exit(1)

    if input_file:
        if not input_file.exists():
            console.print(f"[red]File not found: {input_file}[/red]")
            raise typer.Exit(1)
        text = input_file.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            console.print("[yellow]Reading from stdin… (pipe text in or press Ctrl-D)[/yellow]",
                          file=sys.stderr)
        text = sys.stdin.read()

    if not text.strip():
        console.print("[red]No input text provided.[/red]")
        raise typer.Exit(1)

    c = ALL_COMPRESSORS[compressor]()
    c.initialize()
    result = c.compress(text, rate=rate)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.compressed_text, encoding="utf-8")
    else:
        print(result.compressed_text)

    if stats or output_file:
        saved = result.original_tokens - result.compressed_tokens
        reduction = result.compression_ratio * 100
        print(
            f"[pcb compress] {result.original_tokens} → {result.compressed_tokens} tokens  "
            f"({saved} saved, {reduction:.1f}% reduction)  "
            f"compressor={compressor} rate={rate}",
            file=sys.stderr,
        )


@app.command()
def run(
    compressors: Optional[List[str]] = typer.Option(
        None, "--compressor", "-c",
        help="Compressors to benchmark. Repeat for multiple. Default: all.",
    ),
    tasks: Optional[List[str]] = typer.Option(
        None, "--task", "-t",
        help="Task types: rag, summarization, coding. Default: all.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o",
        help="Output file (.json / .csv / .html). Default: terminal only.",
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", "-d",
        help="Directory containing *_samples.jsonl files.",
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", "-n",
        help="Max samples per task.",
    ),
    rate: float = typer.Option(
        0.5, "--rate", "-r",
        help="Target compression rate 0.0–1.0. Default: 0.5",
    ),
    llm_judge: bool = typer.Option(
        False, "--llm-judge", "-j",
        help="Enable LLM-as-judge scoring via OpenRouter.",
    ),
    judge_model: str = typer.Option(
        DEFAULT_MODEL, "--judge-model", "-m",
        help=(
            f"Model alias or full OpenRouter ID for LLM judge. "
            f"Default: {DEFAULT_MODEL}. Run 'pcb list-models' for options."
        ),
    ),
    openrouter_key: Optional[str] = typer.Option(
        None, "--openrouter-key",
        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.",
    ),
    daily_tokens: Optional[int] = typer.Option(
        None, "--daily-tokens",
        help="Estimated daily input token volume for cost projection (e.g. 5000000).",
    ),
    cost_model: Optional[str] = typer.Option(
        None, "--cost-model",
        help=(
            "Model name for cost projection (e.g. claude-sonnet-4-6, gpt-4.1). "
            "Run 'pcb list-models' to see price data."
        ),
    ),
    token_price: Optional[float] = typer.Option(
        None, "--token-price",
        help="Override: price per million input tokens (USD). Overrides --cost-model lookup.",
    ),
) -> None:
    """Run the benchmark and display a comparison report."""
    selected_compressors = compressors or _DEFAULT_COMPRESSORS
    selected_tasks = tasks or _DEFAULT_TASKS
    resolved_data_dir = _resolve_data_dir(data_dir)

    invalid_c = [c for c in selected_compressors if c not in ALL_COMPRESSORS]
    if invalid_c:
        console.print(f"[red]Unknown compressors: {invalid_c}. Run 'pcb list-compressors'.[/red]")
        raise typer.Exit(1)
    invalid_t = [t for t in selected_tasks if t not in ALL_TASKS]
    if invalid_t:
        console.print(f"[red]Unknown tasks: {invalid_t}. Valid: {list(ALL_TASKS.keys())}[/red]")
        raise typer.Exit(1)

    # Set up LLM judge if requested
    judge = None
    if llm_judge:
        api_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            console.print(
                "[red]--llm-judge requires an OpenRouter API key.\n"
                "Set OPENROUTER_API_KEY env var or pass --openrouter-key.[/red]"
            )
            raise typer.Exit(1)
        try:
            from pcb.evaluators.llm_judge import LLMJudge
            judge = LLMJudge(model=judge_model, api_key=api_key)
            console.print(f"  [bold green]LLM Judge:[/bold green] [cyan]{judge.model_id}[/cyan]")
        except Exception as e:
            console.print(f"[red]Failed to init LLM judge: {e}[/red]")
            raise typer.Exit(1)

    console.print(f"\n[bold cyan]Running benchmark...[/bold cyan]")
    console.print(f"  Compressors : [yellow]{', '.join(selected_compressors)}[/yellow]")
    console.print(f"  Tasks       : [yellow]{', '.join(selected_tasks)}[/yellow]")
    console.print(f"  Data dir    : [yellow]{resolved_data_dir}[/yellow]")
    console.print(f"  Rate        : [yellow]{rate}[/yellow]")
    if max_samples:
        console.print(f"  Max samples : [yellow]{max_samples}[/yellow]")
    console.print()

    report = run_benchmark(
        compressor_names=selected_compressors,
        task_names=selected_tasks,
        data_dir=resolved_data_dir,
        max_samples=max_samples,
        rate=rate,
        judge=judge,
    )

    TerminalReporter().render(report)

    # Cost projection section
    if daily_tokens and (cost_model or token_price):
        price = token_price or _MODEL_PRICES.get(cost_model.lower(), None)
        if price is None:
            console.print(f"[yellow]Cost model '{cost_model}' not in price table. "
                          f"Use --token-price to specify $/1M manually.[/yellow]")
        else:
            model_label = cost_model or f"${token_price}/1M"
            _render_cost_section(report, daily_tokens, price, model_label)

    if output:
        suffix = output.suffix.lower()
        output.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".json":
            JSONReporter().render(report, output)
        elif suffix == ".csv":
            CSVReporter().render(report, output)
        elif suffix == ".html":
            HTMLReporter().render(report, output)
        else:
            console.print(f"[yellow]Unknown extension '{suffix}'. Supported: .json .csv .html[/yellow]")


@app.command("list-compressors")
def list_compressors() -> None:
    """List all available compressors."""
    table = Table(title="Available Compressors", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Name", style="bold white")
    table.add_column("Class")
    table.add_column("Strategy")
    descriptions = {
        "no_compression":   "Pass-through baseline (0% compression, quality ceiling)",
        "tfidf":            "TF-IDF sentence scoring — keeps highest-scoring sentences",
        "selective_context":"Greedy token-budget sentence selection (first-fit)",
        "llmlingua":        "Sentence-level coarse pruning (first+middle+last strategy)",
        "llmlingua2":       "Word-level stopword & low-content token pruning",
    }
    for name, cls in ALL_COMPRESSORS.items():
        table.add_row(name, cls.__name__, descriptions.get(name, ""))
    console.print(table)


@app.command("list-models")
def list_models() -> None:
    """List all supported LLM judge models for --judge-model."""
    table = Table(
        title="Supported LLM Judge Models (via OpenRouter)",
        box=box.ROUNDED,
        header_style="bold magenta",
        caption="Usage: pcb run --llm-judge --judge-model <alias>",
    )
    table.add_column("Alias", style="bold cyan")
    table.add_column("OpenRouter Model ID")
    table.add_column("Provider")

    providers = {
        "anthropic/": "Anthropic",
        "openai/": "OpenAI",
        "google/": "Google",
        "meta-llama/": "Meta",
        "mistralai/": "Mistral",
        "deepseek/": "DeepSeek",
        "x-ai/": "xAI",
        "qwen/": "Alibaba",
    }

    for alias, model_id in SUPPORTED_MODELS.items():
        provider = next((v for k, v in providers.items() if model_id.startswith(k)), "Other")
        marker = " [dim](default)[/dim]" if alias == DEFAULT_MODEL else ""
        table.add_row(f"{alias}{marker}", model_id, provider)

    console.print(table)
    console.print()
    console.print("[dim]Set OPENROUTER_API_KEY or pass --openrouter-key to use LLM judge.[/dim]")


@app.command("show-schema")
def show_schema(
    task: str = typer.Argument(help="Task type: rag / summarization / coding"),
) -> None:
    """Show the JSONL schema expected for a task's dataset."""
    schemas = {
        "rag": {
            "id": "rag_001",
            "context": "<long passage 300-1500 tokens>",
            "question": "<factoid question about the passage>",
            "answer": "<short answer string>",
        },
        "summarization": {
            "id": "sum_001",
            "article": "<news article or passage 300-800 tokens>",
            "summary": "<2-3 sentence reference summary>",
        },
        "coding": {
            "id": "code_001",
            "context": "<imports and helper code>",
            "docstring": "<function description>",
            "solution": "<Python function implementation>",
        },
    }
    if task not in schemas:
        console.print(f"[red]Unknown task '{task}'. Valid: {list(schemas.keys())}[/red]")
        raise typer.Exit(1)
    import json
    console.print(f"\n[bold]Schema for '{task}':[/bold]")
    console.print(json.dumps(schemas[task], indent=2))
    console.print(f"\n[dim]Each line in the .jsonl file is one JSON object matching this schema.[/dim]\n")


if __name__ == "__main__":
    app()
