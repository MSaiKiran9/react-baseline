import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from config.config import Settings, get_settings
from evaluation.evaluate import evaluate as evaluate_dataset
from evaluation.logger import write_trace
from llm.base import LLMError
from llm.ollama_client import OllamaLLM
from react.agent import ReActAgent
from tools.lookup_tool import LookupTool
from tools.search_tool import SearchTool


app = typer.Typer(help="Vanilla ReAct baseline using a local Ollama model.")
console = Console()


def build_agent(settings: Settings) -> ReActAgent:
    if settings.model_provider.lower() != "ollama":
        raise typer.BadParameter(f"Unsupported MODEL_PROVIDER: {settings.model_provider}")

    tools = [
        SearchTool(settings.corpus_path),
        LookupTool(settings.corpus_path),
    ]
    return ReActAgent(
        llm=OllamaLLM(settings),
        tools=tools,
        settings=settings,
    )


@app.command()
def ask(
    question: str,
    corpus_path: Path | None = typer.Option(None, help="Path to local JSON corpus."),
    model_name: str | None = typer.Option(None, help="Override the Ollama model name."),
    max_iterations: int | None = typer.Option(None, min=1, help="Override maximum ReAct iterations."),
    max_parse_retries: int | None = typer.Option(None, min=0, help="Override malformed-response retry count."),
    max_output_tokens: int | None = typer.Option(None, min=32, help="Override maximum generated tokens per LLM call."),
    ollama_timeout: float | None = typer.Option(None, min=1.0, help="Override Ollama request timeout in seconds."),
    temperature: float | None = typer.Option(None, min=0.0, help="Override model temperature."),
    top_p: float | None = typer.Option(None, min=0.0, max=1.0, help="Override model top-p."),
) -> None:
    settings = get_settings()
    if corpus_path is not None:
        settings.corpus_path = corpus_path
    if model_name is not None:
        settings.model_name = model_name
    if max_iterations is not None:
        settings.max_iterations = max_iterations
    if max_parse_retries is not None:
        settings.max_parse_retries = max_parse_retries
    if max_output_tokens is not None:
        settings.max_output_tokens = max_output_tokens
    if ollama_timeout is not None:
        settings.ollama_timeout = ollama_timeout
    if temperature is not None:
        settings.temperature = temperature
    if top_p is not None:
        settings.top_p = top_p
    agent = build_agent(settings)
    started = time.perf_counter()
    try:
        state = agent.run(question)
    except LLMError as exc:
        console.print(f"[bold red]Could not complete the LLM call.[/bold red] {exc}")
        console.print(
            "[dim]For this laptop, try: python main.py ask \"your question\" "
            "--model-name qwen2.5:3b --max-iterations 3 --ollama-timeout 300[/dim]"
        )
        raise typer.Exit(code=1) from exc
    latency = time.perf_counter() - started
    trace_path = write_trace(state, Path("logs"))

    console.print(f"[bold]Answer:[/bold] {state.final_answer}")
    console.print(
        f"[dim]Model: {settings.model_name} | Iterations: {state.iteration_count} | "
        f"Latency: {latency:.2f}s | Trace: {trace_path}[/dim]"
    )


@app.command("evaluate")
def evaluate_command(
    dataset_path: Path | None = typer.Option(None, help="Dataset JSON path."),
    output_path: Path | None = typer.Option(None, help="Evaluation output JSON path."),
    corpus_path: Path | None = typer.Option(None, help="Path to local JSON corpus."),
    model_name: str | None = typer.Option(None, help="Override the Ollama model name."),
    max_iterations: int | None = typer.Option(None, min=1, help="Override maximum ReAct iterations."),
    max_parse_retries: int | None = typer.Option(None, min=0, help="Override malformed-response retry count."),
    max_output_tokens: int | None = typer.Option(None, min=32, help="Override maximum generated tokens per LLM call."),
    ollama_timeout: float | None = typer.Option(None, min=1.0, help="Override Ollama request timeout in seconds."),
) -> None:
    settings = get_settings()
    if dataset_path is not None:
        settings.dataset_path = dataset_path
    if output_path is not None:
        settings.output_path = output_path
    if corpus_path is not None:
        settings.corpus_path = corpus_path
    if model_name is not None:
        settings.model_name = model_name
    if max_iterations is not None:
        settings.max_iterations = max_iterations
    if max_parse_retries is not None:
        settings.max_parse_retries = max_parse_retries
    if max_output_tokens is not None:
        settings.max_output_tokens = max_output_tokens
    if ollama_timeout is not None:
        settings.ollama_timeout = ollama_timeout

    try:
        payload = evaluate_dataset(build_agent(settings), settings.dataset_path, settings.output_path)
    except LLMError as exc:
        console.print(f"[bold red]Could not complete the LLM call.[/bold red] {exc}")
        console.print(
            "[dim]For this laptop, try --model-name qwen2.5:3b --max-iterations 3 --ollama-timeout 300.[/dim]"
        )
        raise typer.Exit(code=1) from exc
    metrics = payload["metrics"]

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for name, value in metrics.items():
        table.add_row(name, f"{value:.4f}" if isinstance(value, float) else str(value))
    console.print(table)
    console.print(f"[dim]Saved results to {settings.output_path}[/dim]")


if __name__ == "__main__":
    app()
