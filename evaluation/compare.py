import ollama
import json
import time
from dataclasses import dataclass
from pathlib import Path

from config.config import Settings
from evaluation.dataset import DatasetSample, load_dataset
from evaluation.metrics import aggregate, exact_match, token_scores
from react.agent import ReActAgent
from openpyxl import Workbook


COMPARISON_COLUMNS = [
    "Model",
    "Dataset",
    "System",
    "n",
    "Accuracy",
    "Sem F1",
    "Token F1",
    "MAR",
    "MKG Matches / n",
    "Retrieval Failures",
    "Reasoning Failures",
    "Accuracy Improvement",
    "Sem F1 Change",
    "Avg Latency",
    "Avg Iterations",
    "Notes",
]


@dataclass(frozen=True)
class ComparisonRun:
    model: str
    dataset_name: str
    system_name: str
    sample_size: int
    metrics: dict[str, float]
    retrieval_failures: int
    reasoning_failures: int
    notes: str

    def row(self) -> dict[str, str]:
        return {
            "Model": self.model,
            "Dataset": self.dataset_name,
            "System": self.system_name,
            "n": str(self.sample_size),
            "Accuracy": f"{self.metrics['accuracy']:.3f}",
            "Sem F1": "N/A",
            "Token F1": f"{self.metrics['f1']:.3f}",
            "MAR": "N/A",
            "MKG Matches / n": "N/A",
            "Retrieval Failures": str(self.retrieval_failures),
            "Reasoning Failures": str(self.reasoning_failures),
            "Accuracy Improvement": "N/A",
            "Sem F1 Change": "N/A",
            "Avg Latency": f"{self.metrics['average_latency']:.2f}s",
            "Avg Iterations": f"{self.metrics['average_iterations']:.2f}",
            "Notes": self.notes,
        }


def compare_models(
    agent_factory,
    settings: Settings,
    dataset_path: Path,
    output_path: Path,
    models: list[str],
    sample_sizes: list[int],
    dataset_name: str,
    system_name: str = "ReAct Baseline",
) -> dict[str, object]:
    dataset = load_dataset(dataset_path)
    client = ollama.Client(host=settings.ollama_url)
    runs: list[ComparisonRun] = []
    detailed_results: list[dict[str, object]] = []

    for sample_size in sample_sizes:
        samples = dataset[:sample_size]
        if len(samples) < sample_size:
            raise ValueError(
                f"Requested sample size n={sample_size}, but dataset only has {len(dataset)} samples."
            )

        for model in models:
            # Unload any previously loaded models
            try:
                running = client.ps()

                # Handle both dict and object return types
                models_loaded = running.get("models", []) if isinstance(running, dict) else running.models

                for m in models_loaded:
                    name = m["name"] if isinstance(m, dict) else m.model

                    try:
                        client.generate(
                            model=name,
                            prompt="",
                            keep_alive=0,
                        )
                    except Exception:
                        pass

            except Exception:
                pass
            settings.model_name = model
            agent = agent_factory(settings)
            run_results = _evaluate_samples(agent, samples)
            metrics = aggregate(run_results)
            retrieval_failures, reasoning_failures = _classify_failures(run_results)
            notes = "Ollama local baseline; non-ReAct columns are N/A"
            runs.append(
                ComparisonRun(
                    model=model,
                    dataset_name=dataset_name,
                    system_name=system_name,
                    sample_size=sample_size,
                    metrics=metrics.__dict__,
                    retrieval_failures=retrieval_failures,
                    reasoning_failures=reasoning_failures,
                    notes=notes,
                )
            )
            detailed_results.append(
    {
        "model": model,
        "dataset": dataset_name,
        "system": system_name,
        "sample_size": sample_size,
        "metrics": metrics.__dict__,
        "retrieval_failures": retrieval_failures,
        "reasoning_failures": reasoning_failures,
    }
)
            # detailed_results.append(
            #     {
            #         "model": model,
            #         "dataset": dataset_name,
            #         "system": system_name,
            #         "sample_size": sample_size,
            #         "metrics": metrics.__dict__,
            #         "retrieval_failures": retrieval_failures,
            #         "reasoning_failures": reasoning_failures,
            #         "results": run_results,
            #     }
            # )
            

    payload = {
        "columns": COMPARISON_COLUMNS,
        "runs": [run.row() for run in runs],
        "details": detailed_results,
    }
    _write_outputs(payload, output_path)
    return payload
    


def _evaluate_samples(agent: ReActAgent, samples: list[DatasetSample]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for sample in samples:
        started = time.perf_counter()
        state = agent.run(sample.question)
        latency = time.perf_counter() - started
        prediction = state.final_answer or ""
        precision, recall, f1 = token_scores(prediction, sample.answer)
        results.append(
            {
                "question": sample.question,
                "expected_answer": sample.answer,
                "prediction": prediction,
                "exact_match": exact_match(prediction, sample.answer),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency_seconds": latency,
                "iterations": state.iteration_count,
                "trajectory": state.trajectory(),
            }
        )
    return results


def _classify_failures(results: list[dict[str, object]]) -> tuple[int, int]:
    retrieval_failures = 0
    reasoning_failures = 0

    for result in results:
        if result["exact_match"]:
            continue
        trajectory = result.get("trajectory", [])
        observations = [
            str(step.get("observation", ""))
            for step in trajectory
            if isinstance(step, dict) and step.get("observation")
        ]
        if any("No matching documents found." in observation for observation in observations):
            retrieval_failures += 1
        elif observations:
            reasoning_failures += 1

    return retrieval_failures, reasoning_failures


# def _write_outputs(payload: dict[str, object], output_path: Path) -> None:
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(format_comparison_table(payload["runs"]), encoding="utf-8")
#     json_path = output_path.with_suffix(".json")
#     json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _write_outputs(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TXT table
    output_path.write_text(
        format_comparison_table(payload["runs"]),
        encoding="utf-8",
    )

    # Excel summary
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Model Comparison"

    # Header
    sheet.append(COMPARISON_COLUMNS)

    # Rows
    for row in payload["runs"]:
        sheet.append([row[col] for col in COMPARISON_COLUMNS])

    workbook.save(output_path.with_suffix(".xlsx"))

    # Optional JSON (only summary)
    summary = {
        "columns": payload["columns"],
        "runs": payload["runs"],
    }

    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def format_comparison_table(rows: object) -> str:
    typed_rows = list(rows)
    if not typed_rows:
        return "No comparison runs were produced.\n"

    widths = {
        column: max(len(column), *(len(str(row[column])) for row in typed_rows))
        for column in COMPARISON_COLUMNS
    }
    separator = " | "
    header = separator.join(column.ljust(widths[column]) for column in COMPARISON_COLUMNS)
    divider = "-+-".join("-" * widths[column] for column in COMPARISON_COLUMNS)
    body = [
        separator.join(str(row[column]).ljust(widths[column]) for column in COMPARISON_COLUMNS)
        for row in typed_rows
    ]
    return "\n".join([header, divider, *body]) + "\n"
