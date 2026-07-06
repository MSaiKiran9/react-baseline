import json
import time
from pathlib import Path

from evaluation.dataset import load_dataset
from evaluation.metrics import aggregate, exact_match, token_scores
from react.agent import ReActAgent


def evaluate(agent: ReActAgent, dataset_path: Path, output_path: Path) -> dict[str, object]:
    samples = load_dataset(dataset_path)
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

    metrics = aggregate(results)
    payload = {
        "metrics": metrics.__dict__,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
