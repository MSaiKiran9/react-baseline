import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    average_latency: float
    average_iterations: float


def normalize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def exact_match(prediction: str, expected: str) -> bool:
    return " ".join(normalize(prediction)) == " ".join(normalize(expected))


def token_scores(prediction: str, expected: str) -> tuple[float, float, float]:
    predicted_tokens = normalize(prediction)
    expected_tokens = normalize(expected)
    if not predicted_tokens and not expected_tokens:
        return 1.0, 1.0, 1.0
    if not predicted_tokens or not expected_tokens:
        return 0.0, 0.0, 0.0

    expected_counts: dict[str, int] = {}
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1

    overlap = 0
    for token in predicted_tokens:
        if expected_counts.get(token, 0) > 0:
            overlap += 1
            expected_counts[token] -= 1

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def aggregate(results: list[dict[str, object]]) -> Metrics:
    if not results:
        return Metrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    accuracy = sum(1 for item in results if item["exact_match"]) / len(results)
    precision = sum(float(item["precision"]) for item in results) / len(results)
    recall = sum(float(item["recall"]) for item in results) / len(results)
    f1 = sum(float(item["f1"]) for item in results) / len(results)
    latency = sum(float(item["latency_seconds"]) for item in results) / len(results)
    iterations = sum(int(item["iterations"]) for item in results) / len(results)
    return Metrics(accuracy, precision, recall, f1, latency, iterations)
