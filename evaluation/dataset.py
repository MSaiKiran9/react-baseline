import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSample:
    question: str
    answer: str


def load_dataset(path: Path) -> list[DatasetSample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Dataset must be a JSON array.")
    return [DatasetSample(question=str(item["question"]), answer=str(item["answer"])) for item in raw]
