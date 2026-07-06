import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    title: str
    text: str


def load_corpus(path: Path) -> list[CorpusDocument]:
    if not path.exists():
        return []

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [
            CorpusDocument(
                id=str(item.get("id", index)),
                title=str(item.get("title", "")),
                text=str(item.get("text", item.get("content", ""))),
            )
            for index, item in enumerate(raw)
        ]

    if isinstance(raw, dict):
        return [
            CorpusDocument(id=str(key), title=str(key), text=str(value))
            for key, value in raw.items()
        ]

    raise ValueError("Corpus must be a JSON list or object.")
