import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class CorpusDocument:
    id: str
    title: str
    text: str


@lru_cache(maxsize=16)
def load_corpus(path: Path | str) -> tuple[CorpusDocument, ...]:
    corpus_path = Path(path).resolve()
    if not corpus_path.exists():
        return ()

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return tuple(
            CorpusDocument(
                id=str(item.get("id", index)),
                title=str(item.get("title", "")),
                text=str(item.get("text", item.get("content", ""))),
            )
            for index, item in enumerate(raw)
        )

    if isinstance(raw, dict):
        return tuple(
            CorpusDocument(id=str(key), title=str(key), text=str(value))
            for key, value in raw.items()
        )

    raise ValueError("Corpus must be a JSON list or object.")
