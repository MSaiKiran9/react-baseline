from pathlib import Path
import re

from tools.base import BaseTool, Observation
from tools.corpus import CorpusDocument, load_corpus


class LookupTool(BaseTool):
    name = "Lookup"
    description = "Look up a local corpus document by id, title, or keyword."

    def __init__(self, corpus_path: Path):
        self.documents = load_corpus(corpus_path)
        self.by_id = {doc.id.lower(): doc for doc in self.documents}
        self.by_title = {doc.title.lower(): doc for doc in self.documents}
        self._cache: dict[str, str] = {}

    def run(self, tool_input: str) -> Observation:
        query = _clean_query(tool_input)
        cache_key = query.lower()
        if cache_key in self._cache:
            return Observation(self.name, tool_input, self._cache[cache_key])

        if not query:
            output = "Lookup query is empty."
        else:
            doc = self._find_document(query)
            output = "No matching document found." if doc is None else f"[{doc.id}] {doc.title}: {self._best_sentences(doc.text, query)}"
        self._cache[cache_key] = output
        return Observation(self.name, tool_input, output)

    def _find_document(self, query: str) -> CorpusDocument | None:
        normalized = query.lower()
        if normalized in self.by_id:
            return self.by_id[normalized]
        if normalized in self.by_title:
            return self.by_title[normalized]

        query_tokens = set(_tokenize(query))
        best_score = 0
        best_doc = None
        for doc in self.documents:
            haystack = (doc.title + " " + doc.text).lower()
            if normalized in haystack:
                return doc
            score = len(query_tokens & set(_tokenize(doc.title + " " + doc.text)))
            if score > best_score:
                best_score = score
                best_doc = doc
        return best_doc if best_score > 0 else None

    def _best_sentences(self, text: str, query: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        query_tokens = set(_tokenize(query))
        scored = []
        for sent in sentences:
            score = len(set(_tokenize(sent)) & query_tokens)
            scored.append((score, sent.strip()))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = [sentence for score, sentence in scored[:5] if sentence]
        if not best:
            best = [sentence.strip() for sentence in sentences[:5]]
        return " ".join(best)[:1400]


def _clean_query(text: str) -> str:
    return text.strip().strip("[](){}'\"")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
