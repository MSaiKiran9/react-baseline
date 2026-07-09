import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from tools.base import BaseTool, Observation
from tools.corpus import CorpusDocument, load_corpus


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "with", "won", "did", "does", "do", "has", "have",
}


@dataclass(frozen=True)
class SearchIndex:
    documents: tuple[CorpusDocument, ...]
    doc_tokens: tuple[Counter[str], ...]
    doc_freq: Counter[str]
    doc_lengths: tuple[int, ...]
    avg_doc_len: float


class SearchTool(BaseTool):
    name = "Search"
    description = "Search the local corpus using cached BM25 ranking."

    def __init__(self, corpus_path: Path, max_results: int = 5):
        self.max_results = max_results
        self.index = _get_search_index(str(Path(corpus_path).resolve()))
        self.documents = self.index.documents
        self._cache: dict[str, str] = {}

    def run(self, tool_input: str) -> Observation:
        cache_key = tool_input.strip().lower()
        if cache_key in self._cache:
            return Observation(self.name, tool_input, self._cache[cache_key])

        query = _tokenize_query(tool_input)
        if not query:
            output = "Search query is empty."
            self._cache[cache_key] = output
            return Observation(self.name, tool_input, output)

        scores = [(self._bm25(query, idx), doc) for idx, doc in enumerate(self.documents)]
        scores.sort(key=lambda item: item[0], reverse=True)
        matches = [doc for score, doc in scores if score > 0][: self.max_results]

        if not matches:
            output = "No matching documents found."
        else:
            output = "\n".join(
                f"[{doc.id}] {doc.title}: {self._best_snippet(doc.text, query)}"
                for doc in matches
            )
        self._cache[cache_key] = output
        return Observation(self.name, tool_input, output)

    def _bm25(self, query: list[str], doc_index: int, k1: float = 1.5, b: float = 0.75) -> float:
        score = 0.0
        tf = self.index.doc_tokens[doc_index]
        dl = self.index.doc_lengths[doc_index]
        if self.index.avg_doc_len == 0:
            return 0.0

        for term in query:
            if term not in tf:
                continue
            df = self.index.doc_freq.get(term, 0)
            idf = math.log(1 + (len(self.documents) - df + 0.5) / (df + 0.5))
            freq = tf[term]
            denominator = freq + k1 * (1 - b + b * dl / self.index.avg_doc_len)
            score += idf * (freq * (k1 + 1)) / denominator
        return score

    def _best_snippet(self, text: str, query: list[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        query_terms = set(query)
        ranked = []
        for sent in sentences:
            tokens = set(_tokenize(sent))
            ranked.append((len(tokens & query_terms), sent.strip()))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [sent for score, sent in ranked[:3] if sent]
        snippet = " ".join(selected) if selected else text[:900]
        return snippet[:900]


@lru_cache(maxsize=8)
def _get_search_index(corpus_path: str) -> SearchIndex:
    documents = load_corpus(corpus_path)
    doc_tokens = []
    doc_freq: Counter[str] = Counter()
    doc_lengths = []

    for doc in documents:
        tokens = _tokenize(doc.title + " " + doc.text)
        counter = Counter(tokens)
        doc_tokens.append(counter)
        doc_lengths.append(len(tokens))
        for token in set(tokens):
            doc_freq[token] += 1

    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return SearchIndex(
        documents=documents,
        doc_tokens=tuple(doc_tokens),
        doc_freq=doc_freq,
        doc_lengths=tuple(doc_lengths),
        avg_doc_len=avg_doc_len,
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _tokenize_query(text: str) -> list[str]:
    tokens = [token for token in _tokenize(text) if token not in STOPWORDS]
    return tokens or _tokenize(text)
