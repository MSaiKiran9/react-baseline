import math
import re
from collections import Counter
from pathlib import Path

from tools.base import BaseTool, Observation
from tools.corpus import CorpusDocument, load_corpus


class SearchTool(BaseTool):
    name = "Search"
    description = "Search the local corpus using BM25 ranking."

    def __init__(self, corpus_path: Path, max_results: int = 3):
        self.max_results = max_results
        self.documents = load_corpus(corpus_path)

        self.doc_tokens = []
        self.doc_freq = Counter()
        self.doc_lengths = []

        for doc in self.documents:
            tokens = self._tokenize(doc.title + " " + doc.text)
            self.doc_tokens.append(Counter(tokens))
            self.doc_lengths.append(len(tokens))

            for token in set(tokens):
                self.doc_freq[token] += 1

        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths
            else 0
        )

        self.N = len(self.documents)

    def run(self, tool_input: str) -> Observation:
        query = self._tokenize(tool_input)

        scores = []

        for idx, doc in enumerate(self.documents):
            score = self._bm25(query, idx)
            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)

        matches = [doc for score, doc in scores if score > 0][: self.max_results]

        if not matches:
            return Observation(
                self.name,
                tool_input,
                "No matching documents found.",
            )

        lines = []

        for doc in matches:
            snippet = self._best_snippet(doc.text, query)
            lines.append(f"[{doc.id}] {doc.title}: {snippet}")

        return Observation(
            self.name,
            tool_input,
            "\n".join(lines),
        )

    def _bm25(self, query, doc_index, k1=1.5, b=0.75):
        score = 0

        tf = self.doc_tokens[doc_index]
        dl = self.doc_lengths[doc_index]

        for term in query:

            if term not in tf:
                continue

            df = self.doc_freq.get(term, 0)

            idf = math.log(
                1 + (self.N - df + 0.5) / (df + 0.5)
            )

            freq = tf[term]

            numerator = freq * (k1 + 1)

            denominator = freq + k1 * (
                1 - b + b * dl / self.avg_doc_len
            )

            score += idf * numerator / denominator

        return score

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())

    def _best_snippet(self, text, query):

        sentences = re.split(r"(?<=[.!?])\s+", text)

        best = ""
        best_score = -1

        for sent in sentences:

            tokens = set(self._tokenize(sent))

            score = len(tokens.intersection(query))

            if score > best_score:
                best_score = score
                best = sent

        if not best:
            return text[:600]

        return best[:600]