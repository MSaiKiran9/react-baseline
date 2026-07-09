from pathlib import Path
import re

from tools.base import BaseTool, Observation
from tools.corpus import load_corpus


class LookupTool(BaseTool):
    name = "Lookup"
    description = "Look up a local corpus document by id, title, or keyword."

    def __init__(self, corpus_path: Path):
        self.documents = load_corpus(corpus_path)

    def run(self, tool_input: str) -> Observation:
        query = tool_input.strip().lower()

        if not query:
            return Observation(
                self.name,
                tool_input,
                "Lookup query is empty.",
            )

        doc = self._find_document(query)

        if doc is None:
            return Observation(
                self.name,
                tool_input,
                "No matching document found.",
            )

        snippet = self._best_sentences(doc.text, query)

        return Observation(
            self.name,
            tool_input,
            f"[{doc.id}] {doc.title}: {snippet}",
        )

    def _find_document(self, query):
        for doc in self.documents:
            if query == doc.id.lower() or query == doc.title.lower():
                return doc

        for doc in self.documents:
            haystack = (doc.title + " " + doc.text).lower()
            if query in haystack:
                return doc

        return None

    def _best_sentences(self, text, query):

        sentences = re.split(r'(?<=[.!?])\s+', text)

        query_tokens = set(self._tokenize(query))

        scored = []

        for sent in sentences:

            sent_tokens = set(self._tokenize(sent))

            score = len(sent_tokens & query_tokens)

            scored.append((score, sent))

        scored.sort(reverse=True)

        best = [
            s.strip()
            for score, s in scored[:5]
            if score > 0
        ]

        if not best:
            best = sentences[:5]

        return " ".join(best)

    @staticmethod
    def _tokenize(text):
        return re.findall(r"[a-z0-9]+", text.lower())