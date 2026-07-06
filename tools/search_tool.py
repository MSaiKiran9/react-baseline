from pathlib import Path

from tools.base import BaseTool, Observation
from tools.corpus import CorpusDocument, load_corpus


class SearchTool(BaseTool):
    name = "Search"
    description = "Search the local text corpus and return matching document summaries."

    def __init__(self, corpus_path: Path, max_results: int = 5) -> None:
        self.corpus_path = corpus_path
        self.max_results = max_results
        self.documents = load_corpus(corpus_path)

    def run(self, tool_input: str) -> Observation:
        query_terms = self._terms(tool_input)
        scored = [
            (self._score(document, query_terms), document)
            for document in self.documents
        ]
        matches = [document for score, document in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]

        if not matches:
            return Observation(self.name, tool_input, "No matching documents found.")

        lines = []
        for document in matches[: self.max_results]:
            snippet = document.text[:240].replace("\n", " ").strip()
            lines.append(f"[{document.id}] {document.title}: {snippet}")
        return Observation(self.name, tool_input, "\n".join(lines))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {term.lower() for term in text.replace(",", " ").replace(".", " ").split() if term.strip()}

    def _score(self, document: CorpusDocument, query_terms: set[str]) -> int:
        haystack = f"{document.title} {document.text}".lower()
        return sum(1 for term in query_terms if term in haystack)
