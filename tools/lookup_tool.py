from pathlib import Path

from tools.base import BaseTool, Observation
from tools.corpus import load_corpus


class LookupTool(BaseTool):
    name = "Lookup"
    description = "Look up a local corpus document by id, title, or keyword."

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self.documents = load_corpus(corpus_path)

    def run(self, tool_input: str) -> Observation:
        needle = tool_input.strip().lower()
        if not needle:
            return Observation(self.name, tool_input, "Lookup query is empty.")

        for document in self.documents:
            if needle in {document.id.lower(), document.title.lower()}:
                return Observation(self.name, tool_input, f"[{document.id}] {document.title}: {document.text}")

        for document in self.documents:
            haystack = f"{document.title} {document.text}".lower()
            if needle in haystack:
                return Observation(self.name, tool_input, f"[{document.id}] {document.title}: {document.text}")

        return Observation(self.name, tool_input, "No matching document found.")
