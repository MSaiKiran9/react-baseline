import json
from pathlib import Path

from tools.lookup_tool import LookupTool
from tools.search_tool import SearchTool


def write_corpus(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "id": "doc-1",
                    "title": "ReAct",
                    "text": "ReAct combines reasoning traces and actions.",
                }
            ]
        ),
        encoding="utf-8",
    )


def test_search_tool_finds_matching_document(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus)

    observation = SearchTool(corpus).run("reasoning")

    assert observation.tool_name == "Search"
    assert "doc-1" in observation.output


def test_lookup_tool_finds_document_by_id(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    write_corpus(corpus)

    observation = LookupTool(corpus).run("doc-1")

    assert observation.tool_name == "Lookup"
    assert "combines reasoning" in observation.output
