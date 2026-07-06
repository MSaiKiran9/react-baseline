import pytest

from react.parser import ParsedAction, ParsedFinish, ReActParser, ParseError


def test_parse_action() -> None:
    parsed = ReActParser().parse("Thought:\nI should search.\n\nAction:\nSearch(\"ReAct\")")

    assert isinstance(parsed, ParsedAction)
    assert parsed.thought == "I should search."
    assert parsed.action_name == "Search"
    assert parsed.action_input == "ReAct"


def test_parse_finish() -> None:
    parsed = ReActParser().parse("Thought:\nI know it.\n\nFinish:\nReason and Act")

    assert isinstance(parsed, ParsedFinish)
    assert parsed.answer == "Reason and Act"


def test_parse_invalid_response() -> None:
    with pytest.raises(ParseError):
        ReActParser().parse("No sections here")


def test_parse_strips_think_blocks_and_preamble() -> None:
    parsed = ReActParser().parse(
        "<think>private reasoning</think>\nHere is the response:\nThought:\nDone.\n\nFinish:\nAnswer"
    )

    assert isinstance(parsed, ParsedFinish)
    assert parsed.thought == "Done."
    assert parsed.answer == "Answer"
