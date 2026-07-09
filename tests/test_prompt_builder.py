from pathlib import Path

from react.agent import PromptBuilder
from tools.base import BaseTool, Observation


class DummyTool(BaseTool):
    name = "Search"
    description = "Search things."

    def run(self, tool_input: str) -> Observation:
        return Observation(self.name, tool_input, "ok")


def test_prompt_builder_includes_question_and_tools() -> None:
    messages = PromptBuilder(Path("prompts/react_prompt.txt")).build("What is ReAct?", [DummyTool()])

    assert len(messages) == 1
    assert "What is ReAct?" in messages[0].content
    assert "Search things." in messages[0].content
    assert "Current date:" in messages[0].content
    assert "Finish must contain only the answer phrase" in messages[0].content
    assert "Finish:" in messages[0].content
