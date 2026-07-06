from llm.base import BaseLLM
from react.loop import ReActLoop
from react.message import Message
from react.parser import ReActParser
from react.state import ReActState
from tools.base import BaseTool, Observation


class ScriptedLLM(BaseLLM):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.index = 0

    def generate(self, messages: list[Message]) -> str:
        output = self.outputs[self.index]
        self.index += 1
        return output


class EchoSearch(BaseTool):
    name = "Search"
    description = "Echo search."

    def run(self, tool_input: str) -> Observation:
        return Observation(self.name, tool_input, f"found {tool_input}")


def test_react_loop_executes_action_then_finishes() -> None:
    llm = ScriptedLLM(
        [
            "Thought:\nNeed evidence.\n\nAction:\nSearch(ReAct)",
            "Thought:\nI can answer.\n\nFinish:\nReason and Act",
        ]
    )
    loop = ReActLoop(llm, [EchoSearch()], ReActParser(), max_iterations=3)

    state = loop.run(ReActState("What is ReAct?"), [Message(role="user", content="question")])

    assert state.final_answer == "Reason and Act"
    assert state.iteration_count == 2
    assert state.steps[0].observation == "found ReAct"


def test_react_loop_recovers_from_parse_error() -> None:
    llm = ScriptedLLM(
        [
            "I forgot the required format.",
            "Thought:\nI can answer now.\n\nFinish:\nRecovered",
        ]
    )
    loop = ReActLoop(llm, [EchoSearch()], ReActParser(), max_iterations=3)

    state = loop.run(ReActState("Test?"), [Message(role="user", content="question")])

    assert state.final_answer == "Recovered"
    assert state.iteration_count == 2
    assert state.steps[0].raw_output == "I forgot the required format."
    assert "Invalid response format" in (state.steps[0].observation or "")


def test_react_loop_stops_after_repeated_parse_errors() -> None:
    llm = ScriptedLLM(
        [
            "Bad format one.",
            "Bad format two.",
        ]
    )
    loop = ReActLoop(llm, [EchoSearch()], ReActParser(), max_iterations=5, max_parse_retries=1)

    state = loop.run(ReActState("Test?"), [Message(role="user", content="question")])

    assert state.final_answer == "Stopped because the model did not return a valid ReAct-formatted response."
    assert state.iteration_count == 2
