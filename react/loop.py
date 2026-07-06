from llm.base import BaseLLM
from react.message import Message
from react.parser import ParsedAction, ParsedFinish, ParseError, ReActParser
from react.state import ReActState, ReActStep
from tools.base import BaseTool


class ReActLoop:
    """Manual implementation of the vanilla ReAct reasoning/action loop."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        parser: ReActParser,
        max_iterations: int,
        max_parse_retries: int = 1,
    ) -> None:
        self.llm = llm
        self.tools = {tool.name.lower(): tool for tool in tools}
        self.parser = parser
        self.max_iterations = max_iterations
        self.max_parse_retries = max_parse_retries

    def run(self, state: ReActState, prompt_messages: list[Message]) -> ReActState:
        messages = list(prompt_messages)
        parse_retries = 0

        while state.iteration_count < self.max_iterations and state.final_answer is None:
            model_output = self.llm.generate(messages)
            try:
                parsed = self.parser.parse(model_output)
            except ParseError as exc:
                parse_retries += 1
                observation_text = (
                    f"Invalid response format: {exc} "
                    "Respond using exactly: Thought: ... then Action: Search(query) or Lookup(query), "
                    "or Thought: ... then Finish: answer."
                )
                state.add_step(
                    ReActStep(
                        thought="Parser could not extract a valid ReAct step.",
                        observation=observation_text,
                        raw_output=model_output,
                    )
                )
                messages.append(Message(role="assistant", content=model_output))
                messages.append(Message(role="user", content=f"Observation:\n{observation_text}"))
                if parse_retries > self.max_parse_retries or not model_output.strip():
                    state.final_answer = "Stopped because the model did not return a valid ReAct-formatted response."
                    break
                continue

            parse_retries = 0
            if isinstance(parsed, ParsedFinish):
                state.final_answer = parsed.answer
                state.add_step(ReActStep(thought=parsed.thought, raw_output=model_output))
                messages.append(Message(role="assistant", content=model_output))
                break

            if isinstance(parsed, ParsedAction):
                observation_text = self._execute(parsed)
                step = ReActStep(
                    thought=parsed.thought,
                    action_name=parsed.action_name,
                    action_input=parsed.action_input,
                    observation=observation_text,
                    raw_output=model_output,
                )
                state.add_step(step)
                messages.append(Message(role="assistant", content=model_output))
                messages.append(Message(role="user", content=f"Observation:\n{observation_text}"))

        if state.final_answer is None:
            state.final_answer = "Maximum iterations reached without Finish."

        return state

    def _execute(self, action: ParsedAction) -> str:
        tool = self.tools.get(action.action_name.lower())
        if tool is None:
            return f"Unknown action '{action.action_name}'. Available actions: {', '.join(sorted(self.tools))}."
        observation = tool.run(action.action_input)
        return observation.output
