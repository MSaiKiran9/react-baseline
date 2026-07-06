from pathlib import Path
from datetime import date

from config.config import Settings
from llm.base import BaseLLM
from react.loop import ReActLoop
from react.message import Message
from react.parser import ReActParser
from react.state import ReActState
from tools.base import BaseTool


class PromptBuilder:
    """Build prompts for the baseline ReAct agent."""

    def __init__(self, template_path: Path) -> None:
        self.template_path = template_path

    def build(self, question: str, tools: list[BaseTool]) -> list[Message]:
        tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        tool_names = ", ".join(tool.name for tool in tools)
        template = self.template_path.read_text(encoding="utf-8")
        content = template.format(
            question=question,
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            current_date=date.today().isoformat(),
        )
        return [Message(role="user", content=content)]


class ReActAgent:
    """High-level agent API around the ReAct loop."""

    def __init__(
        self,
        llm: BaseLLM,
        tools: list[BaseTool],
        settings: Settings,
        prompt_builder: PromptBuilder | None = None,
        parser: ReActParser | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.settings = settings
        self.prompt_builder = prompt_builder or PromptBuilder(Path("prompts/react_prompt.txt"))
        self.parser = parser or ReActParser()

    def run(self, question: str) -> ReActState:
        state = ReActState(question=question)
        messages = self.prompt_builder.build(question, self.tools)
        loop = ReActLoop(
            llm=self.llm,
            tools=self.tools,
            parser=self.parser,
            max_iterations=self.settings.max_iterations,
            max_parse_retries=self.settings.max_parse_retries,
        )
        return loop.run(state=state, prompt_messages=messages)
