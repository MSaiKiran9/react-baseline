import re
from dataclasses import dataclass


class ParseError(ValueError):
    """Raised when an LLM response does not follow the ReAct format."""


@dataclass(frozen=True)
class ParsedAction:
    thought: str
    action_name: str
    action_input: str


@dataclass(frozen=True)
class ParsedFinish:
    thought: str
    answer: str


ParsedResponse = ParsedAction | ParsedFinish


class ReActParser:
    """Parse vanilla ReAct text responses into structured actions."""

    thought_pattern = re.compile(
        r"Thought:\s*(?P<thought>.*?)(?=\n\s*(Action|Finish):)",
        re.IGNORECASE | re.DOTALL,
    )
    action_pattern = re.compile(
        r"Action:\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<input>.*)\)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    finish_pattern = re.compile(r"Finish:\s*(?P<answer>.*)\s*$", re.IGNORECASE | re.DOTALL)
    think_pattern = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

    def parse(self, text: str) -> ParsedResponse:
        text = self._normalize(text)
        thought_match = self.thought_pattern.search(text)
        if not thought_match:
            raise ParseError("Response is missing a Thought section.")

        thought = thought_match.group("thought").strip()
        remainder = text[thought_match.end() :].strip()

        finish_match = self.finish_pattern.search(remainder)
        if finish_match and remainder.lower().startswith("finish:"):
            return ParsedFinish(thought=thought, answer=finish_match.group("answer").strip())

        action_match = self.action_pattern.search(remainder)
        if action_match and remainder.lower().startswith("action:"):
            return ParsedAction(
                thought=thought,
                action_name=action_match.group("name").strip(),
                action_input=self._clean_input(action_match.group("input")),
            )

        raise ParseError("Response must contain either Action: Tool(input) or Finish: answer.")

    @staticmethod
    def _clean_input(value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            return cleaned[1:-1].strip()
        return cleaned

    def _normalize(self, text: str) -> str:
        cleaned = self.think_pattern.sub("", text).strip()
        thought_index = cleaned.lower().find("thought:")
        if thought_index > 0:
            cleaned = cleaned[thought_index:]
        return cleaned
