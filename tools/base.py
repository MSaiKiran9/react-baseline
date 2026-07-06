from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    tool_name: str
    input: str
    output: str


class BaseTool(ABC):
    """Base interface for all ReAct tools."""

    name: str
    description: str

    @abstractmethod
    def run(self, tool_input: str) -> Observation:
        """Execute the tool and return an observation."""
