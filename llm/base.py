from abc import ABC, abstractmethod

from react.message import Message


class LLMError(RuntimeError):
    """Raised when an LLM provider request fails."""


class BaseLLM(ABC):
    """Provider-agnostic language model interface."""

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Generate one model response from a message history."""
