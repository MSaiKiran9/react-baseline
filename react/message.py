from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    """Chat message passed to an LLM provider."""

    role: Literal["system", "user", "assistant"]
    content: str
