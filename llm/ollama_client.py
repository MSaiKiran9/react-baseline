import ollama

from config.config import Settings
from llm.base import BaseLLM, LLMError
from react.message import Message


class OllamaLLM(BaseLLM):
    """Ollama-backed LLM implementation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = ollama.Client(host=settings.ollama_url, timeout=settings.ollama_timeout)

    def generate(self, messages: list[Message]) -> str:
        try:
            response = self.client.chat(
                model=self.settings.model_name,
                messages=[message.model_dump() for message in messages],
                think=self.settings.ollama_think,
                options={
                    "temperature": self.settings.temperature,
                    "top_p": self.settings.top_p,
                    "num_predict": self.settings.max_output_tokens,
                },
            )
            if response.get("done_reason") == "length":
                raise LLMError(
                    "The model returned an incomplete response because it reached the output-token limit."
                )
            return response["message"]["content"]
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
