from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    """Environment-driven runtime configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_provider: str = Field(default="ollama", alias="MODEL_PROVIDER")
    model_name: str = Field(default="qwen3:8b", alias="MODEL_NAME")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")
    top_p: float = Field(default=1.0, alias="TOP_P")
    max_output_tokens: int = Field(default=1024, alias="MAX_OUTPUT_TOKENS")
    max_iterations: int = Field(default=8, alias="MAX_ITERATIONS")
    max_parse_retries: int = Field(default=1, alias="MAX_PARSE_RETRIES")
    dataset_path: Path = Field(default=Path("datasets/sample.json"), alias="DATASET_PATH")
    output_path: Path = Field(default=Path("results/evaluation.json"), alias="OUTPUT_PATH")
    ollama_url: str = Field(default="http://localhost:11434", alias="OLLAMA_URL")
    ollama_timeout: float = Field(default=120.0, alias="OLLAMA_TIMEOUT")
    ollama_think: bool = Field(default=False, alias="OLLAMA_THINK")
    corpus_path: Path = Field(default=Path("datasets/corpus.json"), alias="CORPUS_PATH")


def get_settings() -> Settings:
    return Settings()
