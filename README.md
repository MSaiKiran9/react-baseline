# ReAct Baseline

This repository is a research-oriented baseline implementation of ReAct, the Reason + Act framework introduced in "ReAct: Synergizing Reasoning and Acting in Language Models".

It does not reproduce Google's PaLM model or the exact original experiments. It implements the vanilla ReAct loop with an accessible local model through Ollama so future research can compare new algorithms against the same baseline.

## Architecture

```text
User Question
-> Prompt Builder
-> LLM
-> Thought
-> Action
-> Execute Tool
-> Observation
-> Append Observation
-> Repeat
-> Finish
```

The code is split into small modules:

- `config/`: environment-driven settings
- `llm/`: model abstraction and Ollama implementation
- `react/`: state, parser, prompt builder, agent, and loop
- `tools/`: baseline `Search` and `Lookup` tools over a local corpus
- `evaluation/`: dataset loading, metrics, logging, and evaluation runner
- `prompts/`: ReAct prompt template
- `tests/`: unit tests for parser, prompt construction, tools, and loop behavior

## Install

Install Python 3.11 or newer, then install dependencies with uv:

```bash
uv pip install -r requirements.txt
```

You can also run tests through uv:

```bash
uv run pytest
```

## Ollama

Install Ollama from https://ollama.com/download.

Pull the default local model:

```bash
ollama pull qwen3:8b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Configuration

Copy the example environment file and edit it as needed:

```bash
cp .env.example .env
```

Supported settings:

- `MODEL_PROVIDER`
- `MODEL_NAME`
- `TEMPERATURE`
- `TOP_P`
- `MAX_OUTPUT_TOKENS`
- `MAX_ITERATIONS`
- `MAX_PARSE_RETRIES`
- `DATASET_PATH`
- `OUTPUT_PATH`
- `OLLAMA_URL`
- `OLLAMA_TIMEOUT`
- `OLLAMA_THINK`
- `CORPUS_PATH`

`MODEL_NAME` defaults to `qwen3:8b`, but all runtime code reads it through configuration.

## Run

Ask one question:

```bash
uv run python main.py ask "What is ReAct?"
```

For quick smoke tests with general-knowledge questions, cap the loop to one model call:

```bash
uv run python main.py ask "What's the capital of France?" --max-iterations 1
```

On small GPUs, switching to a smaller Ollama model can be much faster:

```bash
uv run python main.py ask "What's the capital of France?" --model-name qwen2.5:3b --max-iterations 1
```

If Ollama times out on a slower machine, raise the timeout or use a smaller model:

```bash
uv run python main.py ask "What's the capital of France?" --model-name qwen2.5:3b --ollama-timeout 300
```

Use a custom corpus:

```bash
uv run python main.py ask "Who proposed ReAct?" --corpus-path datasets/corpus.json
```

## Evaluate

Datasets are JSON arrays:

```json
[
  {
    "question": "Who proposed ReAct?",
    "answer": "Yao et al."
  }
]
```

Run evaluation:

```bash
uv run python main.py evaluate --dataset-path datasets/sample.json --output-path results/evaluation.json
```

For every sample, the evaluator stores the prediction, expected answer, latency, iterations, and full reasoning trajectory. It also computes accuracy, precision, recall, F1, average latency, and average iterations.

## Reasoning Loop

The model is instructed to emit:

```text
Thought:
...

Action:
Search(...)
```

or:

```text
Thought:
...

Finish:
...
```

The parser extracts the thought and action. The loop executes `Search` or `Lookup`, appends the resulting observation to state, rebuilds the prompt, and repeats until `Finish` or the configured maximum iteration count is reached.

Reasoning traces are logged because they are a core part of the ReAct paper and are important for baseline comparison.

## Add A Tool

Create a class implementing `BaseTool`:

```python
from tools.base import BaseTool, Observation


class MyTool(BaseTool):
    name = "MyTool"
    description = "Describe what the tool does."

    def run(self, tool_input: str) -> Observation:
        return Observation(tool_name=self.name, input=tool_input, output="result")
```

Register it in `build_agent` or in your own experiment harness. The ReAct loop can use any `BaseTool` without changing the loop itself.

## Replace The LLM

Implement `BaseLLM`:

```python
from llm.base import BaseLLM
from react.message import Message


class MyLLM(BaseLLM):
    def generate(self, messages: list[Message]) -> str:
        return "Thought:\n...\n\nFinish:\n..."
```

Pass the implementation into `ReActAgent`. No tool or loop code should call a model provider directly.

## Non-Goals

This baseline intentionally excludes memory, reflection, self-correction, planning modules, hallucination detection, RAG, vector databases, embeddings, function calling, multi-agent systems, and external agent frameworks.
