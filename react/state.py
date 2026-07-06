from dataclasses import dataclass, field


@dataclass
class ReActStep:
    thought: str
    action_name: str | None = None
    action_input: str | None = None
    observation: str | None = None
    raw_output: str | None = None


@dataclass
class ReActState:
    question: str
    history: list[str] = field(default_factory=list)
    steps: list[ReActStep] = field(default_factory=list)
    iteration_count: int = 0
    final_answer: str | None = None

    def add_step(self, step: ReActStep) -> None:
        self.steps.append(step)
        self.iteration_count += 1

    def trajectory(self) -> list[dict[str, str | None]]:
        return [
            {
                "thought": step.thought,
                "action_name": step.action_name,
                "action_input": step.action_input,
                "observation": step.observation,
                "raw_output": step.raw_output,
            }
            for step in self.steps
        ]
