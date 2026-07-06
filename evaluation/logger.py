import json
from datetime import datetime, timezone
from pathlib import Path

from react.state import ReActState


def write_trace(state: ReActState, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"trace_{timestamp}.json"
    payload = {
        "question": state.question,
        "final_answer": state.final_answer,
        "iterations": state.iteration_count,
        "trajectory": state.trajectory(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
