import json
from datetime import datetime
from pathlib import Path


def create_run_dir(root: Path | None = None) -> Path:
    if root is None:
        root = Path(__file__).resolve().parent.parent / "results"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config(run_dir: Path, args, model_constants: dict) -> None:
    payload = {
        "cli": vars(args),
        "model": model_constants,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def write_model_summary(run_dir: Path, model) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    (run_dir / "summary.txt").write_text("\n".join(lines))


def write_final_metrics(run_dir: Path, history) -> None:
    final = {k: float(v[-1]) for k, v in history.history.items()}
    final["epochs_trained"] = len(next(iter(history.history.values()), []))
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=2))
