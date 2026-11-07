"""Run a deterministic local DataCortex-AI demo.

This script executes the real planner pipeline against a repository fixture and
writes demo artifacts to `demo_outputs/`. Remote LLM calls are disabled by
default so the demo is repeatable on a laptop without provider availability.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.planner import PlannerAgent
from src.core.checkpoint import CheckpointManager
from src.core.config import Config


DEFAULT_DATASET = ROOT / "tests" / "fixtures" / "iris.csv"
DEFAULT_OUTPUT_DIR = ROOT / "demo_outputs"


async def run_demo(dataset: Path, target: str, query: str, output_dir: Path) -> dict:
    os.environ.setdefault("DATACORTEX_ENABLE_REMOTE_LLM", "0")
    Config.reset()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []

    async def on_event(event: dict) -> None:
        events.append(event)
        event_type = event.get("type", "event")
        phase = event.get("phase")
        suffix = f" [{phase}]" if phase else ""
        print(f"{event_type}{suffix}")

    planner = PlannerAgent(
        config=Config.get(),
        checkpoint_manager=CheckpointManager(persist_dir=str(checkpoint_dir)),
        on_event=on_event,
    )

    result = await planner.run_pipeline(
        file_content=dataset.read_bytes(),
        file_name=dataset.name,
        user_query=query,
        target_column=target,
        session_id="demo-session",
    )

    report_path = output_dir / "demo_report.md"
    events_path = output_dir / "demo_events.json"
    summary_path = output_dir / "demo_summary.json"

    report_path.write_text(result.get("report", ""), encoding="utf-8")
    events_path.write_text(json.dumps(events, indent=2, default=str), encoding="utf-8")
    summary_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print()
    print(f"Demo status: {result.get('status')}")
    print(f"Report: {report_path}")
    print(f"Events: {events_path}")
    print(f"Summary: {summary_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local DataCortex-AI demo pipeline.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target", default="Species")
    parser.add_argument(
        "--query",
        default="Produce a concise demo analysis with model comparison and key findings.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    result = asyncio.run(run_demo(args.dataset, args.target, args.query, args.output_dir))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
