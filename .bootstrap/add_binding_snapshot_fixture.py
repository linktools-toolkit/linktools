#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = (
    ROOT
    / "linktools-ai/scripts/build/matrix/runtime_agent_binding_snapshot_v1.json"
)


def main() -> None:
    if not FIXTURE.is_file():
        raise RuntimeError("agent binding snapshot V1 fixture is missing")
    print("agent binding snapshot V1 fixture is present")


if __name__ == "__main__":
    main()
