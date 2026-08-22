#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal linktools-ai Runtime using only public composition APIs."""

import os
from pathlib import Path

from linktools.ai.model import ModelRegistry
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import Workspace, open_workspace_runtime


async def run(project: Path) -> object:
    """Open one workspace Runtime and return an inline Agent result."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    models = ModelRegistry.openai(
        model=model,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    workspace = Workspace.load(project)
    spec = AgentSpec(
        "writer",
        1,
        "default",
        system_prompt="You are a careful writer.",
    )
    async with open_workspace_runtime(workspace, models=models) as runtime:
        result = await runtime.agent(spec).run("Say hello.")
        return result.output


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        print(asyncio.run(run(Path(directory))))
