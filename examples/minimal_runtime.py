#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal linktools-ai Runtime using only public composition APIs."""

import os
from pathlib import Path

from linktools.ai.capability import CapabilityGroup
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime
from linktools.ai.workspace import Workspace


async def run(project: Path) -> object:
    """Open one workspace Runtime and return an inline Agent result."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    models = ModelRegistry.openai(
        model=model,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    workspace = Workspace.load(project)
    application = CapabilityGroup[None]("application")
    application.agent(
        "writer",
        model="default",
        system_prompt="You are a careful writer.",
    )
    async with Runtime.open(
        workspace,
        models=models,
        capabilities=(application,),
    ) as runtime:
        result = await runtime.agent("writer").run("Say hello.")
        return result.output


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        print(asyncio.run(run(Path(directory))))
