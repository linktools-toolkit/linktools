#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal linktools-ai Runtime using only public composition APIs."""

import os
from pathlib import Path

from linktools.ai.capability import FileSystem
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.model import create_model
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.workspace import Workspace


async def run(project: Path) -> object:
    """Open one workspace Runtime and return an inline Agent result."""
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    binding = create_model(
        api="openai",
        model=model,
        input_types=(),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
    workspace = Workspace.load(project)
    application = RuntimeFeatures[None]("application")
    application.model("default", binding)
    application.capability(FileSystem(workspace))
    application.agent(
        "writer",
        revision=2,
        model="default",
        allow_tools=("list_files", "search_files", "read_file", "write_file"),
        allow_skills=(),
        system_prompt="You are a careful writer.",
    )
    async with Runtime.open(
        "default", storage=PersistenceStorage.in_memory(),
        features=application,
    ) as runtime:
        result = await runtime.agents.get("writer").run("Say hello.")
        return result.result.output


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        print(asyncio.run(run(Path(directory))))
