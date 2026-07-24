#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal linktools-ai runtime: register a model, build a Runtime, run a
no-tool agent, shut it down. Uses public imports only.

This module is executed by ``tests/ai/docs/test_readme_examples.py`` so the
README example cannot silently drift. The model is a pydantic-ai
``FunctionModel`` (a canned response) so the example runs offline -- swap it
for a real ``OpenAIChatModel`` (via ``RuntimeModelConfig``) in production."""

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model import ModelPolicy, ModelRegistry, ModelResolver
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.storage import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def _canned_model() -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hello from linktools-ai")])

    return FunctionModel(_fn)


async def run(data_dir: Path) -> Any:
    """Build a Runtime over a FilesystemStorage, run one no-tool agent, return
    its output, and close the Runtime. ``data_dir`` is the storage root."""
    registry = ModelRegistry()
    registry.register("standard", model=_canned_model())

    storage = FilesystemStorage(root=data_dir)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=registry),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    try:
        spec = AgentSpec(
            id="writer",
            name="writer",
            model=ModelPolicy(primary="standard"),
            instructions=PromptSpec(instructions="You are a careful writer."),
        )
        result = await runtime.run(spec, "Say hello.")
        return result.output
    finally:
        await runtime.aclose()


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        print(asyncio.run(run(Path(td))))
