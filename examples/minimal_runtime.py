#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal linktools-ai runtime: register a model, build a Runtime over local
directory storage, run a no-tool agent, shut it down. Uses public imports
only.

The model is a pydantic-ai ``FunctionModel`` (a canned response) so the
example runs offline -- swap it for a real model (via ``RuntimeModelConfig``)
in production."""

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model import ModelPolicy, ModelRegistry, ModelResolver
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime


def _canned_model() -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="hello from linktools-ai")])

    return FunctionModel(_fn)


async def run(data_dir: Path) -> Any:
    """Build a Runtime over LocalDirectoryStorage, run one no-tool agent,
    return its output, and close the Runtime. ``data_dir`` is the storage
    root."""
    registry = ModelRegistry()
    registry.register("standard", model=_canned_model())

    storage = LocalDirectoryStorage(root=data_dir)
    await storage.initialize_storage()
    runtime = build_runtime(storage=storage, model_resolver=ModelResolver(registry=registry))
    async with runtime:
        spec = AgentSpec(
            id="writer",
            name="writer",
            model=ModelPolicy(primary="standard"),
            instructions=PromptSpec(instructions="You are a careful writer."),
        )
        return await runtime.run(spec, "Say hello.")


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        print(asyncio.run(run(Path(td))))
