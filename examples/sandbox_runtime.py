#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sandbox + builtin-tool runtime: a ``Sandbox`` supplies a capability, the
``CapabilityResolver`` exposes it as a tool, and every tool call flows through
the ``GovernedToolInvoker`` (policy + approval + security). There is no path
that calls the sandbox backend directly -- a run reaches files only through a
governed builtin tool resolved from the sandbox.

Executed by ``tests/ai/docs/test_readme_examples.py``. The model is a canned
``FunctionModel`` so the example runs offline."""

from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.model import ModelPolicy, ModelRegistry, ModelResolver
from linktools.ai.runtime import Runtime, build_runtime
from linktools.ai.sandbox.local import LocalSandbox
from linktools.ai.storage import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator


def _canned_model() -> FunctionModel:
    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="ok")])

    return FunctionModel(_fn)


async def run(data_dir: Path, workdir: Path) -> Any:
    """Build a Runtime whose builtin file tools are backed by a ``LocalSandbox``
    over ``workdir``. The agent declares ``file-read``; the runtime resolves it
    from the sandbox and governs every call through the invoker."""
    registry = ModelRegistry()
    registry.register("standard", model=_canned_model())

    storage = FilesystemStorage(root=data_dir)
    runtime = build_runtime(
        storage=storage,
        model_resolver=ModelResolver(registry=registry),
        sandbox=LocalSandbox(runtime_dir=workdir),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    try:
        spec = AgentSpec(
            id="reader",
            name="reader",
            model=ModelPolicy(primary="standard"),
            instructions=PromptSpec(instructions="You read files the user names."),
            tools=(ToolRef(kind="builtin", name="file-read"),),
        )
        result = await runtime.run(spec, "Read README.md and summarize.")
        return result.output
    finally:
        await runtime.aclose()


if __name__ == "__main__":  # pragma: no cover - manual smoke
    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "work"
        work.mkdir()
        print(asyncio.run(run(Path(td), work)))
