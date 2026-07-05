#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/agent/test_compiler_tools.py — verifies the AgentCompiler
wires the builtin file/terminal FunctionToolset into the compiled pydantic-ai
Agent iff `workdir` is provided, and that an actual read_file tool call driven
through the compiled agent reads a file written under that workdir."""
import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from linktools.ai.agent.compiler import AgentCompiler
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.core.model_runtime import ModelRegistry
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelRouter


def _registry(model_fn) -> ModelRegistry:
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(model_fn))
    return registry


def _spec() -> AgentSpec:
    return AgentSpec(
        id="agent-tools", name="tools-agent",
        model=ModelPolicy(primary="test-model"),
        instructions=PromptSpec(instructions="hi"),
    )


def _builtin_tool_names(compiled) -> "set[str]":
    """Flatten the names of every tool exposed by any user-supplied
    FunctionToolset the compiled agent carries. pydantic-ai always adds an
    internal `_AgentFunctionToolset` for output-schema dispatch -- that one is
    a FunctionToolset subclass but carries no builtin tools, so filter by
    exact class (`type(...) is FunctionToolset`)."""
    names: "set[str]" = set()
    for ts in compiled.pydantic_agent.toolsets:
        if type(ts) is FunctionToolset and getattr(ts, "tools", None):
            names.update(ts.tools.keys())
    return names


def test_workdir_compiled_agent_has_builtin_file_and_terminal_tools(tmp_path):
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(lambda m, i: ModelResponse(parts=[TextPart(content="ok")]))), workdir=tmp_path)
    compiled = asyncio.run(compiler.compile(_spec()))

    expected = {"list_dir", "read_file", "write_file", "batch_files", "apply_patch", "bash"}
    actual = _builtin_tool_names(compiled)
    assert expected <= actual, f"missing builtin tools: {expected - actual}"


def test_no_workdir_compiled_agent_has_no_builtin_toolsets():
    # workdir=None (default) must not register any builtin toolsets --
    # existing compiler tests rely on this regression contract. pydantic-ai
    # always carries its internal `_AgentFunctionToolset` for output dispatch,
    # so filter by exact class to count only user-supplied FunctionToolsets.
    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(lambda m, i: ModelResponse(parts=[TextPart(content="ok")]))))
    compiled = asyncio.run(compiler.compile(_spec()))

    function_toolsets = [ts for ts in compiled.pydantic_agent.toolsets if type(ts) is FunctionToolset]
    assert function_toolsets == [], f"unexpected builtin toolsets: {function_toolsets}"


def test_read_file_tool_call_reads_file_under_workdir(tmp_path):
    # Drive a real read_file tool call through the compiled agent: write a
    # file under workdir, have the FunctionModel emit a read_file ToolCallPart
    # then a final response that validates against the dict output schema, and
    # assert the file content shows up as a tool-return in the run history.
    (tmp_path / "sample.txt").write_text("hello from workdir", encoding="utf-8")

    def model_fn(messages, info: AgentInfo) -> ModelResponse:
        # First turn: emit a read_file tool call. After the tool returns, the
        # next model invocation must produce a final text response so the run
        # terminates cleanly (pydantic-ai would otherwise loop on tool calls
        # until its request limit). The dict output schema requires a JSON
        # object with a `response` key, so the final turn emits that shape.
        if not any(getattr(p, "part_kind", None) == "tool-return" for m in messages for p in m.parts):
            return ModelResponse(parts=[ToolCallPart(tool_name="read_file", args={"path": "sample.txt"})])
        return ModelResponse(parts=[TextPart(content='{"response": {"status": "done"}}')])

    compiler = AgentCompiler(model_router=ModelRouter(registry=_registry(model_fn)), workdir=tmp_path)
    compiled = asyncio.run(compiler.compile(_spec()))

    async def _run():
        return await compiled.pydantic_agent.run("read sample.txt")
    result = asyncio.run(_run())

    tool_returns = [
        p.content
        for m in result.all_messages()
        for p in m.parts
        if getattr(p, "part_kind", None) == "tool-return"
    ]
    assert tool_returns, "expected read_file to have been called"
    assert "hello from workdir" in str(tool_returns[0])
