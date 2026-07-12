#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime.build(pause_on_approval=True) wires a pause-enabled ToolExecutor
with the storage's approval store. Default-False uses the compiler's default
executor (no pause, no approval store)."""

from linktools.ai.model.router import ModelRouter
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage
from linktools.ai.model.registry import ModelRegistry
from pydantic_ai.models.function import FunctionModel


def _registry() -> ModelRegistry:
    r = ModelRegistry()
    r.register("m", model=FunctionModel(lambda m, i: None))
    return r


def test_build_with_pause_wires_pause_enabled_executor(tmp_path):
    runtime = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
        pause_on_approval=True,
    )
    # The compiler's default tool executor has pause_on_approval=True +
    # approval_store wired.
    executor = runtime.compiler._tool_executor
    assert executor._pause_on_approval is True
    assert executor._approval_store is not None


def test_build_default_has_pause_disabled(tmp_path):
    runtime = Runtime.build(
        storage=FileStorage(root=tmp_path),
        model_router=ModelRouter(registry=_registry()),
    )
    executor = runtime.compiler._tool_executor
    assert executor._pause_on_approval is False
