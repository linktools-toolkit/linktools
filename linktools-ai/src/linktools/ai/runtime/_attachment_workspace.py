#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bind the standard read_attachment tool to one Local execution run."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import linktools.ai.runtime._agent_executor as agent_executor_runtime
import linktools.ai.runtime._local as local_runtime

from ..errors import AIError, ErrorCode
from ._attachment_read import AttachmentReadRuntime


@dataclass(frozen=True, slots=True)
class _AttachmentRunOwner:
    backend: local_runtime.LocalExecutionBackend
    execution_id: str
    tenant_id: str


_run_owner: ContextVar[_AttachmentRunOwner | None] = ContextVar(
    "linktools_ai_attachment_run_owner",
    default=None,
)
_attachment_reader: ContextVar[Any | None] = ContextVar(
    "linktools_ai_attachment_reader",
    default=None,
)
_installed = False
_original_run = local_runtime.LocalExecutionBackend._run
_original_materialize_agent = agent_executor_runtime._materialize_agent
_original_workspace_capabilities = agent_executor_runtime.workspace_capabilities


async def _run_with_attachment_owner(
    self: local_runtime.LocalExecutionBackend,
    request: Any,
    original: Any,
) -> None:
    owner = _AttachmentRunOwner(self, original.execution_id, original.tenant_id)
    token = _run_owner.set(owner)
    try:
        await _original_run(self, request, original)
    finally:
        _run_owner.reset(token)


async def _materialize_agent_with_attachment_reader(*args: Any, **kwargs: Any):
    scope = args[0] if args else kwargs.get("scope")
    owner = _run_owner.get()
    if scope is None or owner is None:
        return await _original_materialize_agent(*args, **kwargs)
    if (
        scope.context.execution_id != owner.execution_id
        or scope.context.principal.tenant_id != owner.tenant_id
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reader = AttachmentReadRuntime(
        owner.backend,
        execution_id=owner.execution_id,
        tenant_id=owner.tenant_id,
        agent_run_sequence=scope.segment_sequence,
    )
    token = _attachment_reader.set(reader.read)
    try:
        return await _original_materialize_agent(*args, **kwargs)
    finally:
        _attachment_reader.reset(token)


def _workspace_capabilities_with_attachment_reader(
    workspace: Any,
    selected_tool_names: Any,
):
    reader = _attachment_reader.get()
    selected = tuple(selected_tool_names)
    if "read_attachment" not in selected:
        return _original_workspace_capabilities(workspace, selected)
    if reader is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return _original_workspace_capabilities(
        workspace,
        selected,
        attachment_reader=reader,
    )


def install_attachment_workspace() -> None:
    """Install the run-scoped read_attachment workspace binding exactly once."""
    global _installed
    if _installed:
        return
    local_runtime.LocalExecutionBackend._run = _run_with_attachment_owner
    agent_executor_runtime._materialize_agent = _materialize_agent_with_attachment_reader
    agent_executor_runtime.workspace_capabilities = (
        _workspace_capabilities_with_attachment_reader
    )
    _installed = True


__all__ = ["install_attachment_workspace"]
