#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bind portable attachment runtime owners to one Local execution run."""

from __future__ import annotations

import os
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import linktools.ai.capability._workspace as workspace_runtime
import linktools.ai.runtime._agent_executor as agent_executor_runtime
import linktools.ai.runtime._attachment_admission as admission_runtime
import linktools.ai.runtime._factory as factory_runtime
import linktools.ai.runtime._local as local_runtime
from pydantic_ai.messages import ModelResponse, ToolCallPart

from ..capability import WorkspaceAccess, attachment_tool_contribution
from ..core import ExecutionStatus, ToolOperationStatus, idempotency_key_digest
from ..errors import AIError, ErrorCode
from ..storage import TransientObjectStore
from ._attachment_active import AttachmentActiveSet
from ._attachment_projection import AttachmentProjectionCapability
from ._attachment_read import AttachmentReadRuntime
from ._object import read_runtime_object
from ._subagent import SubagentAttachmentRuntime
from ._subagent_attachment import SubagentAttachmentPreparer
from .state import ContentRef, ModelExposureEntry, PathOrigin, RuntimeDomain
from .state._attachment_repository import AttachmentRepository
from .state._exposure_repository import ModelExposureRepository


@dataclass(slots=True)
class _AttachmentRunOwner:
    backend: local_runtime.LocalExecutionBackend
    execution_id: str
    tenant_id: str
    access: WorkspaceAccess | None = None
    subagent_bound: bool = False


_run_owner: ContextVar[_AttachmentRunOwner | None] = ContextVar(
    "linktools_ai_attachment_run_owner",
    default=None,
)
_attachment_reader: ContextVar[Any | None] = ContextVar(
    "linktools_ai_attachment_reader",
    default=None,
)
_workspace_access: ContextVar[WorkspaceAccess | None] = ContextVar(
    "linktools_ai_attachment_workspace_access",
    default=None,
)
_installed = False
_original_prepare_start: Any = None
_original_run: Any = None
_original_materialize_agent: Any = None
_original_workspace_capabilities: Any = None
_original_workspace_tool_contributions: Any = None
_original_workspace_for_run: Any = None


def _binding_has_read_attachment(binding: Any) -> bool:
    return any(
        candidate.id == "read_attachment"
        for candidate in binding.definition.selected_tools
    )


async def _prepare_start_with_attachment_owner(
    self: local_runtime.LocalExecutionBackend,
    request: Any,
    execution: Any,
    identity: Any,
):
    if _original_prepare_start is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if execution.attachment_manifest:
        return await _original_prepare_start(self, request, execution, identity)
    binding = self._catalog.binding(execution.binding_digest)
    if not _binding_has_read_attachment(binding):
        return await _original_prepare_start(self, request, execution, identity)

    await self._validate_start(request, execution)
    if execution.status is not local_runtime.ExecutionStatus.PENDING_START:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if execution.binding != binding.snapshot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    now = local_runtime.datetime.now(local_runtime.timezone.utc)
    recovery_input = local_runtime.RecoveryExecutionInput(
        user_prompt=local_runtime._recovery_prompt_payload(request.user_prompt),
        user_prompt_codec=request.user_prompt_codec,
        principal_id=request.principal.principal_id,
        principal_kind=request.principal.kind,
        session_id=execution.session_id,
        memory_scope=execution.memory_scope,
        binding_digest=execution.binding_digest,
        lineage_kind=execution.lineage_kind.value,
        parent_execution_id=execution.parent_execution_id,
        root_execution_id=execution.root_execution_id,
        source_execution_id=execution.source_execution_id,
        base_execution_id=execution.base_execution_id,
        conversation_step_run_id=execution.conversation_step_run_id,
        idempotency=local_runtime.RecoveryIdempotencyInput(
            scope=identity.scope,
            idempotency_key_digest=identity.idempotency_key_digest,
            request_digest=identity.request_digest,
        ),
        mode=execution.mode,
        planning=execution.planning,
        thinking=execution.thinking,
        binding=execution.binding,
        repository_instructions=execution.repository_instructions,
        attachment_manifest=(),
        input_digest=None,
        path_origin=None,
        correlation=execution.correlation,
    )
    candidate = local_runtime.RecoveryCheckpoint(
        execution_id=execution.execution_id,
        tenant_id=execution.tenant_id,
        input=recovery_input,
        step_run_id=None,
        agent_run_sequence=execution.agent_run_sequence,
        state=local_runtime.RecoveryCheckpointState.ADMITTED,
        handoff_phase=local_runtime.RecoveryHandoffPhase.NONE,
        terminal_handoff=None,
        handoff_contract_digest=None,
        pending_operation_id=None,
        revision=0,
        created_at=now,
        updated_at=now,
    )
    expected = (
        await self._expected_session_cursor(execution)
        if execution.session_id is not None
        else None
    )
    started = await self._runtime_commands.commit_start_attempt_checkpoint(
        local_runtime.ExecutionStartClaim(
            execution.execution_id,
            execution.tenant_id,
            execution.revision,
            execution.event_sequence,
            identity.scope,
            identity.idempotency_key_digest,
            identity.request_digest,
            now,
        ),
        recovery_checkpoint=candidate,
        session_id=execution.session_id,
        expected_cursor=expected,
    )
    local_runtime._logger.info(
        "read_attachment recovery owner admitted: execution=%s",
        execution.execution_id,
    )
    return started


async def _run_with_attachment_owner(
    self: local_runtime.LocalExecutionBackend,
    request: Any,
    original: Any,
) -> None:
    if _original_run is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    owner = _AttachmentRunOwner(self, original.execution_id, original.tenant_id)
    token = _run_owner.set(owner)
    try:
        await _original_run(self, request, original)
    finally:
        dispatcher = self._subagent_dispatcher
        if owner.subagent_bound and dispatcher is not None:
            dispatcher.release_attachment_runtime(owner.execution_id)
        if owner.access is not None:
            await owner.access.close()
        _run_owner.reset(token)


async def _materialize_agent_with_attachment_reader(*args: Any, **kwargs: Any):
    if _original_materialize_agent is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    scope = args[0] if args else kwargs.get("scope")
    owner = _run_owner.get()
    if scope is None or owner is None:
        return await _original_materialize_agent(*args, **kwargs)
    if (
        scope.context.execution_id != owner.execution_id
        or scope.context.principal.tenant_id != owner.tenant_id
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    managed = admission_runtime._managed_projection.get()
    if managed is not None and managed.execution_id != owner.execution_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    initial = () if managed is None else managed.activations
    active = AttachmentActiveSet(
        namespace=owner.backend._namespace,
        tenant_id=owner.tenant_id,
        execution_id=owner.execution_id,
        initial=initial,
    )
    await _restore_committed_reads(owner.backend, scope.history, active)

    read_enabled = _binding_has_read_attachment(scope.binding)
    needs_access = read_enabled or bool(scope.subagent_available)
    if needs_access and owner.access is None:
        owner.access = WorkspaceAccess.for_workspace(scope.context.workspace)
    access = owner.access
    reader = (
        None
        if access is None
        else AttachmentReadRuntime(
            owner.backend,
            execution_id=owner.execution_id,
            tenant_id=owner.tenant_id,
            agent_run_sequence=scope.segment_sequence,
            on_committed=active.add_committed_read,
        )
    )

    dispatcher = owner.backend._subagent_dispatcher
    if scope.subagent_available:
        if dispatcher is None or access is None or reader is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        repository = AttachmentRepository(
            owner.backend._execution.executions.state_store,
            namespace=owner.backend._namespace,
            tenant_id=owner.tenant_id,
        )
        preparer = SubagentAttachmentPreparer(repository, scope.context.workspace)

        async def find_child(idempotency_key: str):
            identity = await owner.backend._execution.idempotency.get(
                "execution.subagent",
                idempotency_key_digest(idempotency_key),
                tenant_id=owner.tenant_id,
            )
            if identity is None:
                return None
            return await owner.backend._execution.executions.get(
                identity.resource_id,
                tenant_id=owner.tenant_id,
            )

        dispatcher.bind_attachment_runtime(
            owner.execution_id,
            SubagentAttachmentRuntime(
                preparer,
                lambda path: reader.grant(access, path),
                find_child,
            ),
        )
        owner.subagent_bound = True

    reader_token = _attachment_reader.set(None if reader is None else reader.read)
    access_token = _workspace_access.set(access)
    try:
        result = await _original_materialize_agent(*args, **kwargs)
    finally:
        _workspace_access.reset(access_token)
        _attachment_reader.reset(reader_token)

    if managed is None and not read_enabled and not active.entries():
        return result
    agent, capabilities, runtime_tools, trusted_tools, trusted_mcp = result
    filtered = tuple(
        capability
        for capability in capabilities
        if not isinstance(capability, AttachmentProjectionCapability)
    )
    capability = _runtime_projection_capability(
        owner.backend,
        scope,
        active,
        managed,
    )
    return (
        agent,
        (*filtered, capability),
        runtime_tools,
        trusted_tools,
        trusted_mcp,
    )


async def _restore_committed_reads(
    backend: local_runtime.LocalExecutionBackend,
    history: Any,
    active: AttachmentActiveSet,
) -> None:
    operations = backend._tool_operations
    if operations is None:
        return
    for message in tuple(history):
        if not isinstance(message, ModelResponse):
            continue
        run_id = message.run_id
        if not isinstance(run_id, str) or not run_id:
            continue
        for part in message.parts:
            if not isinstance(part, ToolCallPart) or part.tool_name != "read_attachment":
                continue
            record = await operations.get_by_call(
                run_id,
                part.tool_call_id,
                tenant_id=backend._tenant_id,
            )
            if record is None:
                continue
            if record.tool_name != "read_attachment":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status is ToolOperationStatus.COMPLETED:
                if record.attachment_result is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await active.add_committed_read(
                    record.tool_operation_id,
                    record.attachment_result,
                )
            elif record.attachment_result is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _runtime_projection_capability(
    backend: local_runtime.LocalExecutionBackend,
    scope: Any,
    active: AttachmentActiveSet,
    managed: Any,
) -> AttachmentProjectionCapability:
    tenant_id = scope.context.principal.tenant_id
    execution_id = scope.context.execution_id
    executions = backend._execution.executions
    exposures = ModelExposureRepository(
        backend._recovery.checkpoints.state_store,
        namespace=backend._namespace,
        tenant_id=tenant_id,
    )
    path_origin = (
        managed.path_origin
        if managed is not None
        else PathOrigin(
            1,
            scope.context.workspace.workspace_id,
            "windows" if os.name == "nt" else "posix",
            str(scope.context.workspace.root),
        )
    )

    async def current_execution():
        current = await executions.get(execution_id, tenant_id=tenant_id)
        if current is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if current.status in {ExecutionStatus.CANCELLING, ExecutionStatus.CANCELLED}:
            raise AIError(ErrorCode.EXECUTION_CANCELLED)
        if current.status is not ExecutionStatus.STARTED:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.agent_run_sequence != scope.segment_sequence:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if managed is not None and (
            current.attachment_manifest != managed.manifest
            or current.input_digest != managed.input_digest
            or current.path_origin != managed.path_origin
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return current

    async def check_execution(run_step: int) -> None:
        if isinstance(run_step, bool) or not isinstance(run_step, int) or run_step < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await current_execution()

    async def authorize_entries(
        run_step: int,
        entries: tuple[ModelExposureEntry, ...],
    ) -> None:
        if isinstance(run_step, bool) or not isinstance(run_step, int) or run_step < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await current_execution()
        active.authorize(entries)

    async def commit_exposure(
        run_step: int,
        entries: tuple[ModelExposureEntry, ...],
    ):
        return await exposures.put(
            execution_id=execution_id,
            step_run_id=scope.step_run_id,
            run_step=run_step,
            path_origin=path_origin,
            entries=entries,
        )

    async def read_content(content: ContentRef) -> bytes:
        try:
            domain = RuntimeDomain(content.domain)
        except ValueError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if domain is RuntimeDomain.EXECUTION:
            store = backend._execution_objects
        elif domain is RuntimeDomain.RECOVERY:
            store = backend._recovery_objects
        else:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if isinstance(store, TransientObjectStore):
            if content.owner_scope is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store = store.scoped(f"runtime:{domain.value}:{content.owner_scope}")
        return await read_runtime_object(store, content.object)

    return AttachmentProjectionCapability(
        execution_id=execution_id,
        step_run_id=scope.step_run_id,
        path_origin=path_origin,
        activations=active.entries(),
        activation_provider=active.entries,
        check_execution=check_execution,
        authorize_entries=authorize_entries,
        commit_exposure=commit_exposure,
        read_content=read_content,
    )


async def _workspace_for_run_with_shared_access(self: Any, ctx: Any):
    if _original_workspace_for_run is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    access = getattr(self, "_access", None)
    if access is None:
        return await _original_workspace_for_run(self, ctx)
    current_context = getattr(access, "_run_context", None)
    if current_context is None:
        access._run_context = ctx
    elif current_context is not ctx:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return workspace_runtime._WorkspaceSandboxToolset(
        self._sandbox,
        self._selected_tool_names,
        access=access,
        attachment_reader=self._attachment_reader,
    )


def _workspace_capabilities_with_attachment_reader(
    workspace: Any,
    selected_tool_names: Any,
):
    if _original_workspace_capabilities is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    selected = tuple(selected_tool_names)
    access = _workspace_access.get()
    reader = _attachment_reader.get()
    if access is None:
        if "read_attachment" not in selected:
            return _original_workspace_capabilities(workspace, selected)
        if reader is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return _original_workspace_capabilities(
            workspace,
            selected,
            attachment_reader=reader,
        )
    if not selected:
        return ()
    if "read_attachment" in selected and reader is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    unknown = frozenset(selected).difference(workspace_runtime._WORKSPACE_TOOL_NAMES)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    ordered = tuple(
        name for name in workspace_runtime._WORKSPACE_TOOL_NAMES if name in selected
    )
    sandbox = (
        workspace.sandbox
        if workspace.sandbox is not None
        else workspace_runtime._LocalSandbox(workspace.root)
    )
    toolset = workspace_runtime._WorkspaceSandboxToolset(
        sandbox,
        ordered,
        access=access,
        attachment_reader=reader,
    )
    return (
        workspace_runtime.Toolset(
            toolset,
            id=workspace_runtime._WORKSPACE_SANDBOX_CAPABILITY_ID,
        ),
    )


def _runtime_workspace_tool_contributions(workspace: Any):
    if _original_workspace_tool_contributions is None:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    return (
        *_original_workspace_tool_contributions(workspace),
        attachment_tool_contribution(workspace),
    )


def install_attachment_workspace() -> None:
    """Install portable attachment workspace/runtime binding exactly once."""
    global _installed
    global _original_materialize_agent
    global _original_prepare_start
    global _original_run
    global _original_workspace_capabilities
    global _original_workspace_for_run
    global _original_workspace_tool_contributions
    if _installed:
        return
    _original_prepare_start = local_runtime.LocalExecutionBackend.prepare_start
    _original_run = local_runtime.LocalExecutionBackend._run
    _original_materialize_agent = agent_executor_runtime._materialize_agent
    _original_workspace_capabilities = agent_executor_runtime.workspace_capabilities
    _original_workspace_tool_contributions = factory_runtime.workspace_tool_contributions
    _original_workspace_for_run = workspace_runtime._WorkspaceSandboxToolset.for_run
    local_runtime.LocalExecutionBackend.prepare_start = _prepare_start_with_attachment_owner
    local_runtime.LocalExecutionBackend._run = _run_with_attachment_owner
    agent_executor_runtime._materialize_agent = _materialize_agent_with_attachment_reader
    agent_executor_runtime.workspace_capabilities = (
        _workspace_capabilities_with_attachment_reader
    )
    factory_runtime.workspace_tool_contributions = _runtime_workspace_tool_contributions
    workspace_runtime._WorkspaceSandboxToolset.for_run = _workspace_for_run_with_shared_access
    _installed = True


__all__ = ["install_attachment_workspace"]
