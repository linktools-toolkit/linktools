#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace composition root for the public Runtime."""

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager

from linktools.core import environ

from ..adapter import RuntimeMemoryStore, StepExecutionHistoryReader
from ..agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AgentCompiler,
    AgentDefinition,
    AgentExecutor,
    AssistantTextOutput,
    OutputTypeRegistry,
)
from ..asset import (
    AssetRef,
    AssetRepository,
    AssetStore,
    AssetTypeRegistry,
    DirectoryAssetBackend,
    InMemoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ..capability import CapabilityGrant, CapabilityProvider, SkillCapabilityProvider
from ..core import HmacCursorSigner, TenantAuthorizationPolicy, canonical_sha256
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
from ..runtime import (
    DefaultApprovalService,
    DefaultArtifactService,
    DefaultEvaluationService,
    DefaultEventService,
    DefaultExecutionService,
    DefaultSessionService,
    DefaultTaskService,
    LocalExecutionBackend,
    Runtime,
    RuntimeTaskNodeRunner,
)
from ..runtime.state import (
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    RecoveryState,
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeState,
    RuntimeStatePlan,
    RuntimeStateRoute,
)
from ..spec import AgentCapabilityRef, AgentSpec, PromptSpec, builtin_asset_bindings
from ..storage import StorageLayer, StorageOverlay
from ..task import LocalTaskGraphLauncher
from ._root import Workspace
from ._subagent import SubagentDispatcher

_logger = environ.get_logger("ai.workspace.runtime")


async def _build_default_assets(workspace: Workspace) -> AssetRepository:
    source = DirectoryAssetBackend(str(workspace.storage_root), writable=False, path_adapter=PrefixAssetPathAdapter({"skill": "skills"}))
    writable = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(source, writer=writable, layers=(StorageLayer("workspace-defaults", writable),)))
    await store.initialize()
    repository = _build_asset_repository(store)
    await _put_default_asset(repository, AssetRef("prompt", "default"), PromptSpec("default", 1, "", ()))
    return repository


def _build_asset_repository(store: AssetStore) -> AssetRepository:
    registry = AssetTypeRegistry()
    for binding in builtin_asset_bindings():
        registry.register(binding)
    return AssetRepository(store, registry.freeze())


async def _list_asset_ids(assets: AssetRepository, kind: str) -> tuple[str, ...]:
    cursor: str | None = None
    seen_cursors: set[str] = set()
    values: set[str] = set()
    while True:
        page = await assets.list(kind=kind, cursor=cursor, limit=200)
        values.update(item.ref.id for item in page.items)
        if page.next_cursor is None:
            return tuple(sorted(values))
        if page.next_cursor in seen_cursors:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor


async def _list_asset_refs(assets: AssetRepository, kind: str, *, required: bool) -> tuple[AgentCapabilityRef, ...]:
    return tuple(AgentCapabilityRef("skill" if kind == "skill" else "mcp", item, required=required) for item in await _list_asset_ids(assets, kind))


async def _put_default_asset(repository: AssetRepository, ref: AssetRef, value: object) -> None:
    try:
        await repository.resolve(ref)
    except AIError as error:
        if error.code is not ErrorCode.STORAGE_NOT_FOUND:
            raise
        await repository.put(ref, value)


def _build_default_models(workspace: Workspace) -> ModelRegistry:
    configured = workspace.config.get("model")
    model = configured.strip() if isinstance(configured, str) and configured.strip() else os.getenv("OPENAI_MODEL", "").strip()
    if not model:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or None
    return ModelRegistry.openai(model=model, base_url=base_url, api_key=api_key)


def _default_runtime_state(workspace: Workspace) -> RuntimeState:
    plan = RuntimeStatePlan(conversation=RuntimeStateRoute.filesystem(workspace.storage_root / "runtime" / "conversation"))
    return RuntimeState.from_plan(plan)


async def _compile_recovery_definitions(compiler: AgentCompiler, definitions: dict[str, AgentDefinition], recovery: RecoveryState, *, tenant_id: str) -> None:
    checkpoints = await recovery.checkpoints.list(tenant_id=tenant_id)
    for checkpoint in checkpoints:
        if checkpoint.state is RecoveryCheckpointState.COMPLETED or checkpoint.handoff_phase is not RecoveryHandoffPhase.NONE:
            continue
        definition = await (
            compiler.compile_subagent(
                agent_id=checkpoint.input.agent_id,
                prompt_id=checkpoint.input.prompt_id,
            )
            if checkpoint.input.parent_execution_id is not None
            else compiler.compile(
                agent_id=checkpoint.input.agent_id,
                prompt_id=checkpoint.input.prompt_id,
            )
        )
        if definition.digest != checkpoint.input.binding_digest:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE, safe_details={"execution_id": checkpoint.execution_id})
        definitions[checkpoint.input.binding_digest] = definition


@asynccontextmanager
async def open_workspace_runtime(
    workspace: Workspace,
    *,
    assets: "AssetRepository | None" = None,
    runtime: "RuntimeState | None" = None,
    models: "ModelRegistry | None" = None,
    capabilities: "Sequence[CapabilityGrant]" = (),
    capability_providers: "Sequence[CapabilityProvider]" = (),
) -> AsyncIterator[Runtime]:
    if not isinstance(workspace, Workspace):
        raise TypeError("workspace is required")
    selected_assets = assets
    workspace_owned_assets = selected_assets is None
    if selected_assets is None:
        selected_assets = await _build_default_assets(workspace)
    elif not selected_assets.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "AssetRepository must be ready")
    providers = (SkillCapabilityProvider(), *tuple(capability_providers))
    _validate_providers(providers)
    if workspace_owned_assets:
        skill_refs = await _list_asset_refs(selected_assets, "skill", required=False)
        mcp_refs = await _list_asset_refs(selected_assets, "mcp", required=False) if any(provider.provider == "mcp" for provider in capability_providers) else ()
        await _put_default_asset(selected_assets, AssetRef("agent", "default"), AgentSpec("default", 1, "default", (*skill_refs, *mcp_refs), ASSISTANT_TEXT_OUTPUT_SCHEMA_ID, ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION))
    selected_models = models or _build_default_models(workspace)
    model_resolver = selected_models.snapshot()
    grants = (*_build_workspace_capability_grants(workspace), *tuple(capabilities))
    _validate_grants(grants)
    profile = canonical_sha256({
        "version": 4,
        "workspace_id": workspace.workspace_id,
        "platform_capabilities": {"filesystem": {"version": 1}, "shell": {"version": 1}, "memory": {"version": 1}, "subagent": {"version": 1, "tool": "delegate_task", "nested": False}},
        "direct_grants": [{"id": grant.id, "fingerprint": grant.fingerprint, "inherit_to_subagents": grant.inherit_to_subagents} for grant in sorted(grants, key=lambda value: value.id)],
    })
    output_types = OutputTypeRegistry()
    output_types.register(ASSISTANT_TEXT_OUTPUT_SCHEMA_ID, ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION, AssistantTextOutput)
    output_types.freeze()
    compiler = AgentCompiler(selected_assets, model_resolver=model_resolver, output_types=output_types, capability_providers=providers, capability_grants=grants, execution_profile_fingerprint=profile)
    selected_runtime = runtime or _default_runtime_state(workspace)
    await selected_runtime.initialize(namespace=workspace.workspace_id, tenant_id=workspace.workspace_id)
    setup_cleanup: list[Callable[[], Awaitable[None]]] = [selected_runtime.close]
    try:
        definitions: dict[str, AgentDefinition] = {}
        history = StepExecutionHistoryReader(
            namespace=workspace.workspace_id,
            executions=selected_runtime.execution.executions,
            store=selected_runtime.steps.read_store(RuntimeDomain.EXECUTION),
            cursor_signer=HmacCursorSigner("execution-history", _grant_key(workspace)),
        )
        authorization = TenantAuthorizationPolicy(workspace.workspace_id)
        execution = DefaultExecutionService(
            selected_runtime.execution,
            selected_runtime._object_store(RuntimeDomain.EXECUTION),
            authorization,
            sessions=selected_runtime.conversation.sessions,
            history_reader=history,
            release_terminal=selected_runtime.retention.release_execution_handoff,
        )
        dispatcher = SubagentDispatcher(compiler, definitions, execution)
        executor = AgentExecutor(execution_root=workspace.root)

        def memory_store_factory(memory_tenant: str, execution_id: str, memory_scope: str) -> RuntimeMemoryStore:
            memory_route = selected_runtime.plan.route(RuntimeDomain.MEMORY)
            memory_objects = (
                selected_runtime._working_object_store(RuntimeDomain.MEMORY, owner_scope=f"execution:{execution_id}")
                if memory_route.retention is RuntimeRetentionMode.TRANSIENT
                else selected_runtime._object_store(RuntimeDomain.MEMORY)
            )
            return RuntimeMemoryStore(
                selected_runtime.memory,
                object_store=memory_objects,
                namespace=workspace.workspace_id,
                tenant_id=memory_tenant,
                execution_id=execution_id,
                memory_scope=memory_scope,
                transient=memory_route.retention is RuntimeRetentionMode.TRANSIENT,
            )

        backend = LocalExecutionBackend(
            selected_runtime.conversation,
            selected_runtime.execution,
            selected_runtime.recovery,
            selected_runtime._object_store(RuntimeDomain.EXECUTION),
            selected_runtime._object_store(RuntimeDomain.RECOVERY),
            selected_runtime._metrics,
            workspace.workspace_id,
            selected_runtime.steps,
            executor,
            definitions,
            tenant_id=workspace.workspace_id,
            execution_root=workspace.root,
            step_reads={domain: selected_runtime.steps.read_store(domain) for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY)},
            step_lifecycle=selected_runtime.steps,
            memory_store_factory=memory_store_factory,
            recovery_enabled=RuntimeDomain.RECOVERY in selected_runtime.plan.durable_domains,
            conversation_durable=selected_runtime.plan.route(RuntimeDomain.CONVERSATION).retention is RuntimeRetentionMode.DURABLE,
            handoff_contract_digest=selected_runtime.handoff_contract_digest,
            subagent_dispatcher=dispatcher,
        )
        setup_cleanup.append(backend.close)
        execution.bind_backend(backend)
        execution.bind_subagent_cancellation(dispatcher)
        session = DefaultSessionService(
            selected_runtime.conversation,
            selected_runtime.execution.executions,
            authorization,
            execution,
            HmacCursorSigner("session", _grant_key(workspace)),
            release_terminal=selected_runtime.retention.release_session,
        )
        task_runner = RuntimeTaskNodeRunner(execution, definitions)
        task_launcher = LocalTaskGraphLauncher(selected_runtime.task.tasks, task_runner, owner=f"workspace:{workspace.workspace_id}")
        setup_cleanup.append(task_launcher.shutdown)
        task = DefaultTaskService(selected_runtime.task, authorization, task_launcher, release_terminal=selected_runtime.retention.release_task_graph)
        evaluation = DefaultEvaluationService(
            selected_runtime.evaluation,
            selected_runtime.execution.executions,
            authorization,
            execution,
            release_terminal=selected_runtime.retention.release_evaluation,
            acquire_execution_hold=execution._acquire_dependency_hold,
            release_execution_hold=execution._release_dependency_hold,
            request_execution_handoff=execution._request_terminal_handoff,
        )
        approval = DefaultApprovalService(selected_runtime.recovery.approvals, authorization)
        event = DefaultEventService(
            selected_runtime.execution.executions,
            selected_runtime.execution.events,
            authorization,
        )
        artifact = DefaultArtifactService(selected_runtime.artifact, authorization, grant_key=_grant_key(workspace), cursor_signer=HmacCursorSigner("artifact", _grant_key(workspace)))
        coordinator = _RuntimeCloseCoordinator((task_launcher.shutdown, backend.close, selected_runtime.close))
        runtime_value = Runtime(compiler, execution, session, task, evaluation, approval, event, artifact, definitions=definitions, close_callback=coordinator.close)
        if RuntimeDomain.RECOVERY in selected_runtime.plan.durable_domains:
            await _compile_recovery_definitions(compiler, definitions, selected_runtime.recovery, tenant_id=workspace.workspace_id)
        if RuntimeDomain.RECOVERY in selected_runtime.plan.durable_domains:
            await backend.reconcile()
        _logger.info("workspace runtime opened: workspace=%s durable_domains=%s", workspace.workspace_id, sorted(domain.value for domain in selected_runtime.plan.durable_domains))
    except BaseException as primary:
        await _cleanup_setup(setup_cleanup, primary)
        raise
    try:
        yield runtime_value
    except BaseException as body_error:
        try:
            await runtime_value.close()
        except BaseException as close_error:
            raise close_error from body_error
        raise
    else:
        await runtime_value.close()


class _RuntimeCloseCoordinator:
    def __init__(self, actions: tuple[Callable[[], Awaitable[None]], ...]) -> None:
        self._actions = actions
        self._cursor = 0
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._lock:
            if self._cursor >= len(self._actions):
                return
            task = self._task
            if task is None or task.done():
                task = asyncio.create_task(self._run(), name="linktools-workspace-runtime-close")
                self._task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(task)
            except BaseException as error:
                raise error from cancellation
            raise

    async def _run(self) -> None:
        while self._cursor < len(self._actions):
            await self._actions[self._cursor]()
            self._cursor += 1


async def _cleanup_setup(actions: list[Callable[[], Awaitable[None]]], primary: BaseException) -> None:
    del primary
    for action in reversed(actions):
        task = asyncio.create_task(action(), name="linktools-workspace-setup-cleanup")
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except BaseException:
                _logger.error("workspace setup cleanup failed after cancellation", exc_info=True)
        except BaseException:
            _logger.error("workspace setup cleanup failed", exc_info=environ.debug)


def _validate_providers(providers: Sequence[CapabilityProvider]) -> None:
    names = [provider.provider for provider in providers]
    if len(names) != len(set(names)) or "application" in names:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _validate_grants(grants: Sequence[CapabilityGrant]) -> None:
    identities = [(grant.provider, grant.id) for grant in grants]
    if len(identities) != len(set(identities)):
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)


def _build_workspace_capability_grants(workspace: Workspace) -> tuple[CapabilityGrant, ...]:
    from ._tools import build_workspace_capability_grants

    return build_workspace_capability_grants(workspace.root)


def _grant_key(workspace: Workspace) -> bytes:
    return hashlib.sha256(f"workspace:{workspace.workspace_id}".encode("utf-8")).digest()


__all__ = ["open_workspace_runtime"]
