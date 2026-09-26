#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for execution-owned Asset version bindings."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.agent import (
    AgentBinding,
    AgentBindingContract,
    AgentCatalog,
    AgentCompiler,
    CapabilityPin,
)
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AssetSkillResourceSource,
    CapabilityContribution,
    SkillDefinition,
    SkillResourceVersion,
    SkillSourceRef,
)
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._binding_resolver import _RuntimeBindingResolver
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._runtime_service import Runtime
from linktools.ai.runtime._task_capability_capture import TaskCapabilityCaptureStore
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionRequest
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage, SnapshotLimits
from linktools.ai.runtime.state._contracts import ExecutionRecord, StoredUserInput
from linktools.ai.spec import (
    AgentSpec,
    MCPServerSpec,
    MCPServerSpecCodec,
    SkillSpec,
    mcp_server_selector,
)
from linktools.ai.storage import InMemoryObjectStore, StorageOverlay, StoredPayload
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskNode,
)
from linktools.ai.workspace import BubblewrapSandbox


@dataclass(frozen=True)
class _BindingFixture:
    compiler: AgentCompiler
    catalog: AgentCatalog
    resolver: _RuntimeBindingResolver
    assets: AssetStore
    binding: AgentBinding


class _RecordingExecution:
    def __init__(self) -> None:
        self.binding_digest: str | None = None
        self.binding_contract: AgentBindingContract | None = None

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: str | None = None,
        binding_contract: AgentBindingContract | None = None,
    ) -> ExecutionHandle:
        del request, dependency_hold_id
        self.binding_digest = binding_digest
        self.binding_contract = binding_contract
        return ExecutionHandle("execution")


async def _fixture() -> _BindingFixture:
    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    await assets.initialize()
    await assets.put(AssetKey("skill", "child-skill/guide.txt"), b"original")
    child_asset = (
        await assets.resolve_versions((AssetKey("skill", "child-skill/guide.txt"),))
    )[0]
    child_ref = SkillSourceRef(
        "application",
        "child-skill",
        (SkillResourceVersion("guide.txt", child_asset),),
    )

    candidates = (
        CapabilityContribution.from_declaration(
            SkillDefinition(
                SkillSpec("child-skill", "Use the child guide."),
                child_ref,
            )
        ),
        CapabilityContribution.from_declaration(
            SkillDefinition(
                SkillSpec("unreachable-skill", "Not reachable from a child execution."),
                SkillSourceRef("missing", "unreachable-skill"),
            )
        ),
    )
    specs = {
        "parent": AgentSpec(
            "parent",
            allow_skills=(),
            allow_subagents=("child",),
        ),
        "child": AgentSpec(
            "child",
            allow_skills=("child-skill",),
            allow_subagents=("grandchild",),
        ),
        "grandchild": AgentSpec(
            "grandchild",
            allow_skills=("unreachable-skill",),
            allow_subagents=(),
        ),
    }
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=candidates,
        agents=specs,
    )
    catalog = AgentCatalog(
        {
            agent_id: compiler.compile(spec)
            for agent_id, spec in specs.items()
        }
    )
    resolver = _RuntimeBindingResolver(
        catalog,
        compiler,
    )
    return _BindingFixture(
        compiler,
        catalog,
        resolver,
        assets,
        compiler.bind(catalog.root_agent("parent")),
    )


def _resolved_child(binding_contract: AgentBindingContract) -> AgentBindingContract:
    assert binding_contract.subagent_ids == ("child",)
    assert len(binding_contract.subagent_bindings) == 1
    child = binding_contract.subagent_bindings[0]
    assert child.agent_spec.id == "child"
    assert child.subagents == ()
    assert child.subagent_bindings == ()
    return child


def _skill_ref(child: AgentBindingContract) -> SkillSourceRef:
    pin = next(item for item in child.selected if item.kind == "skill")
    skill = SkillDefinition.from_contract(pin.contract)
    assert skill.source_ref is not None
    assert skill.source_ref.resource_versions
    return skill.source_ref


async def _read_skill(
    fixture: _BindingFixture,
    ref: SkillSourceRef,
) -> bytes:
    source = AssetSkillResourceSource("application", fixture.assets)
    return await source.read(ref, "guide.txt")


@pytest.mark.asyncio
async def test_binding_resolution_preserves_direct_child_asset_versions() -> None:
    fixture = await _fixture()
    try:
        resolved = await fixture.resolver.resolve(fixture.binding)
        ref = _skill_ref(_resolved_child(resolved.binding_contract))

        assert [item.path for item in ref.resource_versions] == ["guide.txt"]
        await fixture.assets.put(
            AssetKey("skill", "child-skill/guide.txt"),
            b"changed",
        )
        assert await _read_skill(fixture, ref) == b"original"
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_task_capture_does_not_build_static_root_closure() -> None:
    fixture = await _fixture()
    try:
        graph = TaskGraph(
            "graph",
            (
                TaskNode(
                    "root",
                    input={
                        "task_id": "linktools.ai.agent",
                        "task_revision": 1,
                        "binding_contract": fixture.binding.binding_contract.to_payload(),
                    },
                ),
            ),
        )
        admission = TaskGraphAdmission.from_request(
            TaskGraphRequest(
                graph,
                Principal("principal", "tenant"),
                "task-capture",
                TaskGraphLimits(),
            )
        )
        captures = TaskCapabilityCaptureStore(
            "namespace",
            fixture.compiler,
            fixture.resolver,
            InMemoryObjectStore("task"),
            agent_task_id="linktools.ai.agent",
        )

        capability_capture = await captures.capture(admission, graph)

        assert capability_capture.roots == {}
        resolved_binding = capability_capture.bindings[fixture.binding.binding_digest]
        assert resolved_binding.binding_digest != fixture.binding.binding_digest
        assert _skill_ref(_resolved_child(resolved_binding)).resource_versions
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_runtime_start_admits_resolved_binding() -> None:
    fixture = await _fixture()
    try:
        execution = _RecordingExecution()
        runtime = Runtime(
            fixture.catalog,
            fixture.compiler,
            execution,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            None,
            namespace="namespace",
            context=RuntimeContext(None),
            _binding_resolver=fixture.resolver,
        )

        compiled_agent = fixture.catalog.root_agent("parent")
        started = await runtime._start_for_agent(
            compiled_agent.spec.id,
            compiled_agent.spec.revision,
            "prompt",
            files=(),
            output=None,
            principal=None,
            session_id=None,
            idempotency_key="runtime-resolution",
            memory_scope=None,
            mode="run",
            planning=None,
            thinking=None,
        )

        assert started.execution_id == "execution"
        assert execution.binding_contract is not None
        assert execution.binding_digest == execution.binding_contract.binding_digest
        assert _skill_ref(_resolved_child(execution.binding_contract)).resource_versions
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_execution_binding_uses_selected_child_asset_versions() -> None:
    fixture = await _fixture()
    try:
        resolved = await fixture.resolver.resolve(fixture.binding)

        assert resolved.binding_contract != fixture.binding.binding_contract
        assert _skill_ref(_resolved_child(resolved.binding_contract)).resource_versions
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_binding_resolution_restores_mcp_execution_contract() -> None:
    server = MCPServerSpec(
        "server",
        "python",
        ("resource:literal-value",),
    )
    specification = AgentSpec(
        "agent",
        allow_tools=(mcp_server_selector(server.id),),
        allow_skills=(),
        allow_subagents=(),
        allow_runtime_capabilities=(),
    )
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(CapabilityContribution.from_declaration(server),),
        agents={specification.id: specification},
    )
    catalog = AgentCatalog(
        {specification.id: compiler.compile(specification)}
    )
    resolver = _RuntimeBindingResolver(
        catalog,
        compiler,
    )

    resolved = await resolver.resolve(
        compiler.bind(catalog.root_agent(specification.id))
    )

    pin = next(item for item in resolved.binding_contract.selected if item.kind == "mcp")
    selected = resolved.compiled_agent.selected_mcp
    assert pin.contract["execution_policy"] == {
        "version": 1,
        "boundary": "host-stdio",
    }
    resource_versions = MCPServerSpecCodec().decode_binding_payload(
        pin.contract,
        declaration=server,
    )
    assert server.args == ("resource:literal-value",)
    assert resource_versions is None
    assert len(selected) == 1
    assert (
        selected[0].id,
        selected[0].revision,
        selected[0].contract,
    ) == (
        pin.id,
        pin.revision,
        dict(pin.contract),
    )


@pytest.mark.asyncio
async def test_binding_resolution_uses_sandbox_policy_without_workspace(
    tmp_path: Path,
) -> None:
    server = MCPServerSpec("server", "python")
    specification = AgentSpec(
        "agent",
        allow_tools=(mcp_server_selector(server.id),),
        allow_skills=(),
        allow_subagents=(),
        allow_runtime_capabilities=(),
    )
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(CapabilityContribution.from_declaration(server),),
        agents={specification.id: specification},
    )
    catalog = AgentCatalog(
        {specification.id: compiler.compile(specification)}
    )
    sandbox = BubblewrapSandbox(
        runtime_root=tmp_path,
        bwrap_executable=tmp_path / "bwrap",
    )
    resolver = _RuntimeBindingResolver(
        catalog,
        compiler,
        sandbox=sandbox,
    )

    resolved = await resolver.resolve(
        compiler.bind(catalog.root_agent(specification.id))
    )

    pin = next(item for item in resolved.binding_contract.selected if item.kind == "mcp")
    assert pin.contract["execution_policy"] == sandbox.stdio_execution_policy()


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_resources", (False, True))
async def test_existing_child_mcp_resolves_asset_versions(
    parent_resources: bool,
) -> None:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        root = AssetKey("mcp", "server/assets")
        resource = AssetKey("mcp", "server/assets/script.py")
        await store.put(resource, b"print('ok')")
        codec = MCPServerSpecCodec()
        server = MCPServerSpec(
            "server",
            "python",
            ("resource:script.py",),
            root,
        )
        versions = await store.resolve_versions((resource,))
        contribution = CapabilityContribution.from_mcp_contract(
            codec.to_binding_payload(
                server,
                versions,
                asset_source_id="application",
            ),
            server,
        )
        specs = {
            "parent": AgentSpec(
                "parent",
                allow_tools=(
                    (mcp_server_selector(server.id),)
                    if parent_resources
                    else ()
                ),
                allow_skills=(),
                allow_subagents=("child",),
                allow_runtime_capabilities=(),
            ),
            "child": AgentSpec(
                "child",
                allow_tools=(mcp_server_selector(server.id),),
                allow_skills=(),
                allow_subagents=(),
                allow_runtime_capabilities=(),
            ),
        }
        compiler = AgentCompiler(
            model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
            candidates=(contribution,),
            agents=specs,
        )
        catalog = AgentCatalog(
            {
                agent_id: compiler.compile(spec)
                for agent_id, spec in specs.items()
            }
        )
        resolver = _RuntimeBindingResolver(catalog, compiler)
        binding = compiler.bind(catalog.root_agent("parent")).binding_contract

        await store.put(resource, b"print('updated')")
        resolved = await resolver.resolve_contract(binding)
        child = resolved.subagent_bindings[0]
        pin = next(item for item in child.selected if item.kind == "mcp")
        current = child.agent_spec
        assert current.id == "child"
        resolved_versions = codec.decode_binding_payload(
            pin.contract,
            declaration=server,
        )
        assert pin.contract["asset_source_id"] == "application"
        assert resolved_versions is not None
        assert await store.read_versions(resolved_versions) == (b"print('ok')",)
        assert await resolver.resolve_contract(resolved) == resolved
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_runtime_storage_snapshot_preserves_asset_version_refs(
    tmp_path: Path,
) -> None:
    fixture = await _fixture()
    storage_root = tmp_path / "state"
    state = RuntimeStorage.filesystem(storage_root)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    try:
        resolved = await fixture.resolver.resolve(fixture.binding)
        now = datetime.now(timezone.utc)
        await state.execution.executions.create(
            ExecutionRecord(
                execution_id="execution",
                session_id=None,
                parent_execution_id=None,
                root_execution_id="execution",
                previous_execution_id=None,
                fork_base_execution_id=None,
                lineage_kind=ExecutionLineageKind.RUN,
                status=ExecutionStatus.PENDING_START,
                revision=0,
                event_sequence=0,
                agent_run_sequence=0,
                error_code=None,
                safe_error_details={},
                created_at=now,
                updated_at=now,
                mode="run",
                planning=False,
                thinking=False,
                binding=resolved.binding_contract,
                principal_id="principal",
                principal_kind="service",
                stored_user_input=StoredUserInput(
                    "text",
                    StoredPayload.inline_text("prompt"),
                ),
            )
        )
        resolved_ref = _skill_ref(_resolved_child(resolved.binding_contract))
    finally:
        await state.close()

    await fixture.assets.put(
        AssetKey("skill", "child-skill/guide.txt"),
        b"changed",
    )

    snapshot_store = InMemoryObjectStore("snapshot")
    read_state = RuntimeStorage.filesystem(storage_root)
    await read_state.initialize(
        namespace="namespace",
        tenant_id="tenant",
        read_only=True,
    )
    try:
        snapshot_ref = await read_state.export_snapshot(
            object_store=snapshot_store,
            limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
        )
    finally:
        await read_state.close()

    restored_root = tmp_path / "restored"
    await RuntimeStorage.restore_snapshot(
        snapshot_ref,
        object_store=snapshot_store,
        root=restored_root,
        limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
    )
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(
        namespace="namespace",
        tenant_id="tenant",
        read_only=True,
    )
    try:
        execution = await restored.execution.executions.get(
            "execution",
            tenant_id="tenant",
        )
        assert execution is not None
        restored_ref = _skill_ref(_resolved_child(execution.binding))
        assert restored_ref == resolved_ref
        assert await _read_skill(fixture, restored_ref) == b"original"
    finally:
        await restored.close()
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_runtime_storage_snapshot_restores_task_capability_manifest(
    tmp_path: Path,
) -> None:
    fixture = await _fixture()
    storage_root = tmp_path / "task-state"
    state = RuntimeStorage.filesystem(storage_root)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    graph = TaskGraph(
        "graph",
        (
            TaskNode(
                "root",
                input={
                    "task_id": "linktools.ai.agent",
                    "task_revision": 1,
                    "binding_contract": fixture.binding.binding_contract.to_payload(),
                },
            ),
        ),
    )
    admission = TaskGraphAdmission.from_request(
        TaskGraphRequest(
            graph,
            Principal("principal", "tenant"),
            "task-snapshot",
            TaskGraphLimits(),
        )
    )
    try:
        capabilities = TaskCapabilityCaptureStore(
            "namespace",
            fixture.compiler,
            fixture.resolver,
            state.object_store(RuntimeDomain.TASK),
            agent_task_id="linktools.ai.agent",
        )
        await capabilities.capture(admission, graph)
        await state.task.admissions.admit(admission, graph)
        loaded = await capabilities.load(admission)
        binding = loaded.bindings[fixture.binding.binding_digest]
        resolved_ref = _skill_ref(_resolved_child(binding))
    finally:
        await state.close()

    snapshot_store = InMemoryObjectStore("task-snapshot")
    read_state = RuntimeStorage.filesystem(storage_root)
    await read_state.initialize(
        namespace="namespace",
        tenant_id="tenant",
        read_only=True,
    )
    try:
        snapshot_ref = await read_state.export_snapshot(
            object_store=snapshot_store,
            limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
        )
    finally:
        await read_state.close()

    restored_root = tmp_path / "task-restored"
    await RuntimeStorage.restore_snapshot(
        snapshot_ref,
        object_store=snapshot_store,
        root=restored_root,
        limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
    )
    restored = RuntimeStorage.from_root(restored_root)
    await restored.initialize(
        namespace="namespace",
        tenant_id="tenant",
        read_only=True,
    )
    try:
        restored_capabilities = TaskCapabilityCaptureStore(
            "namespace",
            fixture.compiler,
            fixture.resolver,
            restored.object_store(RuntimeDomain.TASK),
            agent_task_id="linktools.ai.agent",
        )
        loaded = await restored_capabilities.load(admission)
        binding = loaded.bindings[fixture.binding.binding_digest]
        restored_ref = _skill_ref(_resolved_child(binding))
        assert restored_ref == resolved_ref
        assert await _read_skill(fixture, restored_ref) == b"original"
    finally:
        await restored.close()
        await fixture.assets.close()
