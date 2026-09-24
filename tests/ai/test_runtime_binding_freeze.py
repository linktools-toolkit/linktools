#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for execution-owned binding dependency snapshots."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai.agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    SemanticPin,
)
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AssetSkillResourceSource,
    CapabilityContribution,
    FrozenSkillResourceSource,
    LocalSkillResourceSource,
    SkillDefinition,
    SkillSourceRef,
    SkillSourceRegistry,
)
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._binding_freeze import _RuntimeBindingFreezer
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._runtime_service import Runtime
from linktools.ai.runtime._task_capability_snapshot import TaskCapabilitySnapshotStore
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionRequest
from linktools.ai.runtime.state import RuntimeDomain, RuntimeState, SnapshotLimits
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


@dataclass(frozen=True)
class _BindingFixture:
    compiler: AgentCompiler
    catalog: AgentCatalog
    freezer: _RuntimeBindingFreezer
    objects: InMemoryObjectStore
    binding: AgentBinding
    resource: Path


class _RecordingExecution:
    def __init__(self) -> None:
        self.binding_digest: str | None = None
        self.binding_snapshot: AgentBindingSnapshot | None = None

    async def start(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        dependency_hold_id: str | None = None,
        binding_snapshot: AgentBindingSnapshot | None = None,
    ) -> ExecutionHandle:
        del request, dependency_hold_id
        self.binding_digest = binding_digest
        self.binding_snapshot = binding_snapshot
        return ExecutionHandle("execution")


def _fixture(tmp_path: Path) -> _BindingFixture:
    skill_root = tmp_path / "skills"
    package = skill_root / "child-skill"
    package.mkdir(parents=True)
    resource = package / "guide.txt"
    resource.write_text("original", encoding="utf-8")

    candidates = (
        CapabilityContribution.from_declaration(
            SkillDefinition(
                SkillSpec("child-skill", "Use the child guide."),
                SkillSourceRef("application", "child-skill"),
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
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=candidates,
        agents=specs,
    )
    catalog = AgentCatalog(
        {
            agent_id: compiler.compile(spec)
            for agent_id, spec in specs.items()
        }
    )
    objects = InMemoryObjectStore("execution")
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(
            (LocalSkillResourceSource("application", skill_root),)
        ),
        objects,
        workspace=None,
    )
    return _BindingFixture(
        compiler,
        catalog,
        freezer,
        objects,
        compiler.bind(catalog.root_definition("parent")),
        resource,
    )


def _frozen_child(snapshot: AgentBindingSnapshot) -> AgentBindingSnapshot:
    assert snapshot.subagent_ids == ("child",)
    assert len(snapshot.subagent_bindings) == 1
    child = snapshot.subagent_bindings[0]
    assert child.agent_spec.id == "child"
    assert child.subagents == ()
    assert child.subagent_bindings == ()
    return child


def _skill_snapshot(child: AgentBindingSnapshot):
    pin = next(item for item in child.selected if item.kind == "skill")
    skill = SkillDefinition.from_semantic_contract(pin.contract)
    assert skill.source_ref is not None
    assert skill.source_ref.snapshot is not None
    return skill.source_ref.snapshot


def _physical_ref(store: InMemoryObjectStore | object, ref: ObjectRef) -> ObjectRef:
    store_id = getattr(store, "store_id")
    assert isinstance(store_id, str)
    assert ref.store_id == "runtime"
    return ObjectRef(store_id, ref.key, ref.digest, ref.size)


@pytest.mark.asyncio
async def test_binding_freeze_captures_only_direct_child_resources(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    frozen = await fixture.freezer.freeze(fixture.binding)
    snapshot = _skill_snapshot(_frozen_child(frozen.snapshot))

    assert snapshot.store_id == "runtime"
    assert await fixture.objects.stat(snapshot.key) is not None

    fixture.resource.write_text("changed", encoding="utf-8")
    source = FrozenSkillResourceSource(
        "application",
        {"child-skill": _physical_ref(fixture.objects, snapshot)},
        fixture.objects,
    )
    assert await source.read("child-skill", "guide.txt") == b"original"


@pytest.mark.asyncio
async def test_task_capture_does_not_build_static_root_closure(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    graph = TaskGraph(
        "graph",
        (
            TaskNode(
                "root",
                input={
                    "type": "linktools.ai.agent",
                    "version": 1,
                    "binding": fixture.binding.snapshot.to_payload(),
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
    snapshots = TaskCapabilitySnapshotStore(
        "namespace",
        fixture.compiler,
        fixture.freezer,
        InMemoryObjectStore("task"),
        agent_task_type="linktools.ai.agent",
    )

    frozen = await snapshots.capture(admission, graph)

    assert frozen.roots == {}
    frozen_binding = frozen.bindings[fixture.binding.digest]
    assert frozen_binding.binding_digest != fixture.binding.digest
    snapshot = _skill_snapshot(
        _frozen_child(frozen_binding)
    )
    assert snapshot.store_id == "runtime"



@pytest.mark.asyncio
async def test_runtime_start_admits_frozen_binding(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
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
        _binding_freezer=fixture.freezer,
    )

    started = await runtime._start_for_agent(
        fixture.catalog.root_definition("parent").digest,
        "prompt",
        files=(),
        output=None,
        principal=None,
        session_id=None,
        idempotency_key="runtime-freeze",
        memory_scope=None,
        mode="run",
        planning=None,
        thinking=None,
    )

    assert started.execution_id == "execution"
    assert execution.binding_snapshot is not None
    assert execution.binding_digest == execution.binding_snapshot.binding_digest
    snapshot = _skill_snapshot(_frozen_child(execution.binding_snapshot))
    assert snapshot.store_id == "runtime"


@pytest.mark.asyncio
async def test_execution_binding_freezes_selected_child_skills(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    frozen = await fixture.freezer.freeze(fixture.binding)

    assert frozen.snapshot != fixture.binding.snapshot
    child = _frozen_child(frozen.snapshot)
    assert _skill_snapshot(child).store_id == "runtime"


@pytest.mark.asyncio
async def test_binding_freeze_restores_frozen_mcp_contract() -> None:
    server = MCPServerSpec("server", "python")
    specification = AgentSpec(
        "agent",
        allow_tools=(mcp_server_selector(server.id),),
        allow_skills=(),
        allow_subagents=(),
        allow_capabilities=(),
    )
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        candidates=(CapabilityContribution.from_declaration(server),),
        agents={specification.id: specification},
    )
    catalog = AgentCatalog(
        {specification.id: compiler.compile(specification)}
    )
    freezer = _RuntimeBindingFreezer(
        catalog,
        compiler,
        SkillSourceRegistry(),
        InMemoryObjectStore("runtime"),
        workspace=None,
    )

    frozen = await freezer.freeze(
        compiler.bind(catalog.root_definition(specification.id))
    )

    pin = next(item for item in frozen.snapshot.selected if item.kind == "mcp")
    selected = frozen.definition.selected_mcp
    assert pin.contract["execution_policy"] == {
        "version": 1,
        "boundary": "host-stdio",
    }
    assert len(selected) == 1
    assert selected[0].semantic_contract == dict(pin.contract)
    assert selected[0].fingerprint == pin.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_resources", (False, True))
async def test_non_durable_snapshot_freezes_existing_child_mcp_resources(
    tmp_path: Path,
    parent_resources: bool,
) -> None:
    fixture = _fixture(tmp_path)
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        root = AssetKey("mcp", "server/assets")
        await store.put(AssetKey("mcp", "server/assets/script.py"), b"print('ok')")
        codec = MCPServerSpecCodec()
        pin = SemanticPin(
            "mcp",
            "server",
            codec.to_payload(
                MCPServerSpec(
                    "server",
                    "python",
                    ("resource:script.py",),
                    root,
                )
            ),
        )
        child = replace(
            fixture.compiler.bind_subagent(
                fixture.catalog.root_definition("child")
            ).snapshot,
            selected=(pin,),
        )
        snapshot = replace(
            fixture.binding.snapshot,
            selected=(pin,) if parent_resources else (),
            subagent_bindings=(child,),
        )
        freezer = _RuntimeBindingFreezer(
            fixture.catalog,
            fixture.compiler,
            SkillSourceRegistry(),
            fixture.objects,
            workspace=None,
            mcp_assets={"server": store},
        )
        await store.put(
            AssetKey("mcp", "server/assets/script.py"),
            b"print('updated')",
        )
        await store.put(AssetKey("skill", "unrelated"), b"changed")
        frozen = await freezer.freeze_snapshot(snapshot)
        server, resource_snapshot = codec.from_frozen_payload(
            frozen.subagent_bindings[0].selected[0].contract
        )
        assert server.resource_root == root
        assert resource_snapshot is not None
        resource_store = AssetStore.from_snapshot(
            ObjectRef(
                fixture.objects.store_id,
                resource_snapshot.key,
                resource_snapshot.digest,
                resource_snapshot.size,
            ),
            object_store=fixture.objects,
        )
        await resource_store.initialize()
        try:
            assert await resource_store.get(
                AssetKey("mcp", "server/assets/script.py")
            ) == b"print('updated')"
        finally:
            await resource_store.close()
        assert await freezer.freeze_snapshot(frozen) == frozen
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_runtime_state_snapshot_restores_frozen_skill_objects(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_root = tmp_path / "state"
    state = RuntimeState.filesystem(state_root)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    try:
        freezer = _RuntimeBindingFreezer(
            fixture.catalog,
            fixture.compiler,
            SkillSourceRegistry(
                (
                    LocalSkillResourceSource(
                        "application",
                        fixture.resource.parents[1],
                    ),
                )
            ),
            state.object_store(RuntimeDomain.EXECUTION),
            workspace=None,
        )
        frozen = await freezer.freeze(fixture.binding)
        now = datetime.now(timezone.utc)
        await state.execution.executions.create(
            ExecutionRecord(
                execution_id="execution",
                session_id=None,
                parent_execution_id=None,
                root_execution_id="execution",
                source_execution_id=None,
                base_execution_id=None,
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
                binding=frozen.snapshot,
                principal_id="principal",
                principal_kind="service",
                stored_user_input=StoredUserInput(
                    "text",
                    StoredPayload.inline_text("prompt"),
                ),
            )
        )
        child = _frozen_child(frozen.snapshot)
        skill_ref = _skill_snapshot(child)
        source = FrozenSkillResourceSource(
            "application",
            {
                "child-skill": _physical_ref(
                    state.object_store(RuntimeDomain.EXECUTION),
                    skill_ref,
                )
            },
            state.object_store(RuntimeDomain.EXECUTION),
        )
        assert await source.read("child-skill", "guide.txt") == b"original"
    finally:
        await state.close()

    snapshot_store = InMemoryObjectStore("snapshot")
    read_state = RuntimeState.filesystem(state_root)
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
    await RuntimeState.restore_snapshot(
        snapshot_ref,
        object_store=snapshot_store,
        root=restored_root,
        limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
    )
    restored = RuntimeState.from_root(restored_root)
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
        child = _frozen_child(execution.binding)
        skill_ref = _skill_snapshot(child)
        source = FrozenSkillResourceSource(
            "application",
            {
                "child-skill": _physical_ref(
                    restored.object_store(RuntimeDomain.EXECUTION),
                    skill_ref,
                )
            },
            restored.object_store(RuntimeDomain.EXECUTION),
        )
        assert await source.read("child-skill", "guide.txt") == b"original"
    finally:
        await restored.close()


@pytest.mark.asyncio
async def test_runtime_state_snapshot_restores_task_capability_manifest(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    state_root = tmp_path / "task-state"
    state = RuntimeState.filesystem(state_root)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    graph = TaskGraph(
        "graph",
        (
            TaskNode(
                "root",
                input={
                    "type": "linktools.ai.agent",
                    "version": 1,
                    "binding": fixture.binding.snapshot.to_payload(),
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
        freezer = _RuntimeBindingFreezer(
            fixture.catalog,
            fixture.compiler,
            SkillSourceRegistry(
                (
                    LocalSkillResourceSource(
                        "application",
                        fixture.resource.parents[1],
                    ),
                )
            ),
            state.object_store(RuntimeDomain.EXECUTION),
            workspace=None,
        )
        capabilities = TaskCapabilitySnapshotStore(
            "namespace",
            fixture.compiler,
            freezer,
            state.object_store(RuntimeDomain.TASK),
            agent_task_type="linktools.ai.agent",
        )
        await capabilities.capture(admission, graph)
        await state.task.admissions.admit(admission, graph)
        loaded = await capabilities.load(admission)
        binding = loaded.bindings[fixture.binding.digest]
        skill_ref = _skill_snapshot(_frozen_child(binding))
        source = FrozenSkillResourceSource(
            "application",
            {
                "child-skill": _physical_ref(
                    state.object_store(RuntimeDomain.EXECUTION),
                    skill_ref,
                )
            },
            state.object_store(RuntimeDomain.EXECUTION),
        )
        assert await source.read("child-skill", "guide.txt") == b"original"
    finally:
        await state.close()

    snapshot_store = InMemoryObjectStore("task-snapshot")
    read_state = RuntimeState.filesystem(state_root)
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
    await RuntimeState.restore_snapshot(
        snapshot_ref,
        object_store=snapshot_store,
        root=restored_root,
        limits=SnapshotLimits(max_entries=1024, max_bytes=8 * 1024 * 1024),
    )
    restored = RuntimeState.from_root(restored_root)
    await restored.initialize(
        namespace="namespace",
        tenant_id="tenant",
        read_only=True,
    )
    try:
        restored_freezer = _RuntimeBindingFreezer(
            fixture.catalog,
            fixture.compiler,
            SkillSourceRegistry(),
            restored.object_store(RuntimeDomain.EXECUTION),
            workspace=None,
        )
        restored_capabilities = TaskCapabilitySnapshotStore(
            "namespace",
            fixture.compiler,
            restored_freezer,
            restored.object_store(RuntimeDomain.TASK),
            agent_task_type="linktools.ai.agent",
        )
        loaded = await restored_capabilities.load(admission)
        binding = loaded.bindings[fixture.binding.digest]
        skill_ref = _skill_snapshot(_frozen_child(binding))
        source = FrozenSkillResourceSource(
            "application",
            {
                "child-skill": _physical_ref(
                    restored.object_store(RuntimeDomain.EXECUTION),
                    skill_ref,
                )
            },
            restored.object_store(RuntimeDomain.EXECUTION),
        )
        assert await source.read("child-skill", "guide.txt") == b"original"
    finally:
        await restored.close()
