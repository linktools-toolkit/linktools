#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for execution-owned Asset version bindings."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
    AssetSkillSource,
    CapabilityContribution,
    SkillDefinition,
    SkillResource,
    SkillSourceRef,
)
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    JsonValue,
    Page,
    Principal,
    canonical_json_bytes,
    canonical_sha256,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._agent_binding_resolver import _AgentBindingResolver
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._runtime_identity import task_graph_binding_capture_key
from linktools.ai.runtime._runtime_service import Runtime
from linktools.ai.runtime._task_graph_binding_capture import TaskGraphBindingCaptureStore
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
from linktools.ai.storage import (
    InMemoryObjectStore,
    ObjectStat,
    StorageOverlay,
    StoredPayload,
)
from linktools.ai.task import (
    Task,
    TaskExpansionContext,
    TaskExpander,
    TaskExpanderRef,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphLaunch,
    TaskGraphRequest,
    RecoverGraphRequest,
    TaskNode,
    TaskNodeContext,
    TaskRef,
)
from linktools.ai.workspace import BubblewrapSandbox


@dataclass(frozen=True)
class _BindingFixture:
    compiler: AgentCompiler
    catalog: AgentCatalog
    resolver: _AgentBindingResolver
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
        (SkillResource("guide.txt", child_asset),),
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
    resolver = _AgentBindingResolver(
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
    assert skill.source_ref.resources
    return skill.source_ref


async def _read_skill(
    fixture: _BindingFixture,
    ref: SkillSourceRef,
) -> bytes:
    source = AssetSkillSource("application", fixture.assets)
    return await source.read(ref, "guide.txt")


@pytest.mark.asyncio
async def test_binding_resolution_preserves_direct_child_asset_versions() -> None:
    fixture = await _fixture()
    try:
        resolved = await fixture.resolver.resolve(fixture.binding)
        ref = _skill_ref(_resolved_child(resolved.binding_contract))

        assert [item.path for item in ref.resources] == ["guide.txt"]
        await fixture.assets.put(
            AssetKey("skill", "child-skill/guide.txt"),
            b"changed",
        )
        assert await _read_skill(fixture, ref) == b"original"
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_task_binding_capture_keeps_agent_binding_in_task_declaration() -> None:
    fixture = await _fixture()
    try:
        class CaptureRunner:
            pass

        task = Task.from_runner(
            "test.capture-agent",
            CaptureRunner(),  # type: ignore[arg-type]
            contract={
                "version": 1,
                "type": "agent",
                "effect_policy": "none",
                "output_contract": {"kind": "json"},
                "reconcile": False,
                "config": {
                    "agent_id": "parent",
                    "agent_revision": 1,
                    "binding_contract": fixture.binding.binding_contract.to_payload(),
                    "input_mode": "literal",
                },
            },
        )
        graph = TaskGraph(
            "graph",
            (
                TaskNode("root", task=task),
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
        captures = TaskGraphBindingCaptureStore(
            "namespace",
            InMemoryObjectStore("task"),
        )

        capture = await captures.capture(admission, graph, tasks=(task,))

        assert set(capture.tasks) == {(task.id, task.revision)}
        declaration = capture.tasks[(task.id, task.revision)]
        config = declaration["config"]
        assert isinstance(config, Mapping)
        assert config["binding_contract"] == fixture.binding.binding_contract.to_payload()
        assert not capture.expanders
    finally:
        await fixture.assets.close()


class _ConcurrentCaptureObjectStore(InMemoryObjectStore):
    def __init__(self, capture_key: str) -> None:
        super().__init__("task-capture-race")
        self._capture_key = capture_key
        self._initial_reads = 0
        self._both_read = asyncio.Event()

    async def stat(self, key: str) -> ObjectStat | None:
        value = await super().stat(key)
        if key == self._capture_key and self._initial_reads < 2:
            self._initial_reads += 1
            if self._initial_reads == 2:
                self._both_read.set()
            await self._both_read.wait()
        return value


@pytest.mark.asyncio
async def test_concurrent_task_capture_keeps_the_first_manifest() -> None:
    fixture = await _fixture()
    try:
        async def run_task(_context: TaskNodeContext[None]) -> JsonValue:
            return {"captured": True}

        class Expander:
            id = "test.capture-expander"
            revision = 1

            def expand(
                self,
                _context: TaskExpansionContext,
            ) -> tuple[TaskNode, ...]:
                return ()

        required = Task("test.capture-required", run_task, effect_policy="none")
        unused_a = Task("test.capture-unused-a", run_task, effect_policy="none")
        unused_b = Task("test.capture-unused-b", run_task, effect_policy="none")
        expander = TaskExpander("test.capture-expander", Expander().expand)
        graph = TaskGraph(
            "capture-race",
            (
                TaskNode(
                    "root",
                    task=required,
                    expander=TaskExpanderRef(expander.id, expander.revision),
                ),
            ),
        )
        admission = TaskGraphAdmission.from_request(
            TaskGraphRequest(
                graph,
                Principal("principal", "tenant"),
                "capture-race-run",
                TaskGraphLimits(),
            )
        )
        key = task_graph_binding_capture_key(
            "namespace",
            "tenant",
            admission.graph_id,
            admission.initial_request_digest,
        )
        objects = _ConcurrentCaptureObjectStore(key)
        captures = TaskGraphBindingCaptureStore("namespace", objects)

        first, second = await asyncio.gather(
            captures.capture(
                admission,
                graph,
                tasks=(required, unused_a),
                expanders=(expander,),
            ),
            captures.capture(
                admission,
                graph,
                tasks=(required, unused_b),
                expanders=(expander,),
            ),
        )

        assert first.tasks == second.tasks
        loaded = await captures.load(admission)
        assert loaded.tasks == first.tasks
        captured_optional = {
            identity[0]
            for identity in loaded.tasks
            if identity[0] in {unused_a.id, unused_b.id}
        }
        assert len(captured_optional) == 1
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
        assert _skill_ref(_resolved_child(execution.binding_contract)).resources
    finally:
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_execution_binding_uses_selected_child_asset_versions() -> None:
    fixture = await _fixture()
    try:
        resolved = await fixture.resolver.resolve(fixture.binding)

        assert resolved.binding_contract != fixture.binding.binding_contract
        assert _skill_ref(_resolved_child(resolved.binding_contract)).resources
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
        allow_capabilities=(),
    )
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").capture(),
        candidates=(CapabilityContribution.from_declaration(server),),
        agents={specification.id: specification},
    )
    catalog = AgentCatalog(
        {specification.id: compiler.compile(specification)}
    )
    resolver = _AgentBindingResolver(
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
        allow_capabilities=(),
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
    resolver = _AgentBindingResolver(
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
                allow_capabilities=(),
            ),
            "child": AgentSpec(
                "child",
                allow_tools=(mcp_server_selector(server.id),),
                allow_skills=(),
                allow_subagents=(),
                allow_capabilities=(),
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
        resolver = _AgentBindingResolver(catalog, compiler)
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
async def test_runtime_storage_snapshot_restores_task_binding_capture(
    tmp_path: Path,
) -> None:
    fixture = await _fixture()
    class CaptureRunner:
        pass

    task = Task.from_runner(
        "test.capture-agent",
        CaptureRunner(),  # type: ignore[arg-type]
        contract={
            "version": 1,
            "type": "agent",
            "effect_policy": "none",
            "output_contract": {"kind": "json"},
            "reconcile": False,
            "config": {
                "agent_id": "parent",
                "agent_revision": 1,
                "binding_contract": fixture.binding.binding_contract.to_payload(),
                "input_mode": "literal",
            },
        },
    )
    storage_root = tmp_path / "task-state"
    state = RuntimeStorage.filesystem(storage_root)
    await state.initialize(namespace="namespace", tenant_id="tenant")
    graph = TaskGraph(
        "graph",
        (
            TaskNode("root", task=task),
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
        captures = TaskGraphBindingCaptureStore(
            "namespace",
            state.object_store(RuntimeDomain.TASK),
        )
        await captures.capture(admission, graph, tasks=(task,))
        await state.task.admissions.admit(admission, graph)
        loaded = await captures.load(admission)
        assert set(loaded.tasks) == {("test.capture-agent", 1)}
        assert not loaded.expanders
        captured_declaration = loaded.tasks[("test.capture-agent", 1)]
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
        restored_captures = TaskGraphBindingCaptureStore(
            "namespace",
            restored.object_store(RuntimeDomain.TASK),
        )
        loaded = await restored_captures.load(admission)
        assert set(loaded.tasks) == {("test.capture-agent", 1)}
        assert not loaded.expanders
        assert loaded.tasks[("test.capture-agent", 1)] == captured_declaration
    finally:
        await restored.close()
        await fixture.assets.close()


@pytest.mark.asyncio
async def test_recover_pending_preflights_all_subjects_before_dispatch() -> None:
    actor = Principal("operator", "tenant")
    launches = (
        TaskGraphLaunch("graph-a", Principal("admitted-a", "tenant"), TaskGraphLimits()),
        TaskGraphLaunch("graph-b", Principal("admitted-b", "tenant"), TaskGraphLimits()),
    )

    class Admissions:
        async def list_recoverable_page(
            self,
            *,
            cursor: str | None,
            limit: int,
        ) -> Page[TaskGraphLaunch]:
            assert cursor is None
            assert limit == 10
            return Page(launches)

    admissions = Admissions()
    events: list[tuple[str, str, Principal]] = []

    class Engine:
        def __init__(self, *, fail_on: str | None = None) -> None:
            self.fail_on = fail_on

        async def _activate_graph(
            self,
            graph_id: str,
            principal: Principal,
            *,
            recovery_nodes: object,
            recovery_principal: Principal,
        ) -> None:
            del recovery_nodes, recovery_principal
            events.append(("preflight", graph_id, principal))
            if graph_id == self.fail_on:
                raise AIError(ErrorCode.BINDING_NOT_REGISTERED)

    class GraphService:
        def __init__(self, *, fail_authorize_on: str | None = None) -> None:
            self.fail_authorize_on = fail_authorize_on

        async def recovery_nodes(
            self,
            graph_id: str,
            *,
            principal: Principal,
        ) -> object:
            events.append(("authorize", graph_id, principal))
            if graph_id == self.fail_authorize_on:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return object()

        async def recover(
            self,
            graph_id: str,
            request: RecoverGraphRequest,
        ) -> str:
            events.append(("recover", graph_id, request.principal))
            return graph_id

    runtime = object.__new__(Runtime)
    runtime._closed = False
    runtime._closing = False
    runtime._task_admissions = admissions
    runtime._default_principal = actor
    runtime._context = SimpleNamespace(tenant_id="tenant")
    runtime._namespace = "runtime"
    runtime._graph_service = GraphService()

    page = await runtime._recover_pending_tasks(
        Engine(),
        cursor=None,
        limit=10,
        principal=actor,
    )

    assert page.items == ("graph-a", "graph-b")
    assert events == [
        ("authorize", "graph-a", actor),
        ("preflight", "graph-a", launches[0].principal),
        ("authorize", "graph-b", actor),
        ("preflight", "graph-b", launches[1].principal),
        ("recover", "graph-a", actor),
        ("recover", "graph-b", actor),
    ]

    events.clear()
    with pytest.raises(AIError) as missing_definition:
        await runtime._recover_pending_tasks(
            Engine(fail_on="graph-b"),
            cursor=None,
            limit=10,
            principal=actor,
        )
    assert missing_definition.value.code is ErrorCode.BINDING_NOT_REGISTERED
    assert [event[0] for event in events] == [
        "authorize",
        "preflight",
        "authorize",
        "preflight",
    ]

    events.clear()
    runtime._graph_service = GraphService(fail_authorize_on="graph-b")
    with pytest.raises(AIError) as denied:
        await runtime._recover_pending_tasks(
            Engine(),
            cursor=None,
            limit=10,
            principal=actor,
        )
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert [event[0] for event in events] == [
        "authorize",
        "preflight",
        "authorize",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("format_version", "corruption", "expected_code"),
    (
        (2, "duplicate_task", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "duplicate_expander", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "missing_array", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "invalid_task", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "invalid_schema", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "invalid_expander", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "wrong_kind", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, "unexpected_field", ErrorCode.STORAGE_INTEGRITY_ERROR),
        (3, "unknown_format", ErrorCode.STORAGE_VERSION_UNSUPPORTED),
    ),
)
async def test_task_binding_capture_reader_rejects_invalid_declaration_manifests(
    format_version: object,
    corruption: str,
    expected_code: ErrorCode,
) -> None:
    objects = InMemoryObjectStore("task-capture-reader")
    graph = TaskGraph(
        "capture-reader",
        (
            TaskNode("root", task=TaskRef("test.capture", 1)),
        ),
    )
    admission = TaskGraphAdmission.from_request(
        TaskGraphRequest(
            graph,
            Principal("principal", "tenant"),
            "capture-reader-run",
            TaskGraphLimits(),
        )
    )
    declaration: dict[str, object] = {
        "version": 1,
        "id": "test.capture",
        "revision": 1,
        "type": "function",
        "effect_policy": "none",
        "output_contract": {"kind": "json"},
        "reconcile": False,
    }
    expander_declaration: dict[str, object] = {
        "version": 1,
        "id": "test.expander",
        "revision": 1,
    }
    task_declarations: list[dict[str, object]] = [declaration]
    if corruption == "duplicate_task":
        task_declarations = [declaration, declaration]
    elif corruption == "invalid_task":
        task_declarations = [{**declaration, "effect_policy": []}]
    elif corruption == "invalid_schema":
        task_declarations = [
            {
                **declaration,
                "output_contract": {
                    "kind": "schema",
                    "schema": {"type": 42},
                },
            }
        ]
    expander_declarations: list[dict[str, object]] = []
    if corruption == "duplicate_expander":
        expander_declarations = [expander_declaration, expander_declaration]
    elif corruption == "invalid_expander":
        expander_declarations = [
            {**expander_declaration, "version": 1.0}
        ]
    manifest: dict[str, object] = {
        "kind": "task-definition-capture",
        "format_version": format_version,
        "namespace": "namespace",
        "tenant_id": "tenant",
        "graph_id": admission.graph_id,
        "request_digest": admission.initial_request_digest,
        "tasks": task_declarations,
        "expanders": expander_declarations,
    }
    if corruption == "wrong_kind":
        manifest["kind"] = "invalid-kind"
    if corruption == "unexpected_field":
        manifest["roots"] = {}
    if corruption == "missing_array":
        del manifest["expanders"]
    payload = canonical_json_bytes(manifest)
    async def chunks():
        yield payload

    await objects.put(
        task_graph_binding_capture_key(
            "namespace",
            "tenant",
            admission.graph_id,
            admission.initial_request_digest,
        ),
        chunks(),
        expected_size=len(payload),
        expected_digest=canonical_sha256(manifest),
    )
    reader = TaskGraphBindingCaptureStore("namespace", objects)
    with pytest.raises(AIError) as error:
        await reader.load(admission)
    assert error.value.code is expected_code
