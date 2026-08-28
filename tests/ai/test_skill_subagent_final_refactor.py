#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final Skill/Subagent refactor acceptance and durable-contract regressions."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from linktools.ai.agent import AgentBindingSnapshot, AgentCatalog, AgentCompiler, SemanticPin
from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
from linktools.ai.capability import (
    AssetSkillResourceSource,
    CapabilityGroup,
    LocalSkillResourceSource,
    SkillCapability,
    SkillDefinition,
    SkillLocation,
    SkillResourceView,
    SkillSourceRef,
    SkillSourceRegistry,
    capability_fingerprint,
    normalize_skill_resource_path,
)
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._factory import _default_workspace_store
from linktools.ai.runtime._planner import RuntimeTaskNodeRunner
from linktools.ai.runtime._subagent import SubagentDispatcher
from linktools.ai.runtime.service_api import ExecutionHandle, ExecutionRequest, ExecutionResult
from linktools.ai.spec import (
    AgentSpec,
    AgentSpecCodec,
    AgentUsageLimits,
    SkillMarkdownSpecCodec,
    SkillSpec,
    SkillSpecCodec,
)
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace

_FIXTURES = Path(__file__).with_name("fixtures")


def _load_json(name: str) -> dict[str, object]:
    value = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _compiler(agents: dict[str, AgentSpec], registry: ModelRegistry | None = None) -> AgentCompiler:
    return AgentCompiler(
        model_resolver=(registry or ModelRegistry.openai(model="gpt-test")).snapshot(),
        candidates=(),
        agents=agents,
    )


def test_v1_skill_wire_and_semantic_pin_fingerprint_are_stable() -> None:
    wire = (
        (_FIXTURES / "skill_spec_v1_golden.json")
        .read_text(encoding="utf-8")
        .strip()
        .encode("utf-8")
    )
    decoded = SkillSpecCodec().decode(wire)
    assert decoded == SkillSpec("legacy", "legacy instructions")
    assert decoded.description is None
    assert SkillSpecCodec().encode(decoded) == wire

    fixture = _load_json("skill_semantic_pin_v1_golden.json")
    pin_payload = fixture["pin"]
    assert isinstance(pin_payload, dict)
    pin = SemanticPin.from_payload(pin_payload)
    assert pin.contract_version == 1
    assert pin.fingerprint == fixture["fingerprint"]
    definition = SkillDefinition.from_semantic_contract(pin.contract)
    assert definition.semantic_contract == {
        "version": 1,
        "id": "legacy",
        "content": "legacy instructions",
    }
    assert (
        capability_fingerprint("skill", definition.id, definition.semantic_contract)
        == fixture["fingerprint"]
    )


def test_v1_binding_without_additive_fields_restores_with_original_digest() -> None:
    payload = _load_json("agent_binding_subagent_v1_golden.json")
    snapshot = AgentBindingSnapshot.from_payload(payload)
    compiler = _compiler(
        {
            "parent": AgentSpec("parent", allow_subagents=("child",)),
            "child": AgentSpec("child", allow_subagents=()),
        }
    )

    restored = compiler.restore(snapshot)

    assert restored.digest == "e79d77baf85afd6aa51059c215895d0c838d7d0bc78f3e21f4234fec4ae5a5d8"
    assert restored.snapshot.version == 1
    assert restored.snapshot.selected_subagents == ("child",)
    assert restored.snapshot.subagents[0].to_payload() == {"kind": "agent", "id": "child"}
    assert restored.snapshot.to_payload() == payload


def test_skill_and_agent_use_first_formal_v1_contracts() -> None:
    skill = SkillSpec("review", "instructions", "Review changes")
    assert SkillSpecCodec().to_payload(skill) == {
        "version": 1,
        "id": "review",
        "content": "instructions",
        "description": "Review changes",
    }
    wire = SkillSpecCodec().to_wire_payload(skill)
    assert wire["description"] == "Review changes"
    assert SkillSpecCodec().decode(SkillSpecCodec().encode(skill)) == skill

    plain = SkillDefinition(SkillSpec("review", "instructions"))
    described = SkillDefinition(skill)
    assert plain.semantic_contract == {
        "version": 1,
        "id": "review",
        "content": "instructions",
    }
    assert described.semantic_contract == {
        "version": 1,
        "id": "review",
        "content": "instructions",
        "description": "Review changes",
    }
    assert capability_fingerprint(
        "skill", plain.id, plain.semantic_contract
    ) != capability_fingerprint("skill", described.id, described.semantic_contract)
    assert SkillDefinition.from_semantic_contract(described.semantic_contract) == described

    agent = AgentSpec("agent")
    described_agent = AgentSpec("agent", description="Worker")
    assert AgentSpecCodec().to_payload(agent) == AgentSpecCodec().to_payload(described_agent)
    assert AgentSpecCodec().to_wire_payload(described_agent)["description"] == "Worker"
    plain_definition = _compiler({"agent": agent}).compile(agent)
    described_definition = _compiler({"agent": described_agent}).compile(described_agent)
    assert plain_definition.digest == described_definition.digest


def test_parent_binding_v1_keeps_logical_subagent_refs_and_round_trips() -> None:
    parent = AgentSpec("parent", allow_subagents=("child",))
    child = AgentSpec("child", allow_subagents=(), description="Child worker")
    compiler = _compiler({"parent": parent, "child": child})
    binding = compiler.bind(compiler.compile(parent))

    assert binding.snapshot.version == 1
    assert binding.snapshot.selected_subagents == ("child",)
    assert binding.snapshot.subagents[0].to_payload() == {
        "kind": "agent",
        "id": "child",
        "description": "Child worker",
    }

    payload = binding.snapshot.to_payload()
    assert "selected_subagents" not in payload
    payload["future_metadata"] = {"display": "ignored"}
    restored = compiler.restore(AgentBindingSnapshot.from_payload(payload))
    assert restored.digest == binding.digest
    assert restored.snapshot.subagents == binding.snapshot.subagents
    assert restored.snapshot.to_payload()["future_metadata"] == {"display": "ignored"}


def test_future_semantic_pin_version_is_not_misclassified_as_corruption() -> None:
    with pytest.raises(AIError) as error:
        SemanticPin(
            "skill",
            "review",
            2,
            {
                "version": 2,
                "id": "review",
                "content": "instructions",
            },
        )
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.asyncio
async def test_task_prepare_preserves_future_binding_version_error() -> None:
    binding = _load_json("agent_binding_subagent_v1_golden.json")
    binding["version"] = 2
    node = SimpleNamespace(
        input={
            "type": "linktools.ai.agent",
            "version": 1,
            "binding": binding,
            "user_prompt": "work",
            "user_prompt_codec": "text",
            "mode": "run",
            "planning": False,
            "thinking": False,
        },
        dependencies=(),
    )
    runner = object.__new__(RuntimeTaskNodeRunner)
    runner._catalog = None
    runner._compiler = None

    with pytest.raises(AIError) as error:
        await runner.prepare(
            node,  # type: ignore[arg-type]
            graph_id="graph",
            principal=Principal("principal", "tenant", "service"),
            dependency_results={},
        )

    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_skill_markdown_preserves_description_and_rejects_mismatch() -> None:
    content = "---\nname: review\ndescription: Review changes\n---\n\nDo the review.\n"
    codec = SkillMarkdownSpecCodec()
    decoded = codec.decode(content.encode("utf-8"))
    assert decoded.description == "Review changes"
    assert codec.encode(decoded) == content.encode("utf-8")

    with pytest.raises(AIError) as error:
        codec.encode(SkillSpec("review", content, "Different description"))
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


class _CountingSkillSource:
    def __init__(self) -> None:
        self.inspect_calls = 0
        self.read_calls: list[tuple[str, str]] = []

    @property
    def id(self) -> str:
        return "counting"

    async def inspect(self, root: str) -> SkillResourceView:
        self.inspect_calls += 1
        assert root == "review"
        return SkillResourceView(
            SkillLocation("virtual", "counting/skills/review"),
            ("references/rules.md",),
        )

    async def read(self, root: str, path: str) -> bytes:
        self.read_calls.append((root, path))
        return b"resource text"


@pytest.mark.asyncio
async def test_skill_function_calls_are_progressive_and_use_pinned_instructions() -> None:
    source = _CountingSkillSource()
    capability = SkillCapability(
        (
            SkillDefinition(
                SkillSpec("review", "pinned instructions", "Review changes"),
                SkillSourceRef("counting", "review"),
            ),
        ),
        SkillSourceRegistry((source,)),
    )

    assert await capability.list_skills() == [
        {"id": "review", "description": "Review changes"}
    ]
    assert source.inspect_calls == 0
    assert source.read_calls == []

    root = await capability.load_skill("review")
    assert root["instructions"] == "pinned instructions"
    assert root["resources"] == ["references/rules.md"]
    assert root["location"] == "virtual:counting/skills/review"
    assert "load_skill(skill_id, path)" in str(root["usage_hint"])
    assert source.inspect_calls == 1
    assert source.read_calls == []

    resource = await capability.load_skill("review", "references/rules.md")
    assert resource == {
        "id": "review",
        "path": "references/rules.md",
        "content": "resource text",
    }
    assert source.read_calls == [("review", "references/rules.md")]


@pytest.mark.parametrize(
    "path",
    (
        "/absolute/path",
        "../x",
        "foo/../x",
        "./x",
        "foo//bar",
        "foo\\bar",
        "C:\\x",
        "file://x",
        "virtual:x",
        "x\x00y",
    ),
)
def test_skill_resource_path_validation_rejects_non_relative_paths(path: str) -> None:
    with pytest.raises(AIError) as error:
        normalize_skill_resource_path(path)
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_skill_resource_path_validation_accepts_normalized_relative_paths() -> None:
    assert normalize_skill_resource_path("references/a.md") == "references/a.md"
    assert normalize_skill_resource_path("scripts/check.py") == "scripts/check.py"
    assert normalize_skill_resource_path("assets/templates/a.json") == "assets/templates/a.json"


@pytest.mark.asyncio
async def test_local_skill_source_reports_absolute_location_and_blocks_symlink_escape(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    package = skills_root / "review"
    (package / "references").mkdir(parents=True)
    (package / "scripts").mkdir()
    (package / "SKILL.md").write_text("declaration", encoding="utf-8")
    (package / "references" / "rules.md").write_text("rules", encoding="utf-8")
    (package / "scripts" / "check.py").write_text("print('ok')", encoding="utf-8")

    source = LocalSkillResourceSource("local", skills_root)
    view = await source.inspect("review")
    assert view.location.kind == "local"
    assert Path(view.location.path).is_absolute()
    assert Path(view.location.path) == package.resolve()
    assert view.resources == ("references/rules.md", "scripts/check.py")
    assert await source.read("review", "references/rules.md") == b"rules"

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = package / "references" / "outside-link"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is not supported")
    view = await source.inspect("review")
    assert "references/outside-link" not in view.resources
    with pytest.raises(AIError) as error:
        await source.read("review", "references/outside-link")
    assert error.value.code is ErrorCode.ASSET_PATH_OUTSIDE_ROOT


async def _asset_store() -> AssetStore:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    return store


@pytest.mark.asyncio
async def test_virtual_skill_source_uses_virtual_location_and_targeted_read() -> None:
    store = await _asset_store()
    await store.put(AssetKey("skill", "review/SKILL.md"), b"ignored declaration")
    await store.put(AssetKey("skill", "review/references/rules.md"), b"rules")
    source = AssetSkillResourceSource("workspace", store)
    capability = SkillCapability(
        (
            SkillDefinition(
                SkillSpec("review", "pinned", "Review changes"),
                SkillSourceRef("workspace", "review"),
            ),
        ),
        SkillSourceRegistry((source,)),
    )

    root = await capability.load_skill("review")
    assert root["location"] == "virtual:workspace/skills/review"
    assert root["resources"] == ["references/rules.md"]
    assert "not operating-system paths" in str(root["usage_hint"])
    assert await capability.load_skill("review", "references/rules.md") == {
        "id": "review",
        "path": "references/rules.md",
        "content": "rules",
    }


@pytest.mark.asyncio
async def test_skill_source_missing_and_binary_resource_fail_with_stable_codes() -> None:
    missing = SkillCapability(
        (
            SkillDefinition(
                SkillSpec("review", "pinned"),
                SkillSourceRef("missing", "review"),
            ),
        ),
        SkillSourceRegistry(),
    )
    with pytest.raises(AIError) as missing_error:
        await missing.load_skill("review")
    assert missing_error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY

    store = await _asset_store()
    await store.put(AssetKey("skill", "review/data.bin"), b"\xff\xfe")
    binary = SkillCapability(
        (
            SkillDefinition(
                SkillSpec("review", "pinned"),
                SkillSourceRef("workspace", "review"),
            ),
        ),
        SkillSourceRegistry((AssetSkillResourceSource("workspace", store),)),
    )
    with pytest.raises(AIError) as binary_error:
        await binary.load_skill("review", "data.bin")
    assert binary_error.value.code is ErrorCode.ASSET_CODEC_UNKNOWN


@pytest.mark.asyncio
async def test_default_workspace_declaration_scan_does_not_read_skill_resource_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = Workspace.load(tmp_path)
    package = workspace.storage_root / "skills" / "review"
    (package / "references").mkdir(parents=True)
    (package / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review changes\n---\n\nReview.\n",
        encoding="utf-8",
    )
    resource = package / "references" / "large.txt"
    resource.write_text("resource", encoding="utf-8")

    import linktools.ai.asset._directory as directory_module

    original = directory_module.read_bytes
    reads: list[Path] = []

    def capture(path: Path) -> bytes:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(directory_module, "read_bytes", capture)
    store, backend = _default_workspace_store(workspace)
    await store.initialize()
    try:
        group = CapabilityGroup.from_store(
            "workspace",
            store,
            skill_source=LocalSkillResourceSource(
                "workspace",
                workspace.storage_root / "skills",
            ),
        )
        frozen = await group.freeze()
        assert [(item.kind, item.id) for item in frozen] == [("skill", "review")]
        assert resource not in reads
        assert package / "SKILL.md" in reads
    finally:
        await store.close()
        await backend.close()


class _CountingAssetStore(AssetStore):
    def __init__(self, backend: InMemoryAssetBackend) -> None:
        super().__init__(StorageOverlay(backend, writer=backend))
        self.read_keys: list[AssetKey] = []

    async def get(self, key: AssetKey) -> "bytes | None":
        self.read_keys.append(key)
        return await super().get(key)


@pytest.mark.asyncio
async def test_builtin_loader_reads_skill_declaration_but_not_package_resources() -> None:
    backend = InMemoryAssetBackend()
    store = _CountingAssetStore(backend)
    await store.initialize()
    await store.put(
        AssetKey("skill", "review/SKILL.md"),
        b"---\nname: review\ndescription: Review changes\n---\n\nReview.\n",
    )
    await store.put(AssetKey("skill", "review/references/rules.md"), b"rules")

    frozen = await CapabilityGroup.from_store("workspace", store).freeze()

    assert [(item.kind, item.id) for item in frozen] == [("skill", "review")]
    assert store.read_keys == [AssetKey("skill", "review/SKILL.md")]


@pytest.mark.asyncio
async def test_builtin_loader_rejects_overlapping_skill_package_roots() -> None:
    store = await _asset_store()
    declaration = b"---\nname: parent\ndescription: Parent\n---\n\nParent.\n"
    child = b"---\nname: child\ndescription: Child\n---\n\nChild.\n"
    await store.put(AssetKey("skill", "parent/SKILL.md"), declaration)
    await store.put(AssetKey("skill", "parent/child/SKILL.md"), child)

    with pytest.raises(AIError) as error:
        await CapabilityGroup.from_store("workspace", store).freeze()
    assert error.value.code is ErrorCode.ASSET_LAYOUT_CONFLICT


class _CaptureExecution:
    def __init__(self) -> None:
        self.binding_digest: str | None = None
        self.request: ExecutionRequest | None = None
        self.start_calls = 0
        self.replay_calls = 0
        self.replay_handle: ExecutionHandle | None = None

    async def replay_subagent(
        self,
        *,
        agent_id: str,
        user_prompt: str,
        principal: Principal,
        idempotency_key: str,
        memory_scope: "str | None",
        mode: str,
        parent_execution_id: str,
        root_execution_id: str,
    ) -> "ExecutionHandle | None":
        assert agent_id == "child"
        assert user_prompt
        assert principal == Principal("principal", "tenant", "service")
        assert idempotency_key.startswith("subagent:")
        assert parent_execution_id == "parent-execution"
        assert root_execution_id == "root-execution"
        assert mode in {"run", "plan"}
        self.replay_calls += 1
        return self.replay_handle

    async def start_subagent(
        self,
        binding_digest: str,
        request: ExecutionRequest,
        *,
        parent_execution_id: str,
        root_execution_id: str,
    ) -> ExecutionHandle:
        assert parent_execution_id == "parent-execution"
        assert root_execution_id == "root-execution"
        self.binding_digest = binding_digest
        self.request = request
        self.start_calls += 1
        return ExecutionHandle("child-execution")

    async def wait(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult:
        assert execution_id == "child-execution"
        assert principal == Principal("principal", "tenant", "service")
        return ExecutionResult(
            execution_id,
            ExecutionStatus.SUCCEEDED,
            {"ok": True},
            "f" * 64,
            UsageMetrics(),
        )


def _subagent_compilers() -> tuple[
    AgentCompiler,
    AgentCompiler,
    AgentSpec,
    AgentSpec,
    AgentSpec,
]:
    parent = AgentSpec("parent", allow_subagents=("child",))
    limits = AgentUsageLimits(model_requests=3)
    child_old = AgentSpec(
        "child",
        model="child-old",
        system_prompt="old child",
        allow_subagents=(),
        usage_limits=limits,
        planning=True,
        thinking="high",
        description="Old child",
    )
    child_new = AgentSpec(
        "child",
        model="child-new",
        system_prompt="new child",
        allow_subagents=(),
        usage_limits=AgentUsageLimits(model_requests=9),
        planning=False,
        thinking=False,
        description="New child",
    )
    old_registry = ModelRegistry.openai(model="gpt-test")
    old_registry.register_openai("child-old", model="gpt-child-old")
    new_registry = ModelRegistry.openai(model="gpt-test")
    new_registry.register_openai("child-new", model="gpt-child-new")
    return (
        _compiler({"parent": parent, "child": child_old}, old_registry),
        _compiler({"parent": parent, "child": child_new}, new_registry),
        parent,
        child_old,
        child_new,
    )


def test_child_description_changes_parent_binding_identity_without_pinning_definition() -> None:
    old_compiler, new_compiler, parent, _child_old, _child_new = _subagent_compilers()
    old_binding = old_compiler.bind(old_compiler.compile(parent))
    new_binding = new_compiler.bind(new_compiler.compile(parent))

    assert old_binding.snapshot.version == 1
    assert old_binding.snapshot.subagents[0].to_payload() == {
        "kind": "agent",
        "id": "child",
        "description": "Old child",
    }
    assert new_binding.snapshot.subagents[0].description == "New child"
    assert old_binding.digest != new_binding.digest
    assert old_binding.definition.digest == new_binding.definition.digest


@pytest.mark.asyncio
async def test_restored_parent_resolves_current_child_only_when_delegated() -> None:
    old_compiler, new_compiler, parent, _child_old, child_new = _subagent_compilers()
    old_binding = old_compiler.bind(old_compiler.compile(parent))
    restored = new_compiler.restore(old_binding.snapshot)
    new_child_definition = new_compiler.compile(child_new)
    catalog = AgentCatalog(
        {
            "parent": new_compiler.compile(parent),
            "child": new_child_definition,
        }
    )
    execution = _CaptureExecution()
    dispatcher = SubagentDispatcher(catalog, new_compiler, execution)  # type: ignore[arg-type]
    assert dispatcher.descriptions_for(restored.snapshot.subagents) == {"child": "Old child"}

    result = await dispatcher.dispatch(
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
        memory_scope="memory",
        principal=Principal("principal", "tenant", "service"),
        ref=restored.snapshot.subagents[0],
        mode="run",
        user_prompt="do work",
        invocation_id="call-1",
    )

    assert result["status"] == ExecutionStatus.SUCCEEDED.value
    assert execution.replay_calls == 1
    assert execution.start_calls == 1
    assert execution.binding_digest is not None
    child_binding = catalog.binding(execution.binding_digest)
    assert child_binding.definition.digest == new_child_definition.digest
    assert child_binding.snapshot.subagents == ()
    assert child_binding.definition.spec.usage_limits == AgentUsageLimits(model_requests=9)


@pytest.mark.asyncio
async def test_missing_current_child_does_not_block_parent_restore() -> None:
    old_compiler, _new_compiler, parent, _child_old, _child_new = _subagent_compilers()
    old_binding = old_compiler.bind(old_compiler.compile(parent))
    parent_only_compiler = _compiler({"parent": parent})
    restored = parent_only_compiler.restore(old_binding.snapshot)
    catalog = AgentCatalog({"parent": restored.definition})
    execution = _CaptureExecution()
    dispatcher = SubagentDispatcher(catalog, parent_only_compiler, execution)  # type: ignore[arg-type]

    assert dispatcher.descriptions_for(restored.snapshot.subagents) == {"child": "Old child"}
    with pytest.raises(AIError) as error:
        await dispatcher.dispatch(
            parent_execution_id="parent-execution",
            root_execution_id="root-execution",
            memory_scope=None,
            principal=Principal("principal", "tenant", "service"),
            ref=restored.snapshot.subagents[0],
            mode="run",
            user_prompt="do work",
            invocation_id="call-missing",
        )
    assert error.value.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE
    assert execution.replay_calls == 1
    assert execution.start_calls == 0


@pytest.mark.asyncio
async def test_existing_child_replay_does_not_require_current_child_definition() -> None:
    old_compiler, _new_compiler, parent, _child_old, _child_new = _subagent_compilers()
    parent_binding = old_compiler.bind(old_compiler.compile(parent))
    parent_only_compiler = _compiler({"parent": parent})
    restored = parent_only_compiler.restore(parent_binding.snapshot)
    catalog = AgentCatalog({"parent": restored.definition})
    execution = _CaptureExecution()
    execution.replay_handle = ExecutionHandle("child-execution")
    dispatcher = SubagentDispatcher(catalog, parent_only_compiler, execution)  # type: ignore[arg-type]

    result = await dispatcher.dispatch(
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
        memory_scope="memory",
        principal=Principal("principal", "tenant", "service"),
        ref=restored.snapshot.subagents[0],
        mode="run",
        user_prompt="do work",
        invocation_id="call-replay",
    )

    assert result["status"] == ExecutionStatus.SUCCEEDED.value
    assert execution.replay_calls == 1
    assert execution.start_calls == 0


@pytest.mark.asyncio
async def test_execution_service_replays_subagent_from_persisted_child_binding() -> None:
    service = object.__new__(DefaultExecutionService)
    persisted = SimpleNamespace(
        resource_id="child-execution",
    )
    execution = SimpleNamespace(
        binding_digest="a" * 64,
        binding=object(),
        lineage_kind=ExecutionLineageKind.SUBAGENT,
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
        planning=True,
        thinking="high",
    )
    service._state = SimpleNamespace(
        idempotency=SimpleNamespace(get=AsyncMock(return_value=persisted)),
        executions=SimpleNamespace(get=AsyncMock(return_value=execution)),
    )
    service._binding = lambda digest, snapshot: SimpleNamespace(  # type: ignore[method-assign]
        definition=SimpleNamespace(spec=SimpleNamespace(id="child"))
    )
    start_subagent = AsyncMock(return_value=ExecutionHandle("child-execution"))
    service.start_subagent = start_subagent  # type: ignore[method-assign]

    handle = await service.replay_subagent(
        agent_id="child",
        user_prompt="do work",
        principal=Principal("principal", "tenant", "service"),
        idempotency_key="subagent:" + "b" * 64,
        memory_scope="memory",
        mode="run",
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
    )

    assert handle == ExecutionHandle("child-execution")
    request = start_subagent.await_args.args[1]
    assert request.planning is True
    assert request.thinking == "high"
    assert start_subagent.await_args.args[0] == "a" * 64


@pytest.mark.asyncio
async def test_child_binding_keeps_declared_grandchildren_but_disables_nested_delegation() -> None:
    parent = AgentSpec("parent", allow_subagents=("child",))
    child = AgentSpec("child", allow_subagents=("grandchild",), description="Child")
    grandchild = AgentSpec("grandchild", allow_subagents=())
    source_compiler = _compiler(
        {"parent": parent, "child": child, "grandchild": grandchild}
    )
    parent_binding = source_compiler.bind(source_compiler.compile(parent))
    child_definition = source_compiler.compile(child)

    current_compiler = _compiler(
        {"parent": parent, "child": AgentSpec("child", allow_subagents=())}
    )
    catalog = AgentCatalog(
        {
            "parent": current_compiler.compile(parent),
            "child": child_definition,
        }
    )
    execution = _CaptureExecution()
    dispatcher = SubagentDispatcher(catalog, current_compiler, execution)  # type: ignore[arg-type]

    result = await dispatcher.dispatch(
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
        memory_scope=None,
        principal=Principal("principal", "tenant", "service"),
        ref=parent_binding.snapshot.subagents[0],
        mode="run",
        user_prompt="do work",
        invocation_id="call-grandchild",
    )

    assert result["status"] == ExecutionStatus.SUCCEEDED.value
    assert execution.binding_digest is not None
    child_binding = catalog.binding(execution.binding_digest)
    assert child_binding.definition.digest == child_definition.digest
    assert child_binding.definition.selected_subagents == ("grandchild",)
    assert child_binding.snapshot.selected_subagents == ("grandchild",)
    assert child_binding.snapshot.subagents == ()
    assert child_binding.snapshot.to_payload()["selected_subagents"] == ["grandchild"]

    restored = current_compiler.restore(child_binding.snapshot)
    assert restored.digest == child_binding.digest
    assert restored.definition.digest == child_definition.digest


@pytest.mark.asyncio
async def test_parent_plan_mode_is_security_ceiling_but_keeps_child_thinking_default() -> None:
    old_compiler, _new_compiler, parent, child_old, _child_new = _subagent_compilers()
    parent_binding = old_compiler.bind(old_compiler.compile(parent))
    child_definition = old_compiler.compile(child_old)
    catalog = AgentCatalog({"parent": old_compiler.compile(parent), "child": child_definition})
    execution = _CaptureExecution()
    dispatcher = SubagentDispatcher(catalog, old_compiler, execution)  # type: ignore[arg-type]

    await dispatcher.dispatch(
        parent_execution_id="parent-execution",
        root_execution_id="root-execution",
        memory_scope=None,
        principal=Principal("principal", "tenant", "service"),
        ref=parent_binding.snapshot.subagents[0],
        mode="plan",
        user_prompt="plan work",
        invocation_id="call-3",
    )

    assert execution.request is not None
    assert execution.request.mode == "plan"
    assert execution.request.planning is True
    assert execution.request.thinking == "high"