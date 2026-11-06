#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Skill reads refresh within a run without widening selectors or workspace mounts."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai import Runtime
from linktools.ai.agent import AgentBindingContract, AgentCatalog, AgentCompiler
from linktools.ai.asset import AssetKey, AssetStore, DirectoryAssetBackend, InMemoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.capability import AssetCapability, AssetSkillSource, CapabilityContribution, ExecutionCapability, SkillCapability, SkillDefinition, SkillResource, SkillSourceRef, SkillSourceRegistry
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.spec import AgentSpec, MCPServerSpec, RepositoryInstructionDocument, SkillSpec
from linktools.ai.storage import StorageLayer, StorageOverlay

from ._runtime_test_helpers import _Models, _UsageFunctionModel, native_checkpoint


def _declaration(identity: str, text: str, *, audience: str = "analyst") -> bytes:
    return f"---\nname: {identity}\ndescription: {text}\nmetadata:\n  audience: {audience}\n---\n{text}".encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("selectors", [("*",), ("guide",)])
async def test_runtime_discovers_and_reloads_skills_in_one_native_run(tmp_path, selectors):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    stage = 0
    old_observation = None

    async def respond(messages, info):
        nonlocal stage, old_observation
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        names = {tool.name for tool in info.function_tools}
        assert {"list_skills", "load_skill"}.issubset(names)
        if stage == 0:
            stage += 1
            return ModelResponse(parts=[ToolCallPart("list_skills", {}, "list-empty")])
        if stage == 1:
            assert returns[-1].content == []
            await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "First"))
            await store.put(AssetKey("skill", "hidden/SKILL.md"), _declaration("hidden", "Secret", audience="other"))
            stage += 1
            return ModelResponse(parts=[ToolCallPart("list_skills", {}, "list-new")])
        if stage == 2:
            assert returns[-1].content == [{"id": "guide", "description": "First"}]
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "load-old")])
        if stage == 3:
            old_observation = returns[-1].content
            assert old_observation["instructions"].endswith("First")
            assert old_observation["location"].startswith("virtual:")
            await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "Second"))
            await store.put(AssetKey("skill", "guide/new.txt"), b"new resource")
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "load-new")])
        if stage == 4:
            assert returns[-1].content["instructions"].endswith("Second")
            assert returns[-1].content["resources"] == ["new.txt"]
            assert next(part.content for part in returns if part.tool_call_id == "load-old") == old_observation
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide", "path": "new.txt"}, "resource")])
        if stage == 5:
            assert returns[-1].content["content"] == "new resource"
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "hidden"}, "hidden")])
        assert returns[-1].tool_call_id == "hidden"
        assert returns[-1].outcome == "retried"
        assert "Secret" not in str(returns[-1].content)
        stage += 1
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures(
        "live", workspace_factory=None,
        filter=lambda agent, item: item.kind != "skill" or item.value.spec.metadata.get("audience") == agent.metadata.get("role"),
    )
    features.capability(AssetCapability(store))
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.agent("default", allow_tools=(), allow_skills=selectors, metadata={"role": "analyst"})
    async with Persistence.open("live", storage=PersistenceStorage.sqlite(tmp_path / "live.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            assert runtime.agents.get().derive(model="default").compiled.skill_definitions == ()
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("Use the new skill", idempotency_key="input")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
    assert stage == 7


@pytest.mark.asyncio
async def test_new_resource_membership_does_not_advertise_an_existing_mount():
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "First"))
        await store.put(AssetKey("skill", "guide/old.txt"), b"old")
        features = RuntimeFeatures("live")
        features.capability(AssetCapability(store))

        async def discover():
            return tuple(item.value for item in (await features.resolve()).contributions if item.kind == "skill")

        source = AssetSkillSource("live", store)
        definitions = await discover()
        capability = SkillCapability(
            definitions, SkillSourceRegistry((source,)), discover=discover,
            resource_paths={"guide": "/sandbox/guide"}, mounted_resources={"guide": definitions[0].source_ref},
        )
        assert (await capability.load_skill("guide"))["location"] == "/sandbox/guide"
        await store.put(AssetKey("skill", "guide/new.txt"), b"new")
        assert (await capability.load_skill("guide"))["location"].startswith("virtual:")
        assert (await capability.load_skill("guide", "new.txt"))["content"] == "new"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_derived_skill_permissions_compare_selectors_without_catalog_membership(tmp_path):
    features = RuntimeFeatures("derive")
    features.model("default", _Models(_UsageFunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")]))))
    features.agent("default", allow_skills=("future",))
    async with Persistence.open("derive", storage=PersistenceStorage.sqlite(tmp_path / "derive.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            root = runtime.agents.get()
            assert root.derive(allow_skills=("future",)).compiled.skill_definitions == ()
            assert root.derive(allow_skills=()).compiled.skill_definitions == ()
            for selectors in (("*",), ("other",)):
                with pytest.raises(AIError) as error:
                    root.derive(allow_skills=selectors)
                assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_binding_restores_admitted_filter_metadata_without_changing_digest():
    compiler = AgentCompiler(models={"default": _Models(_UsageFunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")])))}, candidates=(), agents={})
    compiled = compiler.compile(AgentSpec("root", metadata={"role": "analyst"}))
    binding = compiler.bind(compiled)
    restored = compiler.restore(AgentBindingContract.from_payload(binding.binding_contract.to_payload()))
    assert restored.compiled_agent.spec.metadata == {"role": "analyst"}
    assert restored.binding_digest == binding.binding_digest


@pytest.mark.asyncio
async def test_next_run_materializes_current_local_skill_resources(tmp_path):
    assets = tmp_path / "assets"
    package = assets / "skills" / "guide"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_bytes(_declaration("guide", "Instructions"))
    resource = package / "guide.txt"
    resource.write_text("First")
    store = AssetStore(StorageOverlay(DirectoryAssetBackend(
        str(assets), path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",),
    )))
    expected = "First"

    async def respond(messages, info):
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide", "path": "guide.txt"}, "read")])
        assert returns[-1].content["content"] == expected
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("current-resources")
    features.capability(AssetCapability(store))
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.agent("default", allow_tools=(), allow_skills=("guide",))
    async with Persistence.open("current-resources", storage=PersistenceStorage.sqlite(tmp_path / "resources.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            for expected in ("First", "Second"):
                resource.write_text(expected)
                session = await runtime.agents.get().create_session(expected)
                execution = await session.start("Read the skill", idempotency_key=expected)
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.SUCCEEDED, result


@pytest.mark.asyncio
async def test_recovery_keeps_committed_skill_observation_and_reads_current_with_admitted_metadata(tmp_path):
    assets = tmp_path / "assets"
    package = assets / "skills" / "guide"
    package.mkdir(parents=True)
    declaration = package / "SKILL.md"
    declaration.write_bytes(_declaration("guide", "Original"))
    blocked = asyncio.Event()
    old_observation = None

    @asynccontextmanager
    async def open_runtime(model, role):
        store = AssetStore(StorageOverlay(DirectoryAssetBackend(
            str(assets), path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",),
        )))
        features = RuntimeFeatures(
            "skill-recovery", workspace_factory=None,
            filter=lambda agent, item: item.kind != "skill" or item.value.spec.metadata.get("audience") == agent.metadata.get("role"),
        )
        features.capability(AssetCapability(store))
        features.capability(ExecutionCapability(retain_history=True))
        checkpoints = native_checkpoint(database=tmp_path / "steps.db")
        features.capability(checkpoints)
        features.model("default", _Models(_UsageFunctionModel(model)))
        features.agent("default", allow_tools=(), allow_skills=("*",), metadata={"role": role})
        async with Persistence.open("skill-recovery", storage=PersistenceStorage.sqlite(tmp_path / "recovery.db")) as persistence:
            async with Runtime.attach(persistence=persistence, features=features, auto_recover=False) as runtime:
                yield runtime, persistence, checkpoints.store

    async def interrupted(messages, info):
        nonlocal old_observation
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            old_observation = returns[-1].content
            assert old_observation["instructions"].endswith("Original")
            blocked.set()
            await asyncio.Event().wait()
        return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "original")])

    async with open_runtime(interrupted, "analyst") as (runtime, persistence, steps):
        execution = await runtime.agents.get().start("Read the skill", idempotency_key="start")
        await asyncio.wait_for(blocked.wait(), 10)
        await asyncio.wait_for(runtime.handoff(), 10)
        source = await persistence.execution.executions.get_recovery(execution.execution_id, tenant_id="default")
        assert source is not None and source.recovery is not None
        checkpoint = await steps.latest_snapshot(run_id=source.recovery.agent_run_id, include_interrupted=True)
        assert checkpoint is not None
        assert any(
            isinstance(part, ToolReturnPart) and part.content == old_observation
            for message in checkpoint.messages for part in message.parts
        )

    declaration.write_bytes(_declaration("guide", "Current"))
    calls = 0

    async def recovered(messages, info):
        nonlocal calls
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        assert next(part.content for part in returns if part.tool_call_id == "original") == old_observation
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "current")])
        assert returns[-1].content["instructions"].endswith("Current")
        return ModelResponse(parts=[TextPart("recovered")])

    async with open_runtime(recovered, "other") as (runtime, persistence, steps):
        await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal, idempotency_key="recover")
        outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=10)
        assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", (False, True))
async def test_custom_skill_loader_keeps_ownership_and_resolves_current_slot(tmp_path, wrapped):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    key = AssetKey("skill", "custom.txt")
    await store.put(key, b"First")
    other_resource = AssetKey("worker", "other/guide.txt")
    await store.put(other_resource, b"Other resource")

    class Loader:
        async def load(self, context):
            assets.loader("skill", ReplacedLoader())
            assert await context.read(other_resource) == b"Other resource"
            if not context.list(kind="skill"):
                return ()
            reference, = context.bind_versions((key,))
            definition = SkillDefinition(
                SkillSpec("custom", (await context.read(key)).decode()),
                SkillSourceRef(context.group_id, "custom", (SkillResource("custom.txt", reference),)),
            )
            return (
                CapabilityContribution.from_declaration(definition) if wrapped else definition,
                AgentSpec("extra", allow_tools=()),
                MCPServerSpec("fixture", "python"),
                RepositoryInstructionDocument("rule:mixed-loader", ".", "Captured rule"),
                CapabilityContribution.from_declaration(AgentSpec("wrapped-extra", allow_tools=())),
            )

    class OtherLoader:
        async def load(self, context):
            reference, = context.bind_versions((other_resource,))
            return (SkillDefinition(SkillSpec("other", "Other loader"), SkillSourceRef(
                context.group_id, "other", (SkillResource("guide.txt", reference),),
            )),)

    replacement_calls = []

    class ReplacedLoader(Loader):
        async def load(self, context):
            replacement_calls.append(context.group_id)
            return await super().load(context)

    assets = AssetCapability(store)
    assets.loader("skill", Loader())
    assets.loader("other", OtherLoader())
    stage = 0

    async def respond(messages, info):
        nonlocal stage
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if stage == 0:
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "custom"}, "first")])
        if stage == 1:
            assert returns[-1].content["instructions"] == "First"
            await store.put(key, b"Second")
            stage += 1
            return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "custom"}, "second")])
        if stage == 2:
            assert returns[-1].content["instructions"] == "Second"
            await store.delete(key)
            stage += 1
            return ModelResponse(parts=[ToolCallPart("list_skills", {}, "remaining")])
        assert returns[-1].content == []
        stage += 1
        return ModelResponse(parts=[TextPart("done")])

    visible_static = True
    features = RuntimeFeatures(
        "custom-loader", workspace_factory=None,
        filter=lambda agent, item: item.kind != "skill" or item.id == "custom" or visible_static,
    )
    features.capability(assets)
    features.model("default", _Models(_UsageFunctionModel(respond)))
    features.agent("default", allow_tools=())
    capture = await features.resolve()
    custom = next(item.value for item in capture.contributions if item.kind == "skill" and item.id == "custom")
    other = next(item.value for item in capture.contributions if item.kind == "skill" and item.id == "other")
    assert custom.source_ref is not None and custom.source_ref.source_id == "custom-loader"
    assert other.source_ref is not None and other.source_ref.source_id == "custom-loader"
    custom_contribution = next(item for item in capture.contributions if item.kind == "skill" and item.id == "custom")
    assert SkillDefinition.from_contract(custom_contribution.contract).source_ref == custom.source_ref
    features._contributions.append(CapabilityContribution.from_declaration(SkillDefinition(SkillSpec("inline", "Inline"))))
    assets.loader("skill", ReplacedLoader())
    try:
        async with Persistence.open("custom-loader", storage=PersistenceStorage.sqlite(tmp_path / "custom.db")) as persistence:
            async with Runtime.attach(persistence=persistence, features=features.borrow()) as runtime:
                assert {skill.id for skill in runtime.agents.get().derive(model="default").compiled.skill_definitions} == {"custom", "inline", "other"}
                visible_static = False
                execution = await runtime.agents.get().start("Use custom", idempotency_key="start")
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.SUCCEEDED, result
        assert stage == 4
        assert replacement_calls
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ("old script", "new script"))
async def test_same_named_resource_switching_overlay_layer_does_not_advertise_old_mount(tmp_path, content):
    primary = tmp_path / "primary"
    fallback = tmp_path / "fallback"
    for root in (primary, fallback):
        (root / "skill" / "guide").mkdir(parents=True)
    (fallback / "skill" / "guide" / "SKILL.md").write_bytes(_declaration("guide", "Instructions"))
    (fallback / "skill" / "guide" / "run.sh").write_text("old script")
    store = AssetStore(StorageOverlay(
        DirectoryAssetBackend(str(primary), kinds=("skill",)),
        layers=(StorageLayer("fallback", DirectoryAssetBackend(str(fallback), kinds=("skill",))),),
    ))
    await store.initialize()
    try:
        features = RuntimeFeatures("overlay")
        features.capability(AssetCapability(store))

        async def discover():
            return tuple(item.value for item in (await features.resolve()).contributions if item.kind == "skill")

        source = AssetSkillSource("overlay", store)
        definitions = await discover()
        capability = SkillCapability(
            definitions, SkillSourceRegistry((source,)), discover=discover,
            resource_paths={"guide": "/sandbox/old-guide"},
            mounted_resources={"guide": definitions[0].source_ref},
        )
        assert (await capability.load_skill("guide"))["location"] == "/sandbox/old-guide"
        (primary / "skill" / "guide" / "run.sh").write_text(content)
        result = await capability.load_skill("guide")
        assert result["resources"] == ["run.sh"]
        assert result["location"].startswith("virtual:")
        assert (await capability.load_skill("guide", "run.sh"))["content"] == content
    finally:
        await store.close()


def test_catalog_does_not_reuse_binding_with_different_admitted_filter_metadata():
    compiler = AgentCompiler(models={"default": _Models(_UsageFunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("done")])))}, candidates=(), agents={})
    first = compiler.bind(compiler.compile(AgentSpec("root", metadata={"role": "analyst"})))
    second = compiler.bind(compiler.compile(AgentSpec("root", metadata={"role": "other"})))
    assert first.binding_digest == second.binding_digest
    catalog = AgentCatalog({"root": first.compiled_agent})
    assert catalog.register_binding(first) is first
    assert catalog.register_binding(second) is second
    assert second.binding_contract.agent_spec.metadata == {"role": "other"}
