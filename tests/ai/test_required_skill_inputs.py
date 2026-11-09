#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Required rules are fixed execution inputs; actual Skill reads retain provenance."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.asset import AssetKey, AssetStore, DirectoryAssetBackend, InMemoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.capability import (
    AssetCapability, ExecutionCapability, SkillDefinition,
    SkillInput, SkillReadReceipt, SkillResource, SkillSourceRef, TraceCapability,
)
from linktools.ai.core import AuthorizationAction, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import AgentSpec, AgentSpecAdapter, AgentSpecCodec, CapabilityRef, SkillSpec
from linktools.ai.storage import FilesystemObjectStore, StorageLayer, StorageOverlay

from ._runtime_test_helpers import _Models, _UsageFunctionModel, native_checkpoint


def _declaration(name: str, text: str) -> bytes:
    return f"---\nname: {name}\ndescription: {name} guide\n---\n{text}".encode()


class _InlineLoader:
    async def load(self, context):
        return (SkillDefinition(SkillSpec("inline", "INLINE_RULE")),)


def _features(store, model, *, required=("guide",), filtered=False, retained=True):
    features = RuntimeFeatures("required-skills", workspace_factory=None,
                               filter=(lambda agent, item: item.kind != "skill") if filtered else None)
    assets = features.capability(AssetCapability(store))
    assets.loader("inline", _InlineLoader())
    if retained:
        features.capability(ExecutionCapability(retain_history=True))
        features.capability(TraceCapability())
        features.capability(native_checkpoint())
    features.model("default", _Models(_UsageFunctionModel(model)))
    features.agent("default", allow_tools=(), allow_subagents=(), required_skills=required)
    return features


def test_required_skill_declaration_roundtrips_and_does_not_widen_allowlist():
    assert "required_skills" not in AgentSpecCodec().to_contract_payload(AgentSpec("unchanged"))
    spec = AgentSpec("agent", allow_skills=("guide",), required_skills=("guide",))
    assert AgentSpecCodec().decode(AgentSpecCodec().encode(spec)) == spec
    parsed = AgentSpecAdapter().decode_markdown(
        b"---\nrequired-skills: [guide]\nallow-skills: [guide]\n---\n", logical_id="agent")
    assert parsed.required_skills == ("guide",)
    for required, allowed in ((("*",), ("*",)), (("guide",), ())):
        with pytest.raises(AIError) as error:
            AgentSpec("agent", required_skills=required, allow_skills=allowed)
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_required_root_is_preloaded_while_resources_and_other_skills_stay_live(tmp_path: Path):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    root = AssetKey("skill", "guide/SKILL.md")
    resource = AssetKey("skill", "guide/resource.txt")
    await store.put(root, _declaration("guide", "ORIGINAL_RULE"))
    await store.put(resource, b"original resource")
    await store.put(AssetKey("skill", "optional/SKILL.md"), _declaration("optional", "ORIGINAL_OPTIONAL"))
    original_ref, = await store.resolve_versions((root,))
    calls = (("root", {"skill_id": "guide"}), ("resource", {"skill_id": "guide", "path": "resource.txt"}),
             ("optional", {"skill_id": "optional"}), ("unknown", {"skill_id": "inline"}),
             ("missing", {"skill_id": "guide", "path": "missing.txt"}))
    stage = 0

    async def model(messages, info):
        nonlocal stage
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if stage == 0:
            assert "ORIGINAL_RULE" in str((messages, info))
            await store.put(root, _declaration("guide", "CHANGED_RULE"))
            await store.put(resource, b"current resource")
            await store.put(AssetKey("skill", "optional/SKILL.md"), _declaration("optional", "CHANGED_OPTIONAL"))
        elif stage == 1:
            assert returns[-1].content["instructions"].endswith("ORIGINAL_RULE")
            assert returns[-1].content["resources"] == ["resource.txt"]
            assert returns[-1].content["resource_status"] == "current"
            assert "usage_hint" in returns[-1].content
        elif stage == 2:
            assert returns[-1].content["content"] == "current resource"
        elif stage == 3:
            assert returns[-1].content["instructions"].endswith("CHANGED_OPTIONAL")
        elif stage == 4:
            assert returns[-1].content["instructions"] == "INLINE_RULE"
        else:
            assert returns[-1].outcome == "failed"
            return ModelResponse(parts=[TextPart("done")])
        call_id, args = calls[stage]
        stage += 1
        return ModelResponse(parts=[ToolCallPart("load_skill", args, call_id)])

    features = _features(store, model)
    path = tmp_path / "runtime.db"
    async with Persistence.open("required-live", storage=PersistenceStorage.sqlite(path)) as persistence:
        async with Runtime.attach(persistence=persistence, features=features.borrow()) as runtime:
            execution = await runtime.agents.get().start("Use the required guide", idempotency_key="run")
            result = (await execution.wait(timeout_seconds=20)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
            principal = runtime.default_principal
            execution_id = execution.execution_id
            pages = {}
            for index, (call_id, _args) in enumerate(calls, 1):
                page = await runtime.history.history(execution_id, principal=principal,
                    tool_call_id=call_id, agent_run_seq=1, model_request_seq=index, include_content=True)
                pages[call_id] = page.items
                result_item, = (item for item in page.items if item.item_kind == "tool_result")
                assert result_item.tool_operation_id is None
                if call_id == "missing":
                    assert result_item.status == "FAILED" and result_item.skill_read is None
                else:
                    receipt = result_item.skill_read
                    assert isinstance(receipt, SkillReadReceipt)
                    assert result_item.status == "SUCCEEDED"
                    assert receipt.identity_verified is (call_id != "unknown")
                    if call_id == "root":
                        assert receipt.asset == original_ref
                        assert receipt.path == "SKILL.md" and receipt.source_id == "required-skills"
            requests = await runtime.history.read_model_interactions(execution_id, principal=principal,
                identities=((1, 1),), include_content=True)
            assert len(requests) == 1
            receipts = requests[0].request["required_skill_reads"]
            assert SkillReadReceipt.from_payload(receipts[0]).asset == original_ref
            assert "ORIGINAL_RULE" in str(requests[0].request["instructions"])
    await store.close()
    async with Persistence.open("required-live", storage=PersistenceStorage.sqlite(path), read_only=True) as persistence:
        async with History.open(persistence=persistence) as history:
            for call_id, _args in calls:
                page = await history.history(execution_id, principal=principal, tool_call_id=call_id, include_content=True)
                assert page.items == pages[call_id]
            requests = await history.read_model_interactions(execution_id, principal=principal, identities=((1, 1),), include_content=True)
            assert SkillReadReceipt.from_payload(requests[0].request["required_skill_reads"][0]).asset == original_ref


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("missing", "filtered", "directory", "unverifiable", "integrity"))
async def test_required_input_admission_fails_before_model_without_fixed_access(tmp_path: Path, failure: str):
    if failure == "directory":
        package = tmp_path / "assets" / "skills" / "guide"
        package.mkdir(parents=True)
        (package / "SKILL.md").write_bytes(_declaration("guide", "RULE"))
        store = AssetStore(StorageOverlay(DirectoryAssetBackend(str(tmp_path / "assets"),
            path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",))))
    else:
        class CorruptStore(AssetStore):
            async def require_retained_versions(self, refs):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        backend = InMemoryAssetBackend()
        store_class = CorruptStore if failure == "integrity" else AssetStore
        store = store_class(StorageOverlay(backend, writer=backend))
        await store.initialize()
        if failure != "missing":
            await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "RULE"))

    async def model(messages, info):
        raise AssertionError("required-input failure must precede the model")

    features = _features(store, model, required=("inline",) if failure == "unverifiable" else ("guide",), filtered=failure == "filtered", retained=failure != "missing")
    async with Persistence.open("required-denied", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            execution = await runtime.agents.get().start("go", idempotency_key="run")
            result = (await execution.wait(timeout_seconds=20)).result
            expected = ErrorCode.STORAGE_INTEGRITY_ERROR if failure == "integrity" else ErrorCode.CAPABILITY_REQUIRED_MISSING
            assert result.status is ExecutionStatus.FAILED and result.error_code == expected.value
            with pytest.raises(AIError) as replay:
                await runtime.agents.get().start("go", idempotency_key="run")
            assert replay.value.code is expected


@pytest.mark.asyncio
async def test_imported_empty_required_input_cannot_rebind_current_rules(tmp_path: Path):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "RULE"))

    async def model(messages, info):
        raise AssertionError("unbound required rules must not reach the model")

    features = _features(store, model)
    async with Persistence.open("required-import", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=features) as runtime:
            with pytest.raises(AIError) as error:
                await runtime.agents.get().start("go", idempotency_key="run",
                    input_context=ExecutionInputContext.from_messages(()))
            assert error.value.code is ErrorCode.BINDING_CONFLICT


@pytest.mark.asyncio
async def test_retention_checks_actual_layer_and_snapshot_survives_source_change(tmp_path: Path):
    package = tmp_path / "assets" / "skills" / "guide"
    package.mkdir(parents=True)
    declaration = package / "SKILL.md"
    declaration.write_bytes(_declaration("guide", "ORIGINAL_RULE"))
    directory = DirectoryAssetBackend(str(tmp_path / "assets"), path_adapter=PrefixAssetPathAdapter({"skill": "skills"}), kinds=("skill",))
    memory = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(memory, writer=memory, layers=(StorageLayer("directory", directory),)))
    await store.initialize()
    key = AssetKey("skill", "guide/SKILL.md")
    ref, = await store.resolve_versions((key,))
    assert ref.layer_id == "directory"
    with pytest.raises(AIError) as error:
        await store.require_retained_versions((ref,))
    assert error.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
    objects = FilesystemObjectStore(tmp_path / "asset-history", store_id="asset-history")
    snapshot = await store.snapshot((key,), object_store=objects)
    frozen = AssetStore.from_snapshot(snapshot, object_store=objects)
    await frozen.initialize()
    frozen_ref, = await frozen.resolve_versions((key,))
    await frozen.close()
    declaration.write_bytes(_declaration("guide", "CHANGED_RULE"))
    reopened = AssetStore.from_snapshot(snapshot, object_store=FilesystemObjectStore(tmp_path / "asset-history", store_id="asset-history"))
    await reopened.initialize()
    assert (await reopened.require_retained_versions((frozen_ref,)))[0].endswith(b"ORIGINAL_RULE")
    await reopened.close()
    await store.close()


@pytest.mark.asyncio
async def test_required_rules_recover_from_sql_history_after_source_changes(tmp_path: Path):
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.asset import SqlAssetBackend
    from linktools.ai.migrate import provision_asset_database

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    key = AssetKey("skill", "guide/SKILL.md")
    blocked = asyncio.Event()
    old_requests = None

    @asynccontextmanager
    async def open_runtime(model, *, initial=False):
        backend = SqlAssetBackend(engine, namespace="required-recovery")
        store = AssetStore(StorageOverlay(backend, writer=backend))
        await store.initialize()
        if initial:
            await store.put(key, _declaration("guide", "ORIGINAL_RULE"))
        features = _features(store, model)
        async with Persistence.open("required-recovery", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
            async with Runtime.attach(persistence=persistence, features=features.borrow(), auto_recover=False) as runtime:
                yield runtime, store
        await store.close()

    async def first(messages, info):
        assert "ORIGINAL_RULE" in str((messages, info))
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            assert returns[-1].content["resource_status"] == "current"
            assert returns[-1].content["resources"] == []
            blocked.set()
            await asyncio.Event().wait()
        return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "before")])

    try:
        async with open_runtime(first, initial=True) as (runtime, store):
            execution = await runtime.agents.get().start("go", idempotency_key="original")
            await asyncio.wait_for(blocked.wait(), 20)
            old_requests = await runtime.history.read_model_interactions(execution.execution_id,
                principal=runtime.default_principal, identities=((1, 1),), include_content=True)
            await asyncio.wait_for(runtime.handoff(), 20)
            # Same declared revision, different bytes, then no live package at all.
            await store.put(key, _declaration("guide", "CHANGED_RULE"))
            await store.delete(key)

        async def recovered(messages, info):
            assert "ORIGINAL_RULE" in str(info)
            assert "CHANGED_RULE" not in str((messages, info))
            returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
            if returns[-1].tool_call_id == "before":
                return ModelResponse(parts=[ToolCallPart("load_skill", {"skill_id": "guide"}, "after")])
            assert returns[-1].content["instructions"].endswith("ORIGINAL_RULE")
            assert returns[-1].content["resource_status"] == "unavailable"
            return ModelResponse(parts=[TextPart("done")])

        async with open_runtime(recovered) as (runtime, store):
            await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal, idempotency_key="recover")
            outcome = await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=20)
            assert outcome.result.status is ExecutionStatus.SUCCEEDED, outcome.result
            preserved = await runtime.history.read_model_interactions(execution.execution_id,
                principal=runtime.default_principal, identities=((1, 1),), include_content=True)
            assert preserved == old_requests
            page = await runtime.history.history(execution.execution_id, principal=runtime.default_principal,
                tool_call_id="after", include_content=True)
            loaded, = (item for item in page.items if item.item_kind == "tool_result")
            assert loaded.skill_read.asset == SkillReadReceipt.from_payload(old_requests[0].request["required_skill_reads"][0]).asset
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_execution_authorization_precedes_required_retained_read_and_model(tmp_path: Path):
    class RecordingStore(AssetStore):
        reads = 0
        async def require_retained_versions(self, refs):
            self.reads += 1
            return await super().require_retained_versions(refs)

    class Authorization:
        async def authorize(self, principal, action, resource):
            if action is AuthorizationAction.EXECUTION_RUN:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    backend = InMemoryAssetBackend()
    store = RecordingStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    await store.put(AssetKey("skill", "guide/SKILL.md"), _declaration("guide", "RULE"))

    async def model(messages, info):
        raise AssertionError("denied execution must not reach model")

    async with Persistence.open("required-auth", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(store, model), authorization=Authorization()) as runtime:
            # Discovery may parse declarations during configuration. Required
            # retained-version reads are a separate, authorized admission step.
            assert store.reads == 0
            with pytest.raises(AIError) as error:
                await runtime.agents.get().start("go", idempotency_key="denied")
            assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert store.reads == 0


@pytest.mark.asyncio
async def test_required_descriptor_cannot_disguise_another_asset(tmp_path: Path):
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    for key in (AssetKey("skill", "other/SKILL.md"), AssetKey("secret", "guide/SKILL.md")):
        await store.put(key, b"PRIVATE")
        ref, = await store.resolve_versions((key,))
        with pytest.raises(AIError) as error:
            SkillInput(CapabilityRef("skill", "guide", 1),
                       SkillSourceRef("required-skills", "guide", declaration=SkillResource("SKILL.md", ref)))
        assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("attack", ("old-version", "other-skill"))
async def test_imported_required_input_cannot_select_unadmitted_history(tmp_path: Path, attack: str):
    reads = []
    class RecordingStore(AssetStore):
        async def require_retained_versions(self, refs):
            reads.extend(refs)
            return await super().require_retained_versions(refs)

    backend = InMemoryAssetBackend()
    store = RecordingStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    key = AssetKey("skill", "guide/SKILL.md")
    await store.put(key, _declaration("guide", "UNADMITTED_OLD_RULE"))
    old, = await store.resolve_versions((key,))
    await store.put(key, _declaration("guide", "CURRENT_RULE"))
    fake = SkillInput(CapabilityRef("skill", "guide", 1),
                      SkillSourceRef("required-skills", "guide", declaration=SkillResource("SKILL.md", old))).to_payload()
    unadmitted = old
    if attack == "other-skill":
        foreign = AssetKey("skill", "private/SKILL.md")
        await store.put(foreign, _declaration("private", "PRIVATE_RULE"))
        unadmitted, = await store.resolve_versions((foreign,))
        fake["source"]["declaration"]["asset"] = unadmitted.to_payload()

    async def model(messages, info):
        raise AssertionError("unadmitted imported version must not reach the model")

    async with Persistence.open("required-import-fake", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as persistence:
        async with Runtime.attach(persistence=persistence, features=_features(store, model)) as runtime:
            with pytest.raises(AIError) as error:
                await runtime.agents.get().start("go", idempotency_key="run",
                    input_context=ExecutionInputContext.from_messages((), required_skills=(fake,)))
            assert error.value.code is (ErrorCode.BINDING_CONFLICT if attack == "old-version" else ErrorCode.STORAGE_INTEGRITY_ERROR)

    assert unadmitted not in reads
