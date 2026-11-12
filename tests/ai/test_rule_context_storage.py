#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted Rule references preserve exact Asset history without duplicate bodies."""

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.asset import AssetKey, AssetStore, AssetStoreReader, DirectoryAssetBackend, InMemoryAssetBackend, PrefixAssetPathAdapter
from linktools.ai.capability import AssetCapability
from linktools.ai.core import ExecutionStatus, JsonValue, Principal, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext, ExecutionRecord
from linktools.ai.inputs import CaptureInputRequest, ExecutionContextArchive
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.captures import capture_record_value
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, StorageOverlay
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import model_features
from .test_archive_closure import Offline
from .test_captured_execution_context import _NoToolsModels
from .test_execution_context_archive import _publish_context
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _assets() -> AsyncIterator[AssetStore]:
    backend = InMemoryAssetBackend()
    store = AssetStore(StorageOverlay(backend, writer=backend))
    await store.initialize()
    try:
        yield store
    finally:
        await store.close()


async def _rules(store: AssetStore) -> RepositoryInstructions:
    features = RuntimeFeatures("rules")
    features.capability(AssetCapability(store))
    return (await features.resolve()).instructions


def _archive(
    state: Persistence, sources: Mapping[str, AssetStoreReader] | None = None,
) -> ExecutionContextArchive:
    return ExecutionContextArchive(
        state.execution.executions.state_store,
        state.archive_object_store(PersistenceDomain.EXECUTION),
        namespace=state.namespace, tenant_id=state.tenant_id, asset_sources=sources,
        object_reader=lambda reference: state.read_archive_object(PersistenceDomain.EXECUTION, reference),
    )


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_retained_rules_keep_only_references_and_survive_head_deletion_and_gc(tmp_path: Path, backend: str) -> None:
    async with _assets() as assets, Persistence.open("rule-context", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        key = AssetKey("rule", "review.md")
        body = "ORIGINAL_RULE_BODY" * 4096
        await assets.put(key, f"---\nscope: src\n---\n{body}".encode())
        rule, = (await _rules(assets)).documents
        instructions = RepositoryInstructions((rule,
            RepositoryInstructionDocument("agents:AGENTS.md", ".", "accepted Workspace instruction"),
            RepositoryInstructionDocument("rule:custom", ".", "accepted custom resolver content"),
        )).to_payload()
        context = ExecutionInputContext.from_messages((), repository_instructions=instructions)
        service = _archive(state, {"rules": assets})
        prepared = await service.prepare(context)
        await _publish_context(state, prepared)
        component = prepared.reference.payload.decode()["repository_instructions"]
        raw, = await prepared.archive.read_contents((component["content_id"],), codec="json")
        stored = json.loads(raw)
        stored_rule = next(item for item in stored["documents"] if item["source"] == "rule:review")
        assert "content" not in stored_rule
        assert stored_rule["asset"] == rule.asset.to_payload()
        assert body.encode() not in raw
        assert len(raw) < 2048
        assert {item["content"] for item in stored["documents"] if "content" in item} == {
            "accepted Workspace instruction", "accepted custom resolver content",
        }
        content, = await prepared.archive.content_records((component["content_id"],))
        assert content.payload is None and isinstance(content.body, bytes) and not content.object_refs()
        await assets.put(key, b"changed scope and body")
        await assets.delete(key)
        await state.purge_unreferenced_objects((), exclusive=Offline())
        cold = _archive(state, {"rules": assets})
        assert await cold.load_repository_instructions(prepared.reference) == instructions
        assert await cold.load(prepared.reference) == context
        with pytest.raises(AIError) as missing:
            await _archive(state).load(prepared.reference)
        assert missing.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING


async def test_directory_rule_keeps_accepted_body_without_claiming_retention(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    (root / "rules").mkdir(parents=True)
    path = root / "rules" / "review.md"
    path.write_text("accepted directory Rule")
    assets = AssetStore(StorageOverlay(DirectoryAssetBackend(str(root),
        path_adapter=PrefixAssetPathAdapter({"rule": "rules"}), kinds=("rule",))))
    await assets.initialize()
    try:
        async with Persistence.open("directory-rule", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
            instructions = (await _rules(assets)).to_payload()
            service = _archive(state, {"rules": assets})
            context = ExecutionInputContext.from_messages((), repository_instructions=instructions)
            prepared = await service.prepare(context)
            await _publish_context(state, prepared)
            component = prepared.reference.payload.decode()["repository_instructions"]
            raw, = await prepared.archive.read_contents((component["content_id"],), codec="json")
            assert json.loads(raw)["documents"][0]["content"] == "accepted directory Rule"
            path.write_text("changed directory Rule")
            assert await _archive(state).load(prepared.reference) == context
    finally:
        await assets.close()


@pytest.mark.parametrize("field, replacement", (
    ("source", "rule:other"), ("scope", "other"), ("content_digest", "0" * 64),
    ("asset", {"version": 1}),
))
async def test_rule_reference_rejects_corrupt_semantics_or_version_metadata(
    tmp_path: Path, field: str, replacement: JsonValue,
) -> None:
    async with _assets() as assets, Persistence.open("corrupt-rule", storage=PersistenceStorage.in_memory()) as state:
        await assets.put(AssetKey("rule", "review.md"), b"---\nscope: src\n---\naccepted Rule")
        service = _archive(state, {"rules": assets})
        stored = await service.encode_repository_instructions((await _rules(assets)).to_payload())
        stored["documents"][0][field] = replacement
        with pytest.raises(AIError) as failure:
            await service.decode_repository_instructions(stored)
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_rule_reference_never_substitutes_a_missing_version_or_mismatched_body() -> None:
    async with _assets() as assets, Persistence.open("missing-rule", storage=PersistenceStorage.in_memory()) as state:
        key = AssetKey("rule", "review.md")
        await assets.put(key, b"accepted Rule")
        instructions = await _rules(assets)
        service = _archive(state, {"rules": assets})
        stored = await service.encode_repository_instructions(instructions.to_payload())
        stored["documents"][0]["asset"]["revision"] += 100
        with pytest.raises(AIError) as failure:
            await service.decode_repository_instructions(stored)
        assert failure.value.code is ErrorCode.ASSET_VERSION_NOT_FOUND
        forged = RepositoryInstructions((replace(instructions.documents[0], content="different accepted text"),))
        with pytest.raises(AIError) as mismatch:
            await service.encode_repository_instructions(forged.to_payload())
        assert mismatch.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("context_policy", ("captured", "clean"))
async def test_rule_references_and_input_captures_recover_from_separate_asset_and_runtime_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, context_policy: str,
) -> None:
    key = AssetKey("rule", "review.md")
    asset_objects = FilesystemObjectStore(tmp_path / "asset-history", store_id="asset-history")
    async with _assets() as author:
        await author.put(key, b"ORIGINAL_SNAPSHOT_RULE")
        asset_snapshot = await author.snapshot((key,), object_store=asset_objects)
        await author.put(key, b"CHANGED_AUTHOR_RULE")
    assets = AssetStore.from_snapshot(asset_snapshot, object_store=asset_objects)
    await assets.initialize()
    source_root = tmp_path / "source"
    principal = Principal("owner", "default")
    try:
        features = RuntimeFeatures("rules")
        features.capability(AssetCapability(assets))
        features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        async with Persistence.open("portable-rules", storage=PersistenceStorage.from_root(source_root, exclusive_writer=True)) as state:
            async with Runtime.attach(persistence=state, features=model_features(_NoToolsModels(), features).borrow()) as runtime:
                execution = await runtime.agents.get().start("question", principal=principal)
                assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
                record = await state.execution.executions.get(execution.execution_id, tenant_id=principal.tenant_id)
                original = await _archive(state, {"rules": assets}).load(record.input_context)
                assert original.repository_instructions["documents"][0]["content"] == "ORIGINAL_SNAPSHOT_RULE"
    finally:
        await assets.close()
    transport = InMemoryObjectStore("rule-snapshot")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    async with Persistence.open("portable-rules", storage=PersistenceStorage.from_root(source_root), read_only=True) as source:
        snapshot = await source.export_snapshot(object_store=transport, limits=limits)
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=transport, root=restored_root, limits=limits)
    reopened = AssetStore.from_snapshot(asset_snapshot, object_store=FilesystemObjectStore(
        tmp_path / "asset-history", store_id="asset-history"))
    await reopened.initialize()
    try:
        async with Persistence.open("portable-rules", storage=PersistenceStorage.from_root(restored_root)) as restored:
            async with TaskEngine.open(persistence=restored, asset_sources={"rules": reopened}) as engine:
                repository = restored.execution.executions
                group = repository.state_store.storage_group
                assert group.opened_archive is None
                original_get = repository.get
                reads = 0

                async def get_in_content_scope(identity: str, *, tenant_id: str) -> ExecutionRecord | None:
                    nonlocal reads
                    reads += 1
                    scope = active_state_scope()
                    assert scope is not None and scope.group is group and not scope.writable
                    record = await original_get(identity, tenant_id=tenant_id)
                    assert group.opened_archive[1] in scope.transactions
                    return record

                with monkeypatch.context() as patch:
                    patch.setattr(repository, "get", get_in_content_scope)
                    capture = await engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(
                        principal, "snapshot-capture", context_policy=context_policy))
                assert reads == 2
                imported = await engine.inputs.read_agent(capture, principal=principal)
                if context_policy == "captured":
                    assert imported.input_context.digest == original.digest
                else:
                    assert imported.input_context is None
                assert imported.repository_instructions == original.repository_instructions
                records = await restored.task.tasks.state_store.read(lambda transaction: transaction.scan_records())
                captures = [capture_record_value(item) for item in records if item.kind == "input_capture"]
                assert captures and b"ORIGINAL_SNAPSHOT_RULE" not in canonical_json_bytes(captures)
            async with TaskEngine.open(persistence=restored, asset_sources={"rules": reopened}) as cold:
                recaptured = await cold.inputs.read_agent(capture, principal=principal)
                assert recaptured.input_context == imported.input_context
                assert recaptured.repository_instructions == imported.repository_instructions
    finally:
        await reopened.close()


@pytest.mark.parametrize("borrowed", (False, True))
async def test_runtime_tasks_capture_uses_its_configured_rule_asset_sources(borrowed: bool) -> None:
    async with _assets() as assets:
        key = AssetKey("rule", "review.md")
        await assets.put(key, b"ORIGINAL_RUNTIME_RULE")
        features = RuntimeFeatures("rules")
        features.capability(AssetCapability(assets))
        features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        model_features(_NoToolsModels(), features)
        async with Runtime.open("runtime-rule-capture", storage=PersistenceStorage.in_memory(),
                                features=features.borrow() if borrowed else features) as runtime:
            principal = Principal("owner", runtime.tenant_id)
            execution = await runtime.agents.get().start("question", principal=principal)
            assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
            await assets.put(key, b"CHANGED_RUNTIME_RULE")
            engine = await runtime.tasks.bind()
            with pytest.raises(AIError) as denied:
                await engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(
                    runtime.default_principal, "local-trusted-capture"))
            assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
            for policy in ("captured", "clean"):
                capture = await engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(
                    principal, f"runtime-capture-{policy}", context_policy=policy))
                imported = await engine.inputs.read_agent(capture, principal=principal)
                assert imported.repository_instructions["documents"][0]["content"] == "ORIGINAL_RUNTIME_RULE"
                assert (imported.input_context is None) is (policy == "clean")
                cold = await runtime.tasks.bind()
                assert (await cold.inputs.read_agent(capture, principal=principal)).repository_instructions == imported.repository_instructions


@pytest.mark.parametrize("corruption, expected", (
    ("missing_source", ErrorCode.CAPABILITY_REQUIRED_MISSING),
    ("wrong_source", ErrorCode.STORAGE_INTEGRITY_ERROR),
    ("wrong_layer", ErrorCode.ASSET_VERSION_LAYER_UNKNOWN),
))
async def test_rule_reference_rejects_the_wrong_asset_owner_or_layer(corruption: str, expected: ErrorCode) -> None:
    async with _assets() as assets, _assets() as other, Persistence.open(
        "rule-owner", storage=PersistenceStorage.in_memory(),
    ) as state:
        key = AssetKey("rule", "review.md")
        await assets.put(key, b"accepted Rule")
        await other.put(key, b"other owner's Rule")
        service = _archive(state, {"rules": assets, "other": other})
        stored = await service.encode_repository_instructions((await _rules(assets)).to_payload())
        if corruption == "wrong_layer":
            stored["documents"][0]["asset"]["layer_id"] = "missing-layer"
        else:
            stored["documents"][0]["asset_source_id"] = "other" if corruption == "wrong_source" else "missing-source"
        with pytest.raises(AIError) as failure:
            await service.decode_repository_instructions(stored)
        assert failure.value.code is expected
