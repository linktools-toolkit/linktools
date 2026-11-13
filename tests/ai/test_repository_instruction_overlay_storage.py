#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rule overlays retain Asset history without changing accepted barrier semantics."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.asset import AssetKey, AssetStore, SqlAssetBackend
from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionRecord, ExecutionRecoveryDetail, RepositoryInstructionBarrier
from linktools.ai.migrate import provision_asset_database
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.runtime import Runtime
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.inputs import RepositoryInstructionPayload
from linktools.ai.runtime._repository_instructions import _RepositoryInstructionBoundary, _validate_repository_instruction_frontier
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, StorageOverlay, StoredPayload
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features, native_checkpoint, workspace_features
from .test_archive_closure import Offline
from .test_incremental_projection_boundaries import _storage
from .test_rule_context_storage import _archive, _assets, _rules
from .test_runtime_persistence_evidence import _execution

pytestmark = pytest.mark.asyncio


class _OverlayPort:
    load_repository_instructions = LocalExecutionBackend.load_repository_instructions
    _store_repository_instructions = LocalExecutionBackend._store_repository_instructions
    commit_repository_instruction_barrier = LocalExecutionBackend.commit_repository_instruction_barrier

    def __init__(self, state: Persistence, assets: AssetStore) -> None:
        self._context_archive = _archive(state, {"rules": assets})
        self._execution = state.execution
        self._conversation = state.conversation
        self._tenant_id = state.tenant_id
        self._turn_owners = {}

    async def promote_pending_execution(self, execution_id: str) -> None:
        assert execution_id == "execution"


def _barrier(identity: str = "call") -> RepositoryInstructionBarrier:
    return RepositoryInstructionBarrier("run", identity, canonical_sha256({"path": "pkg/file.txt"}))


def _frontier(execution: ExecutionRecord, overlay: RepositoryInstructions) -> str:
    boundary = _RepositoryInstructionBoundary(None, execution, None, overlay)
    boundary.render_overlay()
    return boundary.approval_frontiers((ToolCallPart("read_file", {"path": "pkg/file.txt"}, "call"),))["call"]


async def _create_execution(state: Persistence) -> ExecutionRecord:
    execution = replace(_execution(), recovery=ExecutionRecoveryDetail(agent_run_id="run"))
    await state.execution.executions.create_execution(execution)
    return execution


async def _mixed_overlay(assets: AssetStore) -> RepositoryInstructions:
    return RepositoryInstructions((*(await _rules(assets)).documents,
        RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "accepted Workspace text"),
        RepositoryInstructionDocument("rule:custom", "pkg", "accepted custom text"),
    ))


@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_overlay_keeps_fixed_assets_and_all_original_semantic_digests(tmp_path: Path, backend: str) -> None:
    async with _assets() as assets, Persistence.open("overlay", storage=_storage(backend, tmp_path), tenant_id="tenant") as state:
        key = AssetKey("rule", "review.md")
        body = "retained Rule body " * 4096
        await assets.put(key, f"---\nscope: pkg\n---\n{body}".encode())
        overlay = await _mixed_overlay(assets)
        execution = await _create_execution(state)
        port = _OverlayPort(state, assets)
        barrier = _barrier()
        committed = await port.commit_repository_instruction_barrier(execution, overlay, barrier)
        reference = committed.recovery.repository_instruction_overlay
        wire = reference.payload.decode()
        assert wire["semantic_digest"] == canonical_sha256(overlay.to_payload())
        assert reference.payload.digest == canonical_sha256(wire)
        assert body.encode() not in canonical_json_bytes(wire)
        assert len(canonical_json_bytes(wire)) < 2048
        assert not reference.payload.object_refs()
        assert {item["content"] for item in wire["instructions"]["documents"] if "content" in item} == {
            "accepted Workspace text", "accepted custom text",
        }
        assert committed.recovery.repository_instruction_barriers == (barrier,)
        expected_frontier = canonical_sha256({"initial": [], "overlay": [
            [document.source, document.scope, document.content] for document in overlay.documents
        ]})
        assert _frontier(committed, overlay) == expected_frontier
        await assets.put(key, b"changed Rule")
        await assets.delete(key)
        await state.purge_unreferenced_objects((), exclusive=Offline())
        cold = _OverlayPort(state, assets)
        retained = await state.execution.executions.get_recovery("execution", tenant_id="tenant")
        restored = await cold.load_repository_instructions(retained.recovery.repository_instruction_overlay)
        assert restored.to_payload() == overlay.to_payload()
        _validate_repository_instruction_frontier(retained, restored)
        assert _frontier(retained, restored) == expected_frontier


@pytest.mark.parametrize("outcome", ("acknowledgement_lost", "conflict", "historical_barrier"))
async def test_overlay_commit_uses_durable_readback_without_replacing_a_newer_frontier(
    monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    async with _assets() as assets, Persistence.open("overlay-readback", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await assets.put(AssetKey("rule", "review.md"), b"---\nscope: pkg\n---\naccepted Rule")
        first = await _rules(assets)
        execution = await _create_execution(state)
        port = _OverlayPort(state, assets)
        barrier = _barrier()
        repository = state.execution.executions
        if outcome == "acknowledgement_lost":
            original = repository.compare_and_swap

            async def committed_then_lost(
                execution_id: str, *, tenant_id: str, expected_revision: int, next_record: ExecutionRecord,
            ) -> ExecutionRecord:
                await original(execution_id, tenant_id=tenant_id, expected_revision=expected_revision, next_record=next_record)
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

            monkeypatch.setattr(repository, "compare_and_swap", committed_then_lost)
            committed = await port.commit_repository_instruction_barrier(execution, first, barrier)
            assert committed.recovery.repository_instruction_barriers == (barrier,)
            assert (await port.load_repository_instructions(committed.recovery.repository_instruction_overlay)).to_payload() == first.to_payload()
        else:
            committed = await port.commit_repository_instruction_barrier(execution, first, barrier)
            if outcome == "conflict":
                with pytest.raises(AIError) as failure:
                    await port.commit_repository_instruction_barrier(execution, first, _barrier("different-call"))
                assert failure.value.code is ErrorCode.STORAGE_CONFLICT
                assert (await repository.get_recovery("execution", tenant_id="tenant")).recovery == committed.recovery
            else:
                latest = RepositoryInstructions((*first.documents,
                    RepositoryInstructionDocument("rule:later", "pkg/deep", "later accepted custom text"),
                ))
                extended = await port.commit_repository_instruction_barrier(committed, latest, _barrier("later-call"))
                replayed = await port.commit_repository_instruction_barrier(committed, first, barrier)
                assert replayed.recovery == extended.recovery
                assert (await port.load_repository_instructions(replayed.recovery.repository_instruction_overlay)).to_payload() == latest.to_payload()


async def test_overlay_reads_fixed_asset_and_inline_content_without_digest_rechecks() -> None:
    async with _assets() as assets, Persistence.open("overlay-read", storage=PersistenceStorage.in_memory()) as state:
        await assets.put(AssetKey("rule", "review.md"), b"---\nscope: pkg\n---\naccepted Rule")
        overlay = await _mixed_overlay(assets)
        port = _OverlayPort(state, assets)
        reference = await port._store_repository_instructions(None, overlay)
        wire = reference.payload.decode()
        # These identities are compared when accepting/replacing a buffered
        # admission, not recomputed while loading its authoritative content.
        wire["semantic_digest"] = "0" * 64
        next(item for item in wire["instructions"]["documents"] if "asset" in item)["content_digest"] = "0" * 64
        restored = await port.load_repository_instructions(
            replace(reference, payload=StoredPayload.inline_json(wire))
        )
        assert restored == overlay


@pytest.mark.parametrize("corruption", ("descriptor", "old_shape", "version"))
async def test_overlay_rejects_wrong_descriptors_and_unsupported_shapes(corruption: str) -> None:
    async with _assets() as assets, Persistence.open("overlay-integrity", storage=PersistenceStorage.in_memory()) as state:
        await assets.put(AssetKey("rule", "review.md"), b"---\nscope: pkg\n---\naccepted Rule")
        overlay = await _mixed_overlay(assets)
        port = _OverlayPort(state, assets)
        reference = await port._store_repository_instructions(None, overlay)
        wire = reference.payload.decode()
        if corruption == "old_shape":
            wire = overlay.to_payload()
        elif corruption == "descriptor":
            next(item for item in wire["instructions"]["documents"] if "asset" in item)["scope"] = "wrong-scope"
        else:
            wire["version"] = 2
        malformed = replace(reference, payload=StoredPayload.inline_json(wire))
        with pytest.raises(AIError) as failure:
            await port.load_repository_instructions(malformed)
        assert failure.value.code is (ErrorCode.STORAGE_VERSION_UNSUPPORTED if corruption == "version" else ErrorCode.STORAGE_INTEGRITY_ERROR)


async def test_overlay_snapshot_keeps_asset_references_and_accepted_inline_bodies(tmp_path: Path) -> None:
    key = AssetKey("rule", "review.md")
    asset_objects = FilesystemObjectStore(tmp_path / "asset-history", store_id="asset-history")
    async with _assets() as source:
        await source.put(key, b"---\nscope: pkg\n---\nSNAPSHOT_RULE_BODY")
        asset_snapshot = await source.snapshot((key,), object_store=asset_objects)
        await source.put(key, b"changed current Rule")
    assets = AssetStore.from_snapshot(asset_snapshot, object_store=asset_objects)
    await assets.initialize()
    configuration = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    try:
        async with Persistence.open("overlay-snapshot", storage=configuration, tenant_id="tenant") as state:
            overlay = await _mixed_overlay(assets)
            port = _OverlayPort(state, assets)
            committed = await port.commit_repository_instruction_barrier(await _create_execution(state), overlay, _barrier())
            await state.purge_unreferenced_objects((), exclusive=Offline())
            expected_frontier = _frontier(committed, overlay)
    finally:
        await assets.close()
    transport = InMemoryObjectStore("overlay-snapshot")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    async with Persistence.open("overlay-snapshot", storage=configuration, tenant_id="tenant", read_only=True) as source:
        snapshot = await source.export_snapshot(object_store=transport, limits=limits)
    restored_root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=transport, root=restored_root, limits=limits)
    reopened = AssetStore.from_snapshot(asset_snapshot, object_store=FilesystemObjectStore(
        tmp_path / "asset-history", store_id="asset-history"))
    await reopened.initialize()
    try:
        async with Persistence.open("overlay-snapshot", storage=PersistenceStorage.from_root(restored_root), tenant_id="tenant") as state:
            retained = await state.execution.executions.get_recovery("execution", tenant_id="tenant")
            assert retained.recovery == committed.recovery
            port = _OverlayPort(state, reopened)
            restored = await port.load_repository_instructions(retained.recovery.repository_instruction_overlay)
            assert restored.to_payload() == overlay.to_payload()
            _validate_repository_instruction_frontier(retained, restored)
            assert _frontier(retained, restored) == expected_frontier
            assert RepositoryInstructionPayload.from_reference(retained.recovery.repository_instruction_overlay).semantic_digest == canonical_sha256(overlay.to_payload())
    finally:
        await reopened.close()


async def test_rule_overlay_handoff_recovers_original_sql_asset_and_workspace_bytes(tmp_path: Path) -> None:
    package = tmp_path / "workspace" / "pkg"
    package.mkdir(parents=True)
    workspace_instructions = package / "AGENTS.md"
    workspace_instructions.write_text("ORIGINAL_WORKSPACE_RULE")
    (package / "file.txt").write_text("file contents")
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'assets.db'}")
    await provision_asset_database(engine)
    key = AssetKey("rule", "review.md")
    blocked = asyncio.Event()

    @asynccontextmanager
    async def open_runtime(
        model: Callable[[list[ModelMessage], AgentInfo], Awaitable[ModelResponse]], *, seed: bool = False,
    ) -> AsyncIterator[tuple[Runtime, Persistence, AssetStore]]:
        backend = SqlAssetBackend(engine, namespace="overlay-history")
        assets = AssetStore(StorageOverlay(backend, writer=backend))
        await assets.initialize()
        try:
            if seed:
                await assets.put(key, b"---\nscope: pkg\n---\nORIGINAL_ASSET_RULE")
            features = workspace_features("rules", workspace=Workspace.load(tmp_path / "workspace"), assets=assets)
            features.capability(ExecutionCapability(retain_history=True))
            features.capability(native_checkpoint())
            features.agent("reader", allow_tools=("read_file",), allow_skills=(), allow_subagents=())
            model_features(_Models(_UsageFunctionModel(model)), features)
            async with Persistence.open("overlay-handoff", storage=PersistenceStorage.sqlite(tmp_path / "runtime-handoff.db")) as state:
                async with Runtime.attach(persistence=state, features=features.borrow(), auto_recover=False) as runtime:
                    yield runtime, state, assets
        finally:
            await assets.close()

    async def before_handoff(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns:
            assert "ORIGINAL_ASSET_RULE" in (info.instructions or "")
            assert "ORIGINAL_WORKSPACE_RULE" in (info.instructions or "")
            blocked.set()
            await asyncio.Event().wait()
        assert "ORIGINAL_ASSET_RULE" not in (info.instructions or "")
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "pkg/file.txt"}, "before")])

    async def after_handoff(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert "ORIGINAL_ASSET_RULE" in (info.instructions or "")
        assert "ORIGINAL_WORKSPACE_RULE" in (info.instructions or "")
        assert "CHANGED" not in (info.instructions or "")
        returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        if returns[-1].tool_call_id == "before":
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "pkg/file.txt"}, "after")])
        assert "file contents" in str(returns[-1].content)
        return ModelResponse(parts=[TextPart("done")])

    try:
        async with open_runtime(before_handoff, seed=True) as (runtime, state, assets):
            execution = await runtime.agents.get("reader").start("read the file")
            await asyncio.wait_for(blocked.wait(), 20)
            record = await state.execution.executions.get_recovery(execution.execution_id, tenant_id=runtime.tenant_id)
            reference = record.recovery.repository_instruction_overlay
            assert b"ORIGINAL_ASSET_RULE" not in canonical_json_bytes(reference.payload.decode())
            assert b"ORIGINAL_WORKSPACE_RULE" in canonical_json_bytes(reference.payload.decode())
            await asyncio.wait_for(runtime.handoff(), 20)
            await assets.put(key, b"---\nscope: pkg\n---\nCHANGED_ASSET_RULE")
            await assets.delete(key)
            workspace_instructions.write_text("CHANGED_WORKSPACE_RULE")
        async with open_runtime(after_handoff) as (runtime, state, _assets_owner):
            await runtime.executions.recover(execution.execution_id, principal=runtime.default_principal, idempotency_key="recover")
            result = (await runtime.executions.wait(execution.execution_id, principal=runtime.default_principal, timeout_seconds=20)).result
            assert result.status is ExecutionStatus.SUCCEEDED, result
    finally:
        await engine.dispose()
