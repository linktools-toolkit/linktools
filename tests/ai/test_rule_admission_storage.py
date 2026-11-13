#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Admission Rule references retain source-first and interrupted execution inputs."""

import asyncio
from collections.abc import Hashable
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.asset import AssetKey, AssetStore
from linktools.ai.capability import AssetCapability, ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain, Principal, canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionInputContext, ExecutionRecord, ExecutionRequest, ExecutionStartReservation, ExecutionStartReservationResult
from linktools.ai.inputs import CaptureInputRequest, RepositoryInstructionPayload
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer, _ExecutionPayloadCapture
from linktools.ai.session import SessionRecord
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import FilesystemObjectStore, InMemoryObjectStore, StoredPayload
from linktools.ai.task.engine import TaskEngine

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_archive_closure import Offline
from .test_execution_payload_buffer import _Buffer
from .test_rule_context_storage import _archive, _assets, _rules
from .test_runtime_persistence_evidence import _execution

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("outcome", ("success", "failure", "cancelled"))
async def test_source_first_rule_admission_survives_deleted_head_before_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str,
) -> None:
    entered = asyncio.Event()
    requests = 0
    body = "ORIGINAL_ADMISSION_RULE" * 2048
    admitted = []

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert body in (info.instructions or "")
        assert "CHANGED_RULE" not in (info.instructions or "")
        return ModelResponse(parts=[TextPart("done")])

    async with _assets() as assets:
        key = AssetKey("rule", "admission.md")
        await assets.put(key, body.encode())
        features = RuntimeFeatures("rules")
        features.capability(AssetCapability(assets))
        features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
        features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
        async with Persistence.open("rule-admission", storage=PersistenceStorage.sqlite(tmp_path / "runtime.db")) as state:
            async with Runtime.attach(persistence=state, features=model_features(
                _Models(_UsageFunctionModel(model)), features).borrow()) as runtime:
                principal = Principal("owner", runtime.tenant_id)
                backend = runtime._execution_service.runtime_backend()
                bridge = runtime._execution_service._runtime_bridge
                prepare = bridge.prepare_reserved_start
                original_context = backend._context_archive.prepare

                async def source_first(
                    request: ExecutionRequest, reservation: ExecutionStartReservation, session: SessionRecord | None,
                ) -> ExecutionStartReservationResult | None:
                    reference = reservation.execution.repository_instructions
                    wire = reference.payload.decode()
                    assert reference.source_domain is PersistenceDomain.EXECUTION
                    assert body.encode() not in canonical_json_bytes(wire)
                    assert len(canonical_json_bytes(wire)) < 2048
                    assert not reference.payload.object_refs()
                    accepted = await backend.load_repository_instructions(reference)
                    assert wire["semantic_digest"] == canonical_sha256(accepted.to_payload())
                    await assets.put(key, b"CHANGED_RULE")
                    await assets.delete(key)
                    result = await prepare(request, reservation, session)
                    record = await state.execution.executions.read_content(lambda _group: state.execution.executions.get(
                        reservation.execution.execution_id, tenant_id=runtime.tenant_id))
                    assert record is not None and record.input_context is None
                    assert record.repository_instructions == reference
                    admitted.append(reference)
                    return result

                async def context_boundary(context: ExecutionInputContext, **kwargs: object) -> object:
                    assert context.repository_instructions["documents"][0]["content"] == body
                    entered.set()
                    if outcome == "failure":
                        raise AIError(ErrorCode.INPUT_CONTEXT_UNAVAILABLE)
                    if outcome == "cancelled":
                        await asyncio.Event().wait()
                    return await original_context(context, **kwargs)

                monkeypatch.setattr(bridge, "prepare_reserved_start", source_first)
                monkeypatch.setattr(backend._context_archive, "prepare", context_boundary)
                execution = await runtime.agents.get().start("question", principal=principal)
                await asyncio.wait_for(entered.wait(), 10)
                if outcome == "cancelled":
                    await execution.cancel(idempotency_key="cancel-admission")
                if outcome == "failure":
                    with pytest.raises(AIError) as unavailable:
                        await execution.wait(timeout_seconds=10)
                    assert unavailable.value.code is ErrorCode.INPUT_CONTEXT_UNAVAILABLE
                else:
                    result = (await execution.wait(timeout_seconds=10)).result
                    assert result.status is {
                        "success": ExecutionStatus.SUCCEEDED, "cancelled": ExecutionStatus.CANCELLED,
                    }[outcome]
                assert requests == (1 if outcome == "success" else 0)
                reference, = admitted
                assert (await backend.load_repository_instructions(reference)).documents[0].content == body
                if outcome != "success":
                    record = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                    assert record.input_context is None and record.repository_instructions == reference
                    if outcome == "failure":
                        assert record.status is ExecutionStatus.STARTED and record.result is None
                    engine = await runtime.tasks.bind()
                    for policy in ("captured", "clean"):
                        captured = await engine.inputs.capture_input(execution.execution_id, CaptureInputRequest(
                            principal, "capture-" + policy, context_policy=policy))
                        restored = await engine.inputs.read_agent(captured, principal=principal)
                        assert restored.repository_instructions["documents"][0]["content"] == body
                        assert (restored.input_context is None) is (policy == "clean")
                if outcome == "failure":
                    # A failed storage preparation is not a confirmed business
                    # terminal. Resolve the retained owner through public control.
                    monkeypatch.setattr(backend._context_archive, "prepare", original_context)
                    await execution.cancel(idempotency_key="resolve-failed-admission")
                    assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.CANCELLED
                    assert requests == 0
                    record = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                    assert record.input_context is None and record.repository_instructions == reference


async def test_pre_context_admission_reopens_from_asset_and_runtime_snapshots(tmp_path: Path) -> None:
    key = AssetKey("rule", "admission.md")
    objects = FilesystemObjectStore(tmp_path / "assets", store_id="asset-history")
    async with _assets() as author:
        await author.put(key, b"ADMISSION_SNAPSHOT_RULE")
        snapshot = await author.snapshot((key,), object_store=objects)
    assets = AssetStore.from_snapshot(snapshot, object_store=objects)
    await assets.initialize()
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    principal = Principal("principal", "tenant", "service")
    try:
        async with Persistence.open("admission-snapshot", storage=storage, tenant_id="tenant") as state:
            accepted = RepositoryInstructions((*(await _rules(assets)).documents,
                RepositoryInstructionDocument("agents:AGENTS.md", ".", "accepted Workspace body"),
                RepositoryInstructionDocument("rule:custom", ".", "accepted custom body"),
            ))
            reference = await _archive(state, {"rules": assets}).prepare_repository_instruction_payload(
                accepted, source_domain=PersistenceDomain.EXECUTION)
            await state.execution.executions.create_execution(replace(
                _execution(), status=ExecutionStatus.PENDING_START, agent_run_seq=0,
                repository_instructions=reference,
            ))
            await state.purge_unreferenced_objects((), exclusive=Offline())
    finally:
        await assets.close()
    transport = InMemoryObjectStore("runtime-snapshot")
    limits = SnapshotLimits(max_entries=4096, max_bytes=16 * 1024 * 1024)
    async with Persistence.open("admission-snapshot", storage=storage, tenant_id="tenant", read_only=True) as state:
        runtime_snapshot = await state.export_snapshot(object_store=transport, limits=limits)
    await Persistence.restore_snapshot(runtime_snapshot, object_store=transport, root=tmp_path / "restored", limits=limits)
    assets = AssetStore.from_snapshot(snapshot, object_store=FilesystemObjectStore(tmp_path / "assets", store_id="asset-history"))
    await assets.initialize()
    try:
        for policy in ("captured", "clean"):
            async with Persistence.open("admission-snapshot", storage=PersistenceStorage.from_root(tmp_path / "restored"), tenant_id="tenant") as state:
                async with TaskEngine.open(persistence=state, asset_sources={"rules": assets}) as engine:
                    captured = await engine.inputs.capture_input("execution", CaptureInputRequest(principal, "capture-" + policy, policy))
                    restored = await engine.inputs.read_agent(captured, principal=principal)
                    assert restored.repository_instructions == accepted.to_payload()
                    assert (restored.input_context is None) is (policy == "clean")
                    wire = reference.payload.decode()
                    assert b"ADMISSION_SNAPSHOT_RULE" not in canonical_json_bytes(wire)
                    assert {document["content"] for document in wire["instructions"]["documents"] if "content" in document} == {
                        "accepted Workspace body", "accepted custom body",
                    }
    finally:
        await assets.close()


@pytest.mark.parametrize("outcome", ("accepted", "unknown", "cancelled", "rejected", "changed_semantics", "changed_bytes"))
async def test_admission_reference_buffer_keeps_exact_prefix_and_uncertain_readback(outcome: str) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Buffer(_Buffer):
        async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
            if outcome == "cancelled":
                entered.set()
                await release.wait()
            return await super().append(scope, id, key, content, wait=wait)

    backend = Buffer(max_bytes=0 if outcome == "rejected" else 1024 * 1024)
    backend.fail_append = outcome == "unknown"
    owner = _ExecutionPayloadBuffer(backend)
    async with _assets() as assets, Persistence.open("admission-buffer", storage=PersistenceStorage.in_memory()) as state:
        await assets.put(AssetKey("rule", "admission.md"), b"ORIGINAL_BUFFER_RULE")
        instructions = await _rules(assets)
        reference = await _archive(state, {"rules": assets}).prepare_repository_instruction_payload(
            instructions, source_domain=PersistenceDomain.EXECUTION)
        capture = _ExecutionPayloadCapture(replace(_execution(), repository_instructions=reference))
        if outcome == "rejected":
            with pytest.raises(AIError):
                await owner.append(capture)
            assert owner.failed(capture.execution.execution_id)
            owner.discard(capture.execution.execution_id)
            await owner.close()
            return
        if outcome == "cancelled":
            pending = asyncio.create_task(owner.append(capture))
            await asyncio.wait_for(entered.wait(), 5)
            pending.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            await owner.append(capture)
        if outcome.startswith("changed_"):
            changed = reference.payload.decode()
            if outcome == "changed_semantics":
                changed["semantic_digest"] = "0" * 64
            else:
                changed["instructions"]["documents"][0]["content_digest"] = "0" * 64
            await owner.append(replace(capture, execution=replace(capture.execution,
                repository_instructions=replace(reference, payload=StoredPayload.inline_json(changed)), revision=2)))
            with pytest.raises(AIError) as failure:
                await owner.read(capture.execution.execution_id)
            assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            owner.discard(capture.execution.execution_id)
        else:
            observed, ids = await owner.read(capture.execution.execution_id, seal=True)
            assert observed == capture
            assert b"ORIGINAL_BUFFER_RULE" not in backend.admissions[0][3]
            assert RepositoryInstructionPayload.from_reference(observed.execution.repository_instructions).semantic_digest == canonical_sha256(instructions.to_payload())
            owner.confirm(capture.execution.execution_id, ids)
        await owner.close()


@pytest.mark.parametrize("source, target", (
    (PersistenceDomain.EXECUTION, PersistenceDomain.CHECKPOINT),
    (PersistenceDomain.CHECKPOINT, PersistenceDomain.EXECUTION),
))
async def test_instruction_payload_cannot_change_its_admission_or_checkpoint_owner(
    source: PersistenceDomain, target: PersistenceDomain,
) -> None:
    async with Persistence.open("instruction-domains", storage=PersistenceStorage.in_memory()) as state:
        accepted = RepositoryInstructions((RepositoryInstructionDocument("rule:custom", ".", "accepted"),))
        archive = _archive(state)
        reference = await archive.prepare_repository_instruction_payload(accepted, source_domain=source)
        with pytest.raises(AIError) as wrong_expected:
            await archive.load_repository_instruction_payload(reference, source_domain=target)
        assert wrong_expected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as moved:
            await archive.load_repository_instruction_payload(replace(reference, source_domain=target))
        assert moved.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
