#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session ownership and the authorized execution actor remain distinct."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import AuthorizationAction, CommitPolicy, ExecutionStatus, Principal, ResourceKind, ResourceRef, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.snapshots import SnapshotLimits, snapshot_manifest_identity
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, InMemoryObjectStore, read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class SharedSessionPolicy(TenantAuthorizationPolicy):
    async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
        if (principal.principal_id == "peer" and principal.tenant_id == resource.tenant_id
                and resource.kind is ResourceKind.SESSION and resource.id == "shared"
                and action is AuthorizationAction.SESSION_READ):
            return
        await super().authorize(principal, action, resource)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_commit", (False, True))
async def test_shared_session_preserves_actor_in_live_and_portable_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_commit: bool,
) -> None:
    entered, finish = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("shared-session")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(
        retain_history=True, **({"commit": CommitPolicy.terminal()} if terminal_commit else {}),
    ), scope="runtime")
    cache = InMemoryCacheBackend(max_bytes=2**24)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache, coordination=cache)
    owner, peer = Principal("owner", "default"), Principal("peer", "default")
    policy = SharedSessionPolicy()
    async with Persistence.open("shared-session", storage=storage) as state, Runtime.attach(
        persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
        authorization=policy,
    ) as runtime:
        session = await runtime.agents.get().create_session("shared", principal=owner)
        for denied in (Principal("stranger", "default"), Principal("peer", "other")):
            with pytest.raises(AIError) as error:
                await session.start("denied", principal=denied)
            assert error.value.code is (ErrorCode.AUTHORIZATION_DENIED if denied.tenant_id == "default" else ErrorCode.STORAGE_OWNER_MISMATCH)
        execution = await session.start("peer turn", principal=peer)
        try:
            await asyncio.wait_for(entered.wait(), 5)
            for actor in (owner, peer):
                assert (await session.inspect(principal=actor)).active_execution_id == execution.execution_id
                page = await session.timeline(principal=actor)
                assert [turn.execution_id for turn in page.items] == [execution.execution_id]
                assert not page.items[0].conversation_committed
                detail = await runtime.sessions.refresh("shared", principal=actor, max_age=60)
                assert detail.session.active_execution_id == execution.execution_id
                assert detail.timeline.items == page.items
                history = await runtime.history.session_detail("shared", principal=actor, max_age=60)
                assert history.timeline.items == page.items
        finally:
            finish.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        record = await state.execution.executions.get(execution.execution_id, tenant_id="default")
        assert record.principal_id == "peer"
        assert (await state.conversation.sessions.get("shared", tenant_id="default")).owner_principal_id == "owner"
        for actor in (owner, peer):
            expected = (await session.timeline(principal=actor)).items
            assert expected[0].conversation_committed
            assert expected[0].status is ExecutionStatus.SUCCEEDED
            cold = await runtime.sessions.refresh("shared", principal=actor, max_age=60)
            assert cold.timeline.items == expected

            async def no_transcript(*args, **kwargs):
                pytest.fail("an authorized cached display must not reload transcript bodies")
                yield

            with monkeypatch.context() as patch:
                patch.setattr(state.run_store.conversation_persistence, "iter_conversation_message_range", no_transcript)
                assert await runtime.sessions.refresh("shared", principal=actor, max_age=60) == cold
            for denied in (Principal("stranger", "default"), Principal("peer", "other")):
                with pytest.raises(AIError) as error:
                    await session.timeline(principal=denied)
                assert error.value.code is ErrorCode.AUTHORIZATION_DENIED

    archive = InMemoryObjectStore()
    limits = SnapshotLimits(max_entries=10000, max_bytes=10000000)
    async with Persistence.open("shared-session", storage=storage, read_only=True) as state:
        async with History.open(persistence=state, authorization=policy) as history:
            for actor in (owner, peer):
                assert (await history.session_timeline("shared", principal=actor)).items == expected
        reference = await state.export_snapshot(object_store=archive, limits=limits)
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(reference, object_store=archive, root=root, limits=limits)
    async with Persistence.open("shared-session", storage=PersistenceStorage.from_root(root), read_only=True) as state:
        assert (await state.execution.executions.get(execution.execution_id, tenant_id="default")).principal_id == "peer"
        assert (await state.conversation.sessions.get("shared", tenant_id="default")).owner_principal_id == "owner"
        async with History.open(persistence=state, authorization=policy) as history:
            for actor in (owner, peer):
                assert (await history.session_timeline("shared", principal=actor)).items == expected
        roundtrip = await state.export_snapshot(object_store=archive, limits=limits)
        original_manifest = json.loads(await read_object(archive, reference.key, expected_digest=reference.digest, expected_size=reference.size))
        restored_manifest = json.loads(await read_object(archive, roundtrip.key, expected_digest=roundtrip.digest, expected_size=roundtrip.size))
        assert snapshot_manifest_identity(restored_manifest) == snapshot_manifest_identity(original_manifest)
        assert restored_manifest["native_messages"][0]["messages"] == original_manifest["native_messages"][0]["messages"]
        assert restored_manifest["native_messages"][0]["sequences"] == original_manifest["native_messages"][0]["sequences"]


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("session", "parent", "completion"))
async def test_shared_session_readers_reject_broken_execution_correlation(
    monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    from dataclasses import replace
    from linktools.ai.core import ExecutionLineageKind, HmacCursorSigner
    from linktools.ai.history._session_timeline import project_session_timeline
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence._snapshot_validation import validate_snapshot_references
    from linktools.ai.persistence.codec import _encode_persisted_domain, encode_envelope
    from .test_session_timeline import _commit_empty_turn, _execution, _session

    async with Persistence.open("shared-correlation", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        await state.conversation.sessions.create(_session())
        value = replace(_execution("execution", status=ExecutionStatus.FAILED, prompt="input"), principal_id="peer")
        await state.execution.executions.create(value)
        await state.conversation.sessions.admit_execution("session", tenant_id="tenant", execution_id=value.execution_id, expected=None)
        await _commit_empty_turn(state, value)
        changes = {"session": {"session_id": "other"}, "parent": {"parent_execution_id": "parent"},
                   "completion": {"completed_at": None, "result": None}}[corruption]
        projected = await state.execution.executions.get_timeline_many((value.execution_id,), tenant_id="tenant")

        async def corrupted(*args, **kwargs):
            return {value.execution_id: replace(projected[value.execution_id], **changes)}

        monkeypatch.setattr(state.execution.executions, "get_timeline_many", corrupted)
        with pytest.raises(AIError) as error:
            await project_session_timeline(
                state.conversation, state.execution.executions, TenantAuthorizationPolicy(),
                HmacCursorSigner("shared", b"shared-correlation"), state.run_store.conversation_persistence,
                "session", principal=Principal("owner", "tenant"),
            )
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        records = {
            domain: await store.read(lambda tx: tx.scan_records())
            for domain, store in ((PersistenceDomain.EXECUTION, state.execution.executions.state_store),
                                  (PersistenceDomain.CONVERSATION, state.conversation.sessions.state_store))
        }
        if corruption == "parent":
            changes.update(lineage_kind=ExecutionLineageKind.SUBAGENT, parent_invocation_id="invocation")
        records[PersistenceDomain.EXECUTION] = tuple(
            replace(record, data=encode_envelope({"type": "execution_record",
                "payload": _encode_persisted_domain(replace(value, **changes))}))
            if record.kind == "execution" else record
            for record in records[PersistenceDomain.EXECUTION]
        )
        with pytest.raises(AIError) as error:
            validate_snapshot_references(records, {})
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
