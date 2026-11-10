#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact checkpoint contexts retain compaction independently of latest indexes."""

import json
from dataclasses import replace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import event

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain
from linktools.ai.persistence._step_archive import StateStepArchive
from linktools.ai.persistence._step_contracts import AgentRunRecord, AgentRunCheckpoint
from linktools.ai.persistence.contracts import ContextProjection
from linktools.ai.session import ConversationCursor
from linktools.ai.storage import InMemoryCacheBackend
from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_session_history_projection import _session
from .test_pending_turn_commit import _turn_state, _pending_turn, _commit

pytestmark = pytest.mark.asyncio


class Cache(InMemoryCacheBackend):
    def __init__(self):
        super().__init__(max_bytes=1024 * 1024)
        self.context_keys = []

    async def put(self, key, content):
        if key.startswith("conversation-context:"):
            self.context_keys.append(key)
        await super().put(key, content)


async def publish(archive, history_id, checkpoint):
    run = (await archive.get_agent_run(agent_run_id=checkpoint.agent_run_id)
           or AgentRunRecord(checkpoint.agent_run_id, metadata={"history_id": history_id}))
    batch = await archive.prepare_checkpoints(run, (checkpoint,))
    sequence = await archive.state_store.mutate(lambda tx: archive.materialize_checkpoint_in_transaction(tx, run, batch[0]))
    await archive.confirm_session_context(checkpoint_sequence=sequence, prepared=batch[0])
    return sequence, batch[0].stored.projection_digest


async def read(archive, history_id, reference, count=2):
    return await archive.load_committed_session_model_context(agent_run_id="run", checkpoint_sequence=reference[0], projection_digest=reference[1], tenant_id="tenant")


async def test_exact_context_cache_matches_cold_and_survives_deferred_latest(tmp_path):
    cache = Cache()
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache))
    await state.initialize(namespace="exact-context", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        summary = ModelRequest(parts=[UserPromptPart("committed summary")])
        checkpoint = replace(_checkpoint(2), context_messages=[summary])
        reference = await publish(archive, session.history_id, checkpoint)
        cursor = ConversationCursor("run", session.history_id, 2, reference[0], reference[1])
        await state.conversation.sessions.admit_execution("session", tenant_id="tenant", execution_id="done", expected=None)
        await state.conversation.sessions.advance_continuation("session", tenant_id="tenant", execution_id="done", expected=None, next_cursor=cursor)
        await state.conversation.sessions.release_execution("session", tenant_id="tenant", execution_id="done")
        before = await state.conversation.sessions.get("session", tenant_id="tenant")
        statements = []
        engine = archive.state_store.storage_group.context.engine.sync_engine
        def capture(_connection, _cursor, statement, _parameters, _context, _many):
            if statement.startswith("SELECT"):
                statements.append(statement)
        event.listen(engine, "before_cursor_execute", capture)
        try:
            hot = await read(archive, session.history_id, reference)
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        assert hot.model_messages() == (summary,)
        assert all(message.source is None for message in hot.messages)
        assert cache.context_keys
        assert not any("archive_sequence" in sql for sql in statements)
        # Cached SDK values are isolated; callers cannot mutate the selected revision.
        hot.messages[0].message.parts[0].content = "caller mutation"
        assert (await read(archive, session.history_id, reference)).model_messages() == (summary,)
        await state.run_store.register_agent_run(AgentRunRecord("deferred", metadata={"history_id": session.history_id}))
        pending = ModelRequest(parts=[UserPromptPart("pending")])
        await state.run_store.save_checkpoint(AgentRunCheckpoint("deferred", 1, [pending], state="interrupted",
            context_messages=[summary, pending], transcript_message_count_before=0))
        await state.run_store.materialize_conversation(agent_run_id="deferred", include_interrupted=True)
        assert await state.conversation.sessions.get("session", tenant_id="tenant") == before
        warm = await read(archive, session.history_id, reference)
        for key in cache.context_keys:
            await cache.delete(key)
        cold = await read(archive, session.history_id, reference)
        assert warm == cold
        assert cold.model_messages() == (summary,)
        assert warm.archive_root == cold.archive_root
        assert warm.archive_frontier == cold.archive_frontier
    finally:
        await state.close()


async def test_exact_context_same_count_fact_and_corrupt_cache(tmp_path):
    cache = Cache()
    state = Persistence(PersistenceStorage.sqlite(tmp_path / "state.db", cache=cache))
    await state.initialize(namespace="exact-context", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        old = ModelRequest(parts=[UserPromptPart("old summary")])
        first = replace(_checkpoint(2), context_messages=[old])
        reference = await publish(archive, session.history_id, first)
        new = ModelRequest(parts=[UserPromptPart("new summary")])
        second = replace(first, context_messages=[new], transcript_message_count_before=2)
        newer = await publish(archive, session.history_id, second)
        assert newer[0] > reference[0]
        assert (await read(archive, session.history_id, reference)).model_messages() == (old,)
        assert (await read(archive, session.history_id, newer)).model_messages() == (new,)
        old_key, new_key = cache.context_keys[-2:]
        payload = json.loads(await cache.get(old_key))
        payload["root"] = json.loads(await cache.get(new_key))["root"]
        await cache.put(old_key, json.dumps(payload).encode())
        recovered = await read(archive, session.history_id, reference)
        fact = await archive.checkpoint_at_sequence(agent_run_id="run", checkpoint_sequence=reference[0])
        assert recovered.archive_root == fact.projection.items[0].root
        assert recovered.model_messages() == (old,)
        # The latest projection is a derived index, not a competing context authority.
        await archive.state_store.mutate(lambda tx: archive.transcript_repository.store_projection(tx, session.history_id, ContextProjection(())))
        assert (await archive.latest_checkpoint(agent_run_id="run")).context_messages == [new]
        await archive.release_agent_run(agent_run_id="run")
        with pytest.raises(AIError) as missing:
            await read(archive, session.history_id, reference)
        assert missing.value.code is ErrorCode.SESSION_HISTORY_UNAVAILABLE
    finally:
        await state.close()


@pytest.mark.parametrize("failure", ("rollback", "unknown", "resolved_unknown"))
async def test_unconfirmed_terminal_does_not_publish_context(tmp_path, monkeypatch, failure):
    async with _turn_state("sqlite", tmp_path) as state:
        turn = _pending_turn(state)
        group = state.storage.execution.executions.state_store.storage_group
        mutate, read_many = type(group).mutate, type(group).read_many
        offered = []
        original_confirm = StateStepArchive.confirm_session_context
        async def confirm(self, *args, **kwargs):
            offered.append(True)
            await original_confirm(self, *args, **kwargs)
        monkeypatch.setattr(StateStepArchive, "confirm_session_context", confirm)
        entered = False
        async def fail(self, stores, callback):
            nonlocal entered
            if self is not group:
                return await mutate(self, stores, callback)
            async def operation(tx):
                result = await callback(tx)
                assert not offered
                if failure == "rollback":
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return result
            result = await mutate(self, stores, operation)
            entered = True
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        async def unavailable(self, stores, callback):
            if self is group and entered and failure == "unknown":
                raise OSError("readback unavailable")
            return await read_many(self, stores, callback)
        with monkeypatch.context() as patch:
            patch.setattr(type(group), "mutate", fail)
            patch.setattr(type(group), "read_many", unavailable)
            if failure == "resolved_unknown":
                await _commit(state, turn)
            else:
                with pytest.raises(AIError):
                    await _commit(state, turn)
        assert not offered


async def test_fork_and_snapshot_keep_exact_summary_after_latest_replacement(tmp_path):
    from linktools.ai.core import Principal
    from linktools.ai.execution import ForkSessionRequest
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore
    from .test_session_history_projection import _service

    principal = Principal("owner", "tenant")
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    async with Persistence.open("exact-fork", storage=storage, tenant_id="tenant") as state:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        summary = ModelRequest(parts=[UserPromptPart("committed summary")])
        reference = await publish(archive, session.history_id, replace(_checkpoint(2), context_messages=[summary]))
        cursor = ConversationCursor("run", session.history_id, 2, reference[0], reference[1])
        await state.conversation.sessions.admit_execution("session", tenant_id="tenant", execution_id="publish", expected=None)
        await state.conversation.sessions.advance_continuation("session", tenant_id="tenant", execution_id="publish", expected=None, next_cursor=cursor)
        await state.conversation.sessions.release_execution("session", tenant_id="tenant", execution_id="publish")
        await archive.materialize_checkpoint(AgentRunRecord("other", metadata={"history_id": session.history_id}),
            AgentRunCheckpoint("other", 1, [], context_messages=[ModelRequest(parts=[UserPromptPart("replacement")])], transcript_message_count_before=0))
        service = _service(state)
        service._transcript_store = state.run_store.conversation_persistence
        await service.fork("agent", "session", ForkSessionRequest(principal, "fork", "fork-operation"))
        assert await service.load_model_context("fork", principal=principal) == (summary,)
        source_root = (await read(archive, session.history_id, reference)).archive_root
    objects = InMemoryObjectStore("exact-context-snapshot")
    limits = SnapshotLimits(max_entries=2000, max_bytes=4 * 1024 * 1024)
    async with Persistence.open("exact-fork", storage=PersistenceStorage.sqlite(tmp_path / "source.db"), tenant_id="tenant", read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=tmp_path / "restored", limits=limits)
    async with Persistence.open("exact-fork", storage=PersistenceStorage.from_root(tmp_path / "restored"), tenant_id="tenant", read_only=True) as state:
        child = await state.conversation.sessions.get("fork", tenant_id="tenant")
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        value = await read(archive, child.history_id, reference)
        assert value.model_messages() == (summary,)
        assert value.archive_root.group_id != source_root.group_id
        assert child.continuation.checkpoint_sequence == reference[0]
        assert child.continuation.projection_digest == reference[1]


async def test_context_cache_preserves_references_and_opaque_tool_values(tmp_path):
    import hashlib
    from pydantic_ai.messages import ToolReturnPart
    from linktools.ai.asset import AssetKey, AssetVersionRef
    from linktools.ai.model import ReferencedContent
    from linktools.ai.model.messages import RetiredReferencedContent, SummaryPromptPart
    from linktools.ai.spec import CapabilityRef
    from linktools.ai.storage import StorageEntryRevision

    reference = ReferencedContent(AssetVersionRef(AssetKey("document", "fixture"), "configured",
        StorageEntryRevision(1), hashlib.sha256(b"fixture").hexdigest(), 7),
        media_type="application/pdf", source=CapabilityRef("capability", "files", 1),
        source_parameters={"version": "fixed"})
    retired = RetiredReferencedContent(reference, "retired attachment")
    opaque = {"type": "reference", "uri": "ordinary application data", "metadata": {"$dataclass": "archive_root"}}
    message = ModelRequest(parts=[SummaryPromptPart("trusted summary"),
        UserPromptPart([reference, retired]),
        ToolReturnPart("read", [reference, opaque], tool_call_id="read-call")])
    cache = Cache()
    async with Persistence.open("reference-context", storage=PersistenceStorage.sqlite(tmp_path / "reference.db", cache=cache), tenant_id="tenant") as state:
        session = await state.conversation.sessions.create(_session())
        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        selected = await publish(archive, session.history_id, replace(_checkpoint(2), context_messages=[message]))
        hot = await read(archive, session.history_id, selected)
        for key in cache.context_keys:
            await cache.delete(key)
        cold = await read(archive, session.history_id, selected)
        assert hot == cold
        assert hot.model_messages() == (message,)
        restored = hot.messages[0].message
        assert type(restored.parts[0]) is SummaryPromptPart
        assert type(restored.parts[1].content[0]) is ReferencedContent
        assert type(restored.parts[1].content[1]) is RetiredReferencedContent
        assert restored.parts[2].content[1] == opaque


async def test_context_cache_put_failure_does_not_fail_confirmed_terminal(tmp_path):
    class Unavailable(Cache):
        async def put(self, key, content):
            raise OSError("cache unavailable")

    async with _turn_state("sqlite", tmp_path) as state:
        archive = state.storage.run_store.read_store(PersistenceDomain.CONVERSATION)
        archive._context_cache = Unavailable()
        turn = _pending_turn(state)
        result = await _commit(state, turn)
        assert result.result == turn.commit.result
        current = await state.storage.conversation.sessions.get("session", tenant_id="tenant")
        assert current.continuation.checkpoint_sequence is not None
