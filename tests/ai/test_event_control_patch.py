#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Event publication preserves content owners and atomic fact/CAS boundaries."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import StoredUserInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence.codec import decode_envelope
from linktools.ai.persistence.contracts import ExecutionEventAppend
from linktools.ai.storage import StoredPayload

from .test_runtime_execution_query import _record
from .test_runtime_persistence_evidence import _result


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("outcome", ("success", "conflict", "rollback"))
async def test_event_control_patch_preserves_owners_and_atomicity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, outcome: str,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "events.db")
    async with Persistence.open("event-control", tenant_id="tenant", storage=storage) as state:
        repository, events = state.execution.executions, state.execution.events
        base = _record("execution", agent_id="agent")
        original = replace(base, completed_at=base.updated_at,
            stored_user_input=StoredUserInput("text", StoredPayload.inline_text("input " * 10000)),
            result=replace(_result(base.execution_id, "inline", base.updated_at),
                           output=StoredPayload.inline_json({"text": "output " * 10000})))
        await repository.create(original)
        store, key = repository.state_store, repository._key("execution", (original.execution_id,))
        before = await store.read(lambda tx: tx.get_record(key))
        assert before is not None
        writes = []

        async def reject_bodies(*args, **kwargs):
            raise AssertionError("event append restored payload bodies")

        async def mutate(tx):
            transaction_type = type(tx)
            if outcome == "conflict":
                async def reject_cas(self, record, *, expected_storage_revision):
                    writes.append(record)
                    assert expected_storage_revision == before.storage_revision
                    return False
                monkeypatch.setattr(transaction_type, "replace_record", reject_cas)
            if outcome == "rollback":
                insert = transaction_type.insert_facts

                async def insert_then_fail(self, facts):
                    await insert(self, facts)
                    writes.extend(facts)
                    assert (await self.get_record(key)).storage_revision == before.storage_revision + 1
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                monkeypatch.setattr(transaction_type, "insert_facts", insert_then_fail)
            return await events.append_many_in_transaction(tx, original.execution_id, tenant_id="tenant",
                events=(ExecutionEventAppend("audit", {"value": 1}), ExecutionEventAppend("audit", {"value": 2})),
                expected_event_seq=0)

        with monkeypatch.context() as context:
            context.setattr(ExecutionContent, "read_payloads", reject_bodies)
            if outcome == "success":
                appended = await store.mutate(mutate)
                assert [event.event_seq for event in appended] == [1, 2]
            else:
                with pytest.raises(AIError) as raised:
                    await store.mutate(mutate)
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT and writes
        after = await store.read(lambda tx: tx.get_record(key))
        assert after is not None
        facts = await events.list(original.execution_id, tenant_id="tenant", after_event_seq=0, limit=10)
        if outcome != "success":
            assert after == before and not facts.items
            return
        before_fields = dict(decode_envelope(before.data).value["payload"]["fields"])
        after_fields = dict(decode_envelope(after.data).value["payload"]["fields"])
        for name in ("event_seq", "revision", "updated_at"):
            before_fields.pop(name)
            after_fields.pop(name)
        assert before_fields == after_fields
        assert after.storage_revision == before.storage_revision + 1 and len(facts.items) == 2
        restored = await repository.get_fresh(original.execution_id, tenant_id="tenant")
        assert restored == replace(original, event_seq=2, revision=2, updated_at=restored.updated_at)
