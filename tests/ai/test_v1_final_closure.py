#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adversarial evidence for the final GA V1 closure rules."""

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, step_run_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.runtime.state import FactQuery, StateStepArchive, stream_digest
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_step_envelope,
    decode_domain,
    decode_envelope,
    encode_domain,
    encode_envelope,
    iter_runtime_object_refs,
    parse_envelope,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ExecutionHistoryState,
    ExecutionRecord,
    StoredStepSnapshot,
)
from linktools.ai.task import TaskNode
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord


def _assert_integrity(callback: Callable[[], object]) -> None:
    with pytest.raises(AIError) as raised:
        callback()
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def _assert_unsupported(callback: Callable[[], object]) -> None:
    with pytest.raises(AIError) as raised:
        callback()
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_v1_collections_only_accept_canonical_inverse() -> None:
    value = ("b", "a")
    wire = encode_domain(value)
    assert decode_domain(wire, tuple[str, ...]) == value
    _assert_integrity(lambda: decode_domain(["b", "a"], tuple[str, ...]))
    assert decode_domain({"$tuple": []}, tuple[()]) == ()
    _assert_integrity(lambda: decode_domain({"$tuple": ["a"]}, tuple[str, str]))
    _assert_integrity(
        lambda: decode_domain({"$tuple": ["a", "b", "c"]}, tuple[str, str])
    )
    _assert_integrity(
        lambda: decode_domain(
            {"$tuple": ["a", "b"], "extra": True},
            tuple[str, ...],
        )
    )

    mapping = {"b": 2, "a": 1}
    mapping_wire = encode_domain(mapping)
    assert mapping_wire == {"$mapping": [["a", 1], ["b", 2]]}
    assert decode_domain(mapping_wire, Mapping[str, int]) == mapping
    _assert_integrity(
        lambda: decode_domain(
            {"$mapping": [["b", 2], ["a", 1]]},
            Mapping[str, int],
        )
    )
    _assert_integrity(
        lambda: decode_domain(
            {"$mapping": [["a", 1], ["a", 2]]},
            Mapping[str, int],
        )
    )
    _assert_integrity(lambda: decode_domain({"a": 1}, Mapping[str, int]))
    _assert_integrity(lambda: decode_domain({"$mapping": [["a"]]}, Mapping[str, int]))

    frozen = frozenset(("b", "a"))
    frozen_wire = encode_domain(frozen)
    assert frozen_wire == {"$frozenset": ["a", "b"]}
    assert decode_domain(frozen_wire, frozenset[str]) == frozen
    _assert_integrity(
        lambda: decode_domain(
            {"$frozenset": ["b", "a"]},
            frozenset[str],
        )
    )
    _assert_integrity(
        lambda: decode_domain(
            {"$frozenset": ["a", "a"]},
            frozenset[str],
        )
    )
    _assert_integrity(lambda: decode_domain(["a", "b"], frozenset[str]))
    _assert_integrity(
        lambda: decode_domain(
            {"$frozenset": [1, 1.0]},
            frozenset[Any],
        )
    )
    _assert_integrity(
        lambda: decode_domain(
            {"$mapping": [[1, "a"], [1.0, "b"]]},
            Mapping[Any, Any],
        )
    )


def test_v1_set_and_scalar_wrappers_have_no_alternate_reader() -> None:
    with pytest.raises(TypeError):
        encode_domain({"value"})
    _assert_unsupported(lambda: decode_domain({"$set": []}, set[str]))

    _assert_integrity(lambda: decode_domain("open", ExecutionHistoryState))
    _assert_integrity(
        lambda: decode_domain(
            {"$enum": "execution_history_state", "value": "open", "extra": 1},
            ExecutionHistoryState,
        )
    )
    _assert_integrity(lambda: decode_domain("2025-01-01T00:00:00+00:00", datetime))
    _assert_integrity(
        lambda: decode_domain(
            {"$datetime": "2025-01-01T00:00:00+00:00", "extra": 1},
            datetime,
        )
    )
    _assert_integrity(lambda: decode_domain("YQ==", bytes))
    _assert_integrity(lambda: decode_domain({"$bytes": "bad!"}, bytes))
    _assert_integrity(lambda: decode_domain(None, datetime))


def test_v1_dataclass_and_any_readers_require_exact_shapes() -> None:
    cursor = ConversationCursor("run")
    wire = encode_domain(cursor)
    fields = dict(wire["fields"])
    fields.pop("history_id")
    _assert_integrity(
        lambda: decode_domain(
            {"$dataclass": "conversation_cursor", "fields": fields},
            ConversationCursor,
        )
    )
    fields = dict(wire["fields"])
    fields["unknown"] = None
    _assert_integrity(
        lambda: decode_domain(
            {"$dataclass": "conversation_cursor", "fields": fields},
            ConversationCursor,
        )
    )
    _assert_integrity(
        lambda: decode_domain(
            {"$dataclass": "conversation_cursor", "fields": wire["fields"], "extra": 1},
            ConversationCursor,
        )
    )

    task = TaskNode("node", ("dependency",), input={"value": 1}, budget_cost=2)
    task_wire = encode_domain(task)
    task_fields = dict(task_wire["fields"])
    task_fields.pop("budget_cost")
    _assert_integrity(
        lambda: decode_domain(
            {"$dataclass": "task_node", "fields": task_fields},
            TaskNode,
        )
    )
    task_fields = dict(task_wire["fields"])
    task_fields["extra"] = None
    _assert_integrity(
        lambda: decode_domain(
            {"$dataclass": "task_node", "fields": task_fields},
            TaskNode,
        )
    )
    _assert_integrity(lambda: decode_domain({"plain": 1}, Any))
    _assert_integrity(lambda: decode_domain({"$tuple": [], "$mapping": []}, Any))
    assert decode_domain(encode_domain({"value": 1}), Any) == {"value": 1}


def test_v1_envelopes_and_step_wire_have_exact_ownership() -> None:
    cursor = ConversationCursor("run")
    envelope = encode_envelope(
        {"type": wire_type_id(cursor), "payload": encode_domain(cursor)}
    )
    assert decode_envelope(envelope).version == 1
    _assert_integrity(
        lambda: parse_envelope({"v": 1, "value": envelope["value"], "extra": 1})
    )
    _assert_integrity(lambda: parse_envelope({"value": envelope["value"]}))
    _assert_integrity(
        lambda: _decode_enveloped_domain(
            {
                "v": 1,
                "value": {
                    "type": "conversation_cursor",
                    "payload": encode_domain(cursor),
                    "extra": 1,
                },
            },
            ConversationCursor,
        )
    )
    _assert_integrity(
        lambda: _decode_enveloped_domain(
            {"v": 1, "value": {"payload": encode_domain(cursor)}},
            ConversationCursor,
        )
    )

    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    stored = StoredStepSnapshot("run", 1, timestamp, "complete", "d" * 64)
    assert _decode_step_envelope(_encode_step_envelope(stored)) == stored
    _assert_integrity(
        lambda: _decode_step_envelope(
            {
                "v": 1,
                "value": {"type": "stored_step_snapshot", "payload": {}, "extra": 1},
            }
        )
    )
    _assert_unsupported(
        lambda: _decode_step_envelope(
            {"v": 1, "value": {"type": "continuable_snapshot", "payload": {}}}
        )
    )
    with pytest.raises(TypeError, match="unsupported domain type: ContinuableSnapshot"):
        wire_type_id(ContinuableSnapshot)
    with pytest.raises(TypeError):
        _encode_step_envelope(
            ContinuableSnapshot("run", 1, [], None, None, None, timestamp)
        )


def test_object_ref_traversal_rejects_raw_payload_representation() -> None:
    raw_payload = {
        "kind": "object",
        "encoding": None,
        "digest": "a" * 64,
        "size": 1,
        "ref": {"store_id": "store", "key": "key"},
    }
    with pytest.raises(AIError) as raised:
        tuple(
            iter_runtime_object_refs(
                raw_payload,
                default_domain=RuntimeDomain.EXECUTION,
            )
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_stored_snapshot_is_durable_authority_across_reopen(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="tenant",
        session_id=None,
        binding_digest="binding",
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
    )
    run_id = step_run_id(
        namespace="closure-reopen",
        tenant_id="tenant",
        execution_id="execution",
        segment_sequence=1,
    )
    run = RunRecord(
        run_id=run_id,
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        metadata={"segment_sequence": "1"},
        started_at=now,
    )
    snapshot = ContinuableSnapshot(
        run_id=run_id,
        step_index=1,
        messages=[ModelRequest(parts=[UserPromptPart(content="hello")])],
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        timestamp=now,
    )
    root = tmp_path / "runtime"
    state = RuntimeState.filesystem(root)
    await state.initialize(namespace="closure-reopen", tenant_id="tenant")
    try:
        await state.execution.executions.create_with_history_head(execution)
        archive = state.steps.read_store(RuntimeDomain.EXECUTION)
        assert isinstance(archive, StateStepArchive)
        await archive.materialize_snapshot(run, snapshot, execution_id="execution")
        facts = await archive.state_store.read(
            lambda transaction: transaction.list_facts(
                FactQuery(
                    stream_digest(
                        "closure-reopen",
                        "tenant",
                        "execution",
                        "snapshot",
                        run_id,
                    ),
                    latest=True,
                )
            )
        )
        assert len(facts) == 1
        assert facts[0].data["value"]["type"] == "stored_step_snapshot"
        assert "messages" not in facts[0].data["value"]["payload"]
    finally:
        await state.close()

    reopened = RuntimeState.filesystem(root)
    await reopened.initialize(namespace="closure-reopen", tenant_id="tenant")
    try:
        archive = reopened.steps.read_store(RuntimeDomain.EXECUTION)
        assert isinstance(archive, StateStepArchive)
        assert (
            await archive.latest_snapshot(
                run_id=run_id,
                include_interrupted=True,
            )
            == snapshot
        )
    finally:
        await reopened.close()
