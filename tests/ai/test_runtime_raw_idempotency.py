#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Raw replay identities preserve scope, byte bounds and diagnostic privacy."""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import NAMESPACE_URL, UUID, uuid5

import pytest

from linktools.ai.core import (
    ApprovalStatus, ExternalCallStatus, IdempotencyStatus, OperationStatus,
    ResourceKind, canonical_json_bytes, canonical_sha256, deterministic_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import IdempotencyRecord, IdempotencyTerminalUpdate, OperationTerminalUpdate
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._repository_common import domain_data
from linktools.ai.persistence.codec import _decode_enveloped_domain, decode_envelope, encode_envelope
from linktools.ai.persistence.contracts import ApprovalRecord, ExternalCallRecord, ExecutionEventRecord
from linktools.ai.persistence.state import record_key
from linktools.ai.runtime._execution import DefaultExecutionService, ExecutionStartIdentity


NOW = datetime(2026, 11, 11, tzinfo=timezone.utc)
KEY = "界" * 84 + ":key"


def _record(scope: str = "execution.run", key: str = KEY) -> IdempotencyRecord:
    return IdempotencyRecord(scope, key, "a" * 64, ResourceKind.EXECUTION, "execution",
        IdempotencyStatus.RESERVED, None, None, NOW, NOW)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_raw_replay_keys_preserve_maximum_bytes_and_component_boundaries(
    backend: str, tmp_path: Path,
) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    namespace, tenant = "n" * 256, "t" * 128
    records = (_record("s" * 128), _record("a:b", "c"), _record("a", "b:c"))
    assert len(KEY.encode("utf-8")) == 256
    async with Persistence.open(namespace, tenant_id=tenant, storage=storage) as state:
        repository = state.execution.idempotency
        for item in records:
            assert await repository.reserve(item) == item
            assert await repository.reserve(item) == item
        for item in records:
            assert await repository.get(item.scope, item.idempotency_key, tenant_id=tenant) == item
            assert await repository.get(item.scope, item.idempotency_key, tenant_id="other") is None
        with pytest.raises(AIError) as conflict:
            await repository.reserve(replace(records[0], request_digest="b" * 64))
        assert conflict.value.code is ErrorCode.STORAGE_CONFLICT
        assert await repository.get(records[0].scope, KEY, tenant_id=tenant) == records[0]
        with pytest.raises(AIError) as invalid:
            await repository.get(records[0].scope, KEY + "x", tenant_id=tenant)
        assert invalid.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID
        address = record_key(namespace, tenant, "execution", "idempotency", (records[0].scope, KEY))
        assert len(address) <= 1152 and KEY.encode("utf-8") in address


@pytest.mark.parametrize("value", (
    _record(),
    IdempotencyTerminalUpdate("scope", KEY, IdempotencyStatus.STARTED,
        IdempotencyStatus.COMPLETED, "a" * 64, "b" * 64, None),
    ApprovalRecord("approval", "execution", ApprovalStatus.PENDING, KEY, None, None, None, NOW, None),
    ExternalCallRecord("call", "execution", ExternalCallStatus.PENDING, KEY, NOW, None),
))
def test_raw_replay_wire_roundtrips_and_invalid_stored_keys_fail_closed(value: object) -> None:
    assert KEY not in repr(value)
    encoded = domain_data(value)
    assert _decode_enveloped_domain(encoded, type(value)) == value
    payload = deepcopy(decode_envelope(encoded).value)
    fields = payload["payload"] if isinstance(value, IdempotencyTerminalUpdate) else payload["payload"]["fields"]
    assert fields["idempotency_key"] == KEY
    for key in (KEY + "x", "", "bad\x00key", " key", "\ud800"):
        with pytest.raises(ValueError):
            replace(value, idempotency_key=key)
        fields["idempotency_key"] = key
        if key == "\ud800":
            # Invalid UTF-8 cannot reach a current persisted envelope writer.
            continue
        with pytest.raises(AIError) as corrupt:
            _decode_enveloped_domain(encode_envelope(payload), type(value))
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_raw_execution_keys_are_hidden_in_claims_and_event_repr() -> None:
    values = (
        ExecutionStartIdentity("scope", KEY, "a" * 64),
        OperationTerminalUpdate(KEY, OperationStatus.RUNNING, OperationStatus.SUCCEEDED, None, None, None),
        ExecutionEventRecord("execution", 1, "execution.cancel_requested", {"operation_id": KEY}),
    )
    assert all(KEY not in repr(value) for value in values)


@pytest.mark.asyncio
async def test_remote_turn_lookup_compacts_scope_and_key_without_delimiter_collisions() -> None:
    repository = SimpleNamespace(get=AsyncMock(return_value=None))
    coordination = SimpleNamespace(supports_pending_views=True, supports_request_lookup=True,
        read_request_identity_version=AsyncMock(return_value=2))
    service = object.__new__(DefaultExecutionService)
    service._state = SimpleNamespace(idempotency=repository)
    service._runtime_bridge = SimpleNamespace(pending_turns=SimpleNamespace(
        peek_request=Mock(return_value=None), coordination=coordination))
    pairs = (("a:b", "c"), ("a", "b:c"), ("s" * 128, KEY))
    for scope, key in pairs:
        assert await service._request_identity(scope, key, tenant_id="tenant", include_remote=True) == (2, None)
    requests = [call.args[0] for call in coordination.read_request_identity_version.call_args_list]
    assert requests == [canonical_sha256(list(pair)) for pair in pairs]
    assert len(set(requests)) == len(pairs)
    assert all(len(key) <= 385 and len("request:" + key) <= 393 for key in requests)


@pytest.mark.parametrize("parts", ((None,), ("é", "e\u0301"), ([1, 2],), ({"z": 1, "a": None},)))
def test_generated_uuid_uses_one_canonical_composite_identity(parts: tuple) -> None:
    identity = deterministic_id(*parts)
    assert UUID(identity).version == 5 and len(identity) == 36
    assert identity == str(uuid5(NAMESPACE_URL, canonical_json_bytes(parts).decode("utf-8")))
    assert deterministic_id(None) != deterministic_id("None")
    assert deterministic_id("a", "b") != deterministic_id("b", "a")
    assert deterministic_id("a:b", "c") != deterministic_id("a", "b:c")
    assert deterministic_id({"a": 1, "b": 2}) == deterministic_id({"b": 2, "a": 1})


@pytest.mark.asyncio
async def test_cancellation_keeps_raw_operation_key_out_of_repr_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from linktools.ai.execution import _management
    from .test_pending_session_projection import _runtimes

    logs: list[str] = []

    def record(message: str, *args: object, **kwargs: object) -> None:
        logs.append(message % args)

    monkeypatch.setattr(_management, "_logger", SimpleNamespace(
        debug=record, info=record, warning=record, error=record, exception=record))
    async with _runtimes(tmp_path, cache=None, shared=False) as (runtimes, entered, finish, block):
        runtime, state = runtimes[0]
        session = await runtime.agents.get().create_session("session")
        block()
        execution = await session.start("question", idempotency_key="start")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await execution.cancel(idempotency_key=KEY)
        finally:
            finish.set()
        receipt = await state.execution.operations.get(KEY, tenant_id=runtime.tenant_id)
        assert receipt is not None and receipt.operation_id == KEY
        assert KEY not in repr(receipt)
        assert all(KEY not in message for message in logs)
