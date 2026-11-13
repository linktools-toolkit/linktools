#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal control updates retain fixed sources without consuming their bytes."""

from copy import deepcopy
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, StopReason, UsageMetrics
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice, NativeTextReference, NativeUserInputReference
from linktools.ai.persistence._repository_common import project_record
from linktools.ai.persistence.codec import native_text_reference, native_user_input_reference
from linktools.ai.persistence.contracts import ExecutionRecord, ResultRecord
from linktools.ai.persistence.state import StoredRecord
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload

from .test_step_projection import _execution


def _terminal() -> tuple[ExecutionRecord, StoredRecord]:
    initial = _execution()
    value = replace(initial, status=ExecutionStatus.SUCCEEDED, completed_at=initial.updated_at,
        result=ResultRecord(StoredPayload.inline_json("answer"), StopReason.END_TURN, UsageMetrics(), initial.updated_at))
    record = project_record(namespace="test", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
        kind="execution", identity=(value.execution_id,), value=value)
    data = deepcopy(dict(record.data))
    fields = data["value"]["payload"]["fields"]
    root = NativeMessageRoot("source", native_store_key("test", "tenant"),
        (NativeMessageSlice("a" * 64, 1, 0, 1),))
    fields["stored_user_input"]["fields"] = {"content": NativeUserInputReference(root, 0).to_json()}
    fields["result"]["fields"]["output"]["fields"] = {"content": NativeTextReference(root).to_json()}
    assert native_user_input_reference(fields["stored_user_input"]) is not None
    assert native_text_reference(fields["result"]["fields"]["output"]) is not None
    return value, replace(record, data=data)


@pytest.mark.asyncio
async def test_unchanged_native_terminal_control_does_not_read_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    value, current = _terminal()
    changed = deepcopy(dict(current.data))
    changed["value"]["payload"]["fields"]["retention_closed"] = True
    candidate = replace(current, data=changed, storage_revision=current.storage_revision + 1)
    content = ExecutionContent(InMemoryStateStore(), namespace="test", tenant_id="tenant")

    async def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("control-only update consumed an unchanged Native source")

    monkeypatch.setattr(content, "read_payloads", unexpected)
    assert await content.terminal_matches(current, candidate, replace(value, retention_closed=True))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_slot", ("root", "part", "result"))
async def test_changed_native_terminal_uses_previous_source_validation(
    monkeypatch: pytest.MonkeyPatch, changed_slot: str,
) -> None:
    value, current = _terminal()
    changed = deepcopy(dict(current.data))
    fields = changed["value"]["payload"]["fields"]
    if changed_slot == "root":
        fields["result"]["fields"]["output"]["fields"]["content"]["$native_text"]["group_id"] = "other"
    elif changed_slot == "part":
        fields["stored_user_input"]["fields"]["content"]["part_index"] = 1
    else:
        fields["result"]["fields"]["output"]["fields"]["content"] = "changed output"
    candidate = replace(current, data=changed)
    content = ExecutionContent(InMemoryStateStore(), namespace="test", tenant_id="tenant")
    reads = []

    async def unavailable(records: tuple[StoredRecord, ...]) -> None:
        reads.append(records)
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    monkeypatch.setattr(content, "read_payloads", unavailable)
    with pytest.raises(AIError) as raised:
        await content.terminal_matches(current, candidate, value)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert reads == [(current,)]
