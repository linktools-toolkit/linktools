#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured attachment fact query regressions."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionStatus,
    HmacCursorSigner,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.runtime._history import StepExecutionHistoryReader
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
    StoredUserInput,
)
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.storage import StoredPayload


def _fact(
    *,
    fact: str,
    attachment_id: str,
    digest: str,
    position: int,
    source: str = "binary",
    call_id: str | None = None,
) -> dict[str, object]:
    return {
        "fact": fact,
        "attachment_id": attachment_id,
        "source": source,
        "media_type": "image/png",
        "size": 4,
        "digest": digest,
        "content_key": digest,
        "position": position,
        "call_id": call_id,
    }


def _interaction(
    run_id: str,
    sequence: int,
    attachments: tuple[dict[str, object], ...],
) -> ModelInteractionRecord:
    return ModelInteractionRecord(
        run_id=run_id,
        step_index=sequence,
        request_sequence=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=ContextProjection(()),
        request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b"{}"),
            RuntimeDomain.EXECUTION,
        ),
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=1,
        usage=None,
        attachments=attachments,  # type: ignore[arg-type]
    )


class _Executions:
    def __init__(self, record: object) -> None:
        self.record = record

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.record
        return None


class _Store:
    def __init__(self, run: RunRecord, interactions: list[ModelInteractionRecord]) -> None:
        self.run = run
        self.interactions = interactions

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        return self.run if run_id == self.run.run_id else None

    async def list_model_interactions(
        self,
        *,
        run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[object]:
        assert run_id == self.run.run_id
        values = [
            value
            for value in self.interactions
            if after_request_sequence is None
            or value.request_sequence > after_request_sequence
        ]
        return values if limit is None else values[:limit]


@pytest.mark.asyncio
async def test_attachment_fact_cursor_freezes_model_request_high_water() -> None:
    namespace = "attachment-history"
    tenant_id = "tenant"
    execution_id = "execution"
    run_id = step_run_id(
        namespace=namespace,
        tenant_id=tenant_id,
        execution_id=execution_id,
        segment_sequence=1,
    )
    initial_id = "a" * 64
    initial_digest = "b" * 64
    tool_id = "c" * 64
    tool_digest = "d" * 64
    late_id = "e" * 64
    late_digest = "f" * 64

    stored_input = StoredUserInput(
        "user-content-v1",
        StoredPayload.inline_json({"items": []}),
        {
            "version": 1,
            "prompt": {"kind": "items", "items": []},
            "files": [],
            "attachments": [
                _fact(
                    fact="accepted",
                    attachment_id=initial_id,
                    digest=initial_digest,
                    position=0,
                )
            ],
        },
    )
    record = SimpleNamespace(
        execution_id=execution_id,
        status=ExecutionStatus.STARTED,
        binding_kind="agent",
        agent_run_sequence=1,
        stored_user_input=stored_input,
    )
    run = RunRecord(
        run_id=run_id,
        conversation_id=step_conversation_id(
            namespace=namespace,
            tenant_id=tenant_id,
            execution_id=execution_id,
        ),
        metadata={"segment_sequence": "1", "agent_name": "default"},
    )
    interactions = [
        _interaction(
            run_id,
            1,
            (
                _fact(
                    fact="included_in_request",
                    attachment_id=initial_id,
                    digest=initial_digest,
                    position=0,
                ),
            ),
        ),
        _interaction(
            run_id,
            2,
            (
                _fact(
                    fact="accepted",
                    attachment_id=tool_id,
                    digest=tool_digest,
                    position=0,
                    source="attach_files",
                    call_id="call-1",
                ),
                _fact(
                    fact="included_in_request",
                    attachment_id=tool_id,
                    digest=tool_digest,
                    position=1,
                    source="attach_files",
                    call_id="call-1",
                ),
            ),
        ),
    ]
    store = _Store(run, interactions)
    reader = StepExecutionHistoryReader(
        namespace=namespace,
        executions=_Executions(record),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        cursor_signer=HmacCursorSigner(
            "attachment-history",
            b"attachment-history-key",
        ),
    )

    first = await reader.attachment_facts(
        execution_id,
        tenant_id=tenant_id,
        cursor=None,
        limit=2,
    )
    assert [(item.fact, item.attachment_id) for item in first.items] == [
        ("accepted", initial_id),
        ("included_in_request", initial_id),
    ]
    assert first.next_cursor is not None

    interactions.append(
        _interaction(
            run_id,
            3,
            (
                _fact(
                    fact="accepted",
                    attachment_id=late_id,
                    digest=late_digest,
                    position=0,
                    source="attach_files",
                    call_id="call-2",
                ),
                _fact(
                    fact="included_in_request",
                    attachment_id=late_id,
                    digest=late_digest,
                    position=2,
                    source="attach_files",
                    call_id="call-2",
                ),
            ),
        )
    )

    second = await reader.attachment_facts(
        execution_id,
        tenant_id=tenant_id,
        cursor=first.next_cursor,
        limit=10,
    )
    assert [(item.fact, item.attachment_id) for item in second.items] == [
        ("accepted", tool_id),
        ("included_in_request", tool_id),
    ]
    assert second.next_cursor is None
    assert second.items[0].request_sequence is None
    assert second.items[0].step_index is None
    assert second.items[0].call_id == "call-1"
    assert second.items[1].request_sequence == 2
    assert second.items[1].step_index == 2
    assert second.items[1].processing_status == "unknown"

    fresh = await reader.attachment_facts(
        execution_id,
        tenant_id=tenant_id,
        cursor=None,
        limit=10,
    )
    assert len(fresh.items) == 6
    assert fresh.items[-1].attachment_id == late_id
    assert fresh.items[-1].request_sequence == 3
