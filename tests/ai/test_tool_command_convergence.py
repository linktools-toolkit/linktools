#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outer transaction ownership tests for ToolOperation commands."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from linktools.ai.core import ToolOperationStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationRecord
from linktools.ai.runtime.state import RuntimeStateCommands, ToolOperationAdmission
from linktools.ai.storage import StoredPayload


def _record(
    *,
    status: ToolOperationStatus = ToolOperationStatus.CLAIMED,
    owner: str = "owner",
    fence: int = 1,
) -> ToolOperationRecord:
    now = datetime.now(timezone.utc)
    return ToolOperationRecord(
        "operation",
        "tenant",
        "step",
        "call",
        canonical_sha256({"call": "call"}),
        "tool",
        canonical_sha256({"args": True}),
        canonical_sha256({"binding": True}),
        True,
        status,
        owner,
        fence,
        now + timedelta(seconds=60) if status is ToolOperationStatus.CLAIMED else None,
        None,
        now,
        now,
    )


def _admission() -> ToolOperationAdmission:
    return ToolOperationAdmission(
        tenant_id="tenant",
        tool_operation_id="operation",
        step_run_id="step",
        recovery_step_run_id=None,
        tool_call_id="call",
        idempotency_key_digest=canonical_sha256({"call": "call"}),
        tool_name="tool",
        arguments_digest=canonical_sha256({"args": True}),
        binding_digest=canonical_sha256({"binding": True}),
        replay_safe=True,
        owner="owner",
        lease_seconds=60,
    )


class _AdmissionTransaction:
    def __init__(self) -> None:
        self.value = object()

    def transaction(self, store: object) -> object:
        del store
        return self.value


class _AdmissionGroup:
    def __init__(self) -> None:
        self.attempts = 0

    async def mutate(self, stores, callback):
        del stores
        self.attempts += 1
        if self.attempts == 1:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await callback(_AdmissionTransaction())


class _AdmissionStateStore:
    def __init__(self, group: _AdmissionGroup) -> None:
        self.storage_group = group


class _AdmissionTools:
    def __init__(self, current: ToolOperationRecord) -> None:
        self.current = current
        self.group = _AdmissionGroup()
        self.state_store = _AdmissionStateStore(self.group)

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        return self.current

    async def admit_in_transaction(
        self,
        transaction: object,
        request: ToolOperationAdmission,
    ) -> ToolOperationRecord:
        del transaction, request
        return self.current


@pytest.mark.asyncio
async def test_tool_admission_conflict_reenters_fresh_repository_attempt() -> None:
    current = _record()
    tools = _AdmissionTools(current)
    commands = object.__new__(RuntimeStateCommands)
    commands._tools = tools

    result = await commands.commit_tool_admission(_admission())

    assert result == current
    assert tools.group.attempts == 2


class _GroupTransaction:
    def __init__(self) -> None:
        self.value = object()

    def transaction(self, store: object) -> object:
        del store
        return self.value


class _CancellationGroup:
    def __init__(self) -> None:
        self.attempts = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def mutate(self, stores, callback):
        del stores
        self.attempts += 1
        if self.attempts == 1:
            self.started.set()
            await self.release.wait()
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return await callback(_GroupTransaction())


class _StateStore:
    def __init__(self, group: _CancellationGroup) -> None:
        self.storage_group = group


class _TerminalTools:
    def __init__(self) -> None:
        self.current = _record()
        self.group = _CancellationGroup()
        self.state_store = _StateStore(self.group)

    async def get_operation(
        self,
        tool_operation_id: str,
        *,
        tenant_id: str,
    ) -> ToolOperationRecord:
        del tool_operation_id, tenant_id
        return self.current

    async def complete_in_transaction(
        self,
        transaction: object,
        tool_operation_id: str,
        *,
        tenant_id: str,
        owner: str,
        fence: int,
        result_payload: StoredPayload,
    ) -> ToolOperationRecord:
        del transaction, tool_operation_id, tenant_id
        self.current = replace(
            self.current,
            status=ToolOperationStatus.COMPLETED,
            owner=owner,
            fence=fence,
            lease_expires_at=None,
            result_payload=result_payload,
        )
        return self.current

    async def fail_in_transaction(self, *args, **kwargs) -> ToolOperationRecord:
        del args, kwargs
        raise AssertionError("failure path is not expected")


@pytest.mark.asyncio
async def test_tool_terminal_cancellation_finishes_durable_retry_before_propagating() -> None:
    tools = _TerminalTools()
    commands = object.__new__(RuntimeStateCommands)
    commands._tools = tools
    commands._background_tasks = set()
    payload = StoredPayload.inline_bytes(b"result")

    task = asyncio.create_task(
        commands.commit_tool_terminal(
            "operation",
            tenant_id="tenant",
            owner="owner",
            fence=1,
            result_payload=payload,
        )
    )
    await tools.group.started.wait()
    task.cancel()
    tools.group.release.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert tools.group.attempts == 2
    assert tools.current.status is ToolOperationStatus.COMPLETED
    assert tools.current.result_payload == payload
