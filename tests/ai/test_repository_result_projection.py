#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Result-only reads retain result integrity without executable dependencies."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence.state import RecordQuery

from .test_execution_result_read_io import _seed

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("corruption", ("binding", "input", "result", "result_source"))
async def test_result_dependency_boundary(corruption: str) -> None:
    async with Persistence.open("result-dependencies", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        original = await _seed(state)
        repository = state.execution.executions
        if corruption == "binding":
            _, archive = await repository.state_store.storage_group.open_archive()
            rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
            binding = next(row for row in rows if row.data["codec"] == "execution-binding")
            await archive.mutate(lambda tx: tx.delete_record(binding.key))
        else:
            async def corrupt(tx):
                row = await tx.get_record(repository.execution_record_key(original.execution_id))
                data = deepcopy(dict(row.data))
                fields = data["value"]["payload"]["fields"]
                if corruption == "input":
                    fields["stored_user_input"]["fields"]["content"]["$execution_payload"] = "execution-payload:" + "0" * 64
                elif corruption == "result_source":
                    fields["result"]["fields"]["content"]["$execution_payload"] = "execution-payload:" + "0" * 64
                else:
                    fields["result"] = "invalid"
                await tx.delete_record(row.key)
                await tx.insert_record(replace(row, data=data))
            await repository.state_store.mutate(corrupt)
        if corruption in {"result", "result_source"}:
            with pytest.raises(AIError) as raised:
                await repository.get_result(original.execution_id, tenant_id="tenant")
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            assert await repository.get_result(original.execution_id, tenant_id="tenant") == original.result
            with pytest.raises(AIError):
                await repository.get(original.execution_id, tenant_id="tenant")
        assert await repository.get_result(original.execution_id, tenant_id="other") is None
        assert await repository.get_result("absent", tenant_id="tenant") is None


async def test_custom_result_repository_keeps_full_read_override() -> None:
    class CustomRepository(ExecutionRepositoryImpl):
        pass
    repository = object.__new__(CustomRepository)
    result = object()
    repository.get = AsyncMock(return_value=SimpleNamespace(result=result))
    repository.get_result_record = AsyncMock(side_effect=AssertionError("custom source bypassed"))
    assert await repository.get_result("execution", tenant_id="tenant") is result
    repository.get.assert_awaited_once_with("execution", tenant_id="tenant")


@pytest.mark.parametrize("status", ("SUCCEEDED", "FAILED", "CANCELLED"))
async def test_direct_result_keeps_terminal_outcome(status: str) -> None:
    from linktools.ai.core import ExecutionStatus
    async with Persistence.open("terminal-result", storage=PersistenceStorage.in_memory(), tenant_id="tenant") as state:
        original = await _seed(state)
        repository = state.execution.executions
        if status != "SUCCEEDED":
            result = replace(original.result, output=None)
            current = replace(original, status=ExecutionStatus[status], result=result,
                error_code=(ErrorCode.EXECUTION_CANCELLED if status == "CANCELLED" else ErrorCode.INTERNAL_ERROR).value,
                revision=original.revision + 1)
            await repository.compare_and_swap(original.execution_id, tenant_id="tenant",
                expected_revision=original.revision, next_record=current)
        else:
            result = original.result
        assert await repository.get_result(original.execution_id, tenant_id="tenant") == result
