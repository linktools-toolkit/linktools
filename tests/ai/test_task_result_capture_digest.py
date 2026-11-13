#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture identity reuses the digest already checked by the result reader."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from linktools.ai.core import ExecutionStatus, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.history import _history
from linktools.ai.evaluation.engine._service import Evaluations
from linktools.ai.evaluation.engine import _service


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", ("same", "different", "missing"))
async def test_builtin_capture_hashes_result_once(monkeypatch: pytest.MonkeyPatch, expected: str) -> None:
    body = {"large": "x" * 100000, "number": 1.0}
    digest = canonical_sha256(body)
    principal = object()
    history = object.__new__(History)
    history._task_result_record = AsyncMock(return_value=SimpleNamespace(execution_id="execution", result_digest=digest))
    history.result = AsyncMock(return_value=SimpleNamespace(status=ExecutionStatus.SUCCEEDED, output=body))
    evaluation = object.__new__(Evaluations)
    evaluation._history = history
    hashes = []

    def counted(value):
        hashes.append(value)
        return canonical_sha256(value)

    monkeypatch.setattr(_history, "canonical_sha256", counted)
    monkeypatch.setattr(_service, "canonical_sha256", counted)
    selected = digest if expected == "same" else "0" * 64 if expected == "different" else None
    if expected == "same":
        assert await evaluation._capture_task_result("graph", "node", principal=principal, expected_digest=selected) == body
    else:
        with pytest.raises(AIError) as error:
            await evaluation._capture_task_result("graph", "node", principal=principal, expected_digest=selected)
        assert error.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
    assert len(hashes) == (2 if expected == "missing" else 1)
    history._task_result_record.assert_awaited_once_with("graph", "node", principal=principal)
    history.result.assert_awaited_once_with("execution", principal=principal)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("authorization", "recovery", "corrupt", "not_succeeded"))
async def test_expected_digest_keeps_reader_failures(failure: str) -> None:
    history = object.__new__(History)
    body = {"result": 1}
    history._task_result_record = AsyncMock(return_value=SimpleNamespace(execution_id="execution", result_digest=canonical_sha256(body)))
    history.result = AsyncMock(return_value=SimpleNamespace(status=ExecutionStatus.SUCCEEDED, output=body))
    code = ErrorCode.STORAGE_INTEGRITY_ERROR
    if failure == "authorization":
        code = ErrorCode.AUTHORIZATION_DENIED
        history._task_result_record.side_effect = AIError(code)
    elif failure == "recovery":
        code = ErrorCode.STORAGE_RECOVERY_REQUIRED
        history.result.side_effect = AIError(code)
    elif failure == "corrupt":
        history.result.return_value.output = {"result": 2}
    else:
        history.result.return_value.status = ExecutionStatus.FAILED
    with pytest.raises(AIError) as error:
        await history.task_result("graph", "node", principal=object(), expected_digest="0" * 64)
    assert error.value.code is code
    if failure == "authorization":
        history.result.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("matches", (True, False))
async def test_custom_history_keeps_original_signature_and_validation(matches: bool) -> None:
    class CustomHistory(History):
        async def task_result(self, graph_id, node_id, *, principal):
            return {"custom": True}

    evaluation = object.__new__(Evaluations)
    evaluation._history = object.__new__(CustomHistory)
    digest = canonical_sha256({"custom": True}) if matches else "0" * 64
    if matches:
        assert await evaluation._capture_task_result("g", "n", principal=object(), expected_digest=digest) == {"custom": True}
    else:
        with pytest.raises(AIError) as error:
            await evaluation._capture_task_result("g", "n", principal=object(), expected_digest=digest)
        assert error.value.code is ErrorCode.INPUT_CAPTURE_UNAVAILABLE
