#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution metadata queries must not re-read decoded candidates."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
)
from linktools.ai.runtime import ListExecutionRequest
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime.state._contracts import (
    ExecutionCandidate,
    ExecutionCandidatePage,
)


class _Reader:
    pass


class _Allow:
    async def authorize(self, _principal, _action, _resource) -> None:
        return None


class _Candidates:
    def __init__(self) -> None:
        self.list_calls = 0

    async def list_candidates(self, **_kwargs) -> ExecutionCandidatePage:
        self.list_calls += 1
        record = SimpleNamespace(
            execution_id="exec-1",
            agent_id="agent",
            status=ExecutionStatus.SUCCEEDED,
            lineage_kind=ExecutionLineageKind.RUN,
            parent_execution_id=None,
            root_execution_id="exec-1",
            parent_invocation_id=None,
            session_id=None,
        )
        return ExecutionCandidatePage(
            (ExecutionCandidate(record, "cursor-1"),),  # type: ignore[arg-type]
            False,
        )

    async def get_header(self, *_args, **_kwargs):
        raise AssertionError("list() must not re-read candidate headers")


@pytest.mark.asyncio
async def test_execution_list_authorizes_decoded_candidate_without_header_reread() -> None:
    repository = _Candidates()
    service = DefaultExecutionHistoryService(
        repository,  # type: ignore[arg-type]
        _Allow(),  # type: ignore[arg-type]
        _Reader(),  # type: ignore[arg-type]
        HmacCursorSigner("execution", b"query-key"),
    )

    page = await service.list(
        ListExecutionRequest(Principal("caller", "tenant", "service"))
    )

    assert repository.list_calls == 1
    assert [item.execution_id for item in page.items] == ["exec-1"]
