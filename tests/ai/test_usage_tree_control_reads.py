#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usage traversal preserves per-node authorization without loading payloads."""

import pytest

from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl

from .test_runtime_execution_query import _record
from .test_runtime_usage_summary import _GraphUsageHistory, _UsageService


@pytest.mark.asyncio
@pytest.mark.parametrize("graph", (False, True))
@pytest.mark.parametrize("outcome", ("success", "denied", "missing"))
async def test_usage_tree_reads_only_membership_after_initial_authorization(
    monkeypatch: pytest.MonkeyPatch, graph: bool, outcome: str,
) -> None:
    async with Persistence.open("execution-query", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        repository = state.execution.executions
        for record in (_record("root", agent_id="agent"),
                       _record("child", agent_id="agent", parent_execution_id="root")):
            await repository.create(record)
        service = _UsageService()
        history_type = _GraphUsageHistory if graph else History
        history = history_type(service, tenant_id="tenant", executions=repository,
                               authorization=TenantAuthorizationPolicy("tenant"))
        original_get = repository.get
        original_header = repository.get_header
        original_usage = service.usage
        full_reads = []

        async def read_full(execution_id, *, tenant_id):
            full_reads.append(execution_id)
            assert not graph and full_reads == ["root"]
            return await original_get(execution_id, tenant_id=tenant_id)

        async def read_header(execution_id, *, tenant_id):
            if outcome == "missing" and execution_id == "child":
                return None
            return await original_header(execution_id, tenant_id=tenant_id)

        async def usage(execution_id, **kwargs):
            if outcome == "denied" and execution_id == "child":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            return await original_usage(execution_id, **kwargs)

        async def reject_full_children(*args, **kwargs):
            raise AssertionError("usage traversal loaded child execution bodies")

        monkeypatch.setattr(repository, "get", read_full)
        monkeypatch.setattr(repository, "get_header", read_header)
        monkeypatch.setattr(service, "usage", usage)
        monkeypatch.setattr(ExecutionRepositoryImpl, "_decode_many", reject_full_children)
        principal = Principal("caller", "tenant", "service")
        pending = history.graph_usage("graph", principal=principal) if graph else history.usage(
            "root", principal=principal, include_children=True)
        if outcome == "success":
            result = await pending
            assert result.logical_requests == 2 and service.calls == ["root", "child"]
        else:
            with pytest.raises(AIError) as raised:
                await pending
            assert raised.value.code is (ErrorCode.AUTHORIZATION_DENIED if outcome == "denied"
                                         else ErrorCode.STORAGE_INTEGRITY_ERROR)
            assert service.calls == ["root"]
        assert full_reads == ([] if graph else ["root"])
