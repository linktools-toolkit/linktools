#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local AI composition and history contracts independent of Web extras."""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeHistory, UsageSummary
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskBindingContract
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import (
    _load_workspace,
    _local_metrics,
    _local_runtime_storage,
)
from linktools.commands.ai.run import Command as RunCommand


def _record(
    execution_id: str,
    created_at: datetime,
    *,
    error_code: str | None = None,
) -> SimpleNamespace:
    binding = TaskBindingContract("local-history", 1, "none", {}, None, 1, 0)
    return SimpleNamespace(
        execution_id=execution_id,
        binding_kind="task",
        agent_id=None,
        task_id="local-history",
        status=ExecutionStatus.FAILED if error_code else ExecutionStatus.SUCCEEDED,
        lineage_kind=ExecutionLineageKind.RUN,
        parent_execution_id=None,
        root_execution_id=execution_id,
        parent_invocation_id=None,
        session_id=None,
        created_at=created_at,
        updated_at=created_at,
        started_at=created_at,
        binding=binding,
        binding_digest=binding.binding_digest,
        stored_user_input=StoredUserInput(
            "task-input-v1", StoredPayload.inline_json({}),
            {"version": 1, "kind": "task"},
        ),
        result=None,
        error_code=error_code,
        safe_error_details={} if error_code is None else {"stage": "runtime"},
        error_diagnostics=None,
    )


class _RecentRepository:
    def __init__(self, records: tuple[SimpleNamespace, ...]) -> None:
        self.records = records

    async def list_candidates(
        self, *, cursor: str | None, **kwargs: object,
    ) -> SimpleNamespace:
        start = 0 if cursor is None else int(cursor) + 1
        end = min(len(self.records), start + 2)
        return SimpleNamespace(
            items=tuple(
                SimpleNamespace(record=self.records[index], cursor=str(index))
                for index in range(start, end)
            ),
            has_more=end < len(self.records),
        )

    async def get_header(
        self, execution_id: str, *, tenant_id: str,
    ) -> ResourceRef | None:
        if any(value.execution_id == execution_id for value in self.records):
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(
        self, execution_id: str, *, tenant_id: str,
    ) -> SimpleNamespace | None:
        return next(
            (value for value in self.records if value.execution_id == execution_id),
            None,
        )


class _RecentHistoryService:
    async def usage(
        self, execution_id: str, *, principal: Principal,
    ) -> UsageSummary:
        return UsageSummary()


def test_local_run_uses_default_storage_configuration() -> None:
    parser = RunCommand().create_parser()
    assert "storage" not in {action.dest for action in parser._actions}


def test_workspace_discovery_does_not_walk_up_without_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Workspace.initialize(tmp_path)
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)

    loaded = _load_workspace()

    assert loaded.root == nested
    assert not (nested / ".linktools").exists()


@pytest.mark.asyncio
async def test_local_storage_uses_separate_runtime_and_metrics_databases(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    storage = _local_runtime_storage(workspace)
    metrics = await _local_metrics(workspace)
    runtime_root = workspace.storage_root / "runtime"

    assert all(
        storage.plan.route(domain).path == (runtime_root / "runtime.db").resolve()
        for domain in storage.plan.durable_domains
    )
    assert metrics.namespace == "default"
    assert (runtime_root / "metrics.db").is_file()
    assert not (runtime_root / "runtime.db").exists()


@pytest.mark.asyncio
async def test_existing_invalid_metrics_database_fails_before_runtime_use(
    tmp_path: Path,
) -> None:
    workspace = Workspace.initialize(tmp_path)
    runtime_root = workspace.storage_root / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    database = runtime_root / "metrics.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE marker (id INTEGER PRIMARY KEY)")

    with pytest.raises(AIError) as error:
        await _local_metrics(workspace)

    assert error.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING


@pytest.mark.asyncio
async def test_runtime_history_returns_exact_recent_executions_and_errors() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    records = (
        _record("exec-middle", base + timedelta(minutes=2)),
        _record("exec-old", base),
        _record("exec-new", base + timedelta(minutes=3), error_code="FAILED_CODE"),
        _record("exec-less-old", base + timedelta(minutes=1)),
        _record("exec-newer", base + timedelta(minutes=4)),
    )
    history = RuntimeHistory(
        _RecentHistoryService(),  # type: ignore[arg-type]
        tenant_id="default",
        executions=_RecentRepository(records),  # type: ignore[arg-type]
        authorization=TenantAuthorizationPolicy("default"),
    )
    principal = Principal("cli", "default", "service")

    recent = await history.recent_executions(principal=principal, limit=3)
    inspected = await history.inspect_execution("exec-new", principal=principal)

    assert [value.execution_id for value in recent] == [
        "exec-newer", "exec-new", "exec-middle",
    ]
    assert inspected.error_code == "FAILED_CODE"
    assert inspected.safe_error_details == {"stage": "runtime"}
    assert inspected.error_diagnostics is None
