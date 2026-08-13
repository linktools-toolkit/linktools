#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evidence for the selective RuntimeStorage contract."""

import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from linktools.ai import RuntimeStorage, StorageDomain
from linktools.ai.adapter import SqlStepStore, build_filesystem_runtime
from linktools.ai.asset import SqlAssetBackend
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRecord, SessionRecord
from linktools.ai.workspace import open_workspace_runtime


def test_runtime_storage_normalizes_domains_without_hidden_defaults(tmp_path: Path) -> None:
    assert RuntimeStorage.filesystem(tmp_path).persist == frozenset({StorageDomain.CONVERSATION})
    assert RuntimeStorage.filesystem(tmp_path, persist=()).persist == frozenset()
    assert RuntimeStorage.sqlite(tmp_path / "runtime.db", persist=StorageDomain.ALL).persist == StorageDomain.durable()
    with pytest.raises(AIError) as error:
        RuntimeStorage.filesystem(tmp_path, persist={StorageDomain.ALL, StorageDomain.MEMORY})
    assert error.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_filesystem_domains_are_exact_across_cold_restart(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    runtime = build_filesystem_runtime(
        str(root),
        namespace="selective",
        persist=frozenset({StorageDomain.CONVERSATION}),
    )
    now = datetime.now(timezone.utc)
    await runtime.initialize()
    await runtime.persistence.conversation.sessions.create(
        SessionRecord(
            session_id="session",
            tenant_id="tenant",
            owner_principal_id="owner",
            binding_digest="binding",
            status=SessionStatus.OPEN,
            revision=0,
            resource_generation=0,
            cwd=None,
            metadata={},
            created_at=now,
            updated_at=now,
            closed_at=None,
        )
    )
    await runtime.persistence.execution.executions.create(
        ExecutionRecord(
            execution_id="execution",
            tenant_id="tenant",
            session_id="session",
            binding_digest="binding",
            parent_execution_id=None,
            root_execution_id="execution",
            source_execution_id=None,
            base_execution_id=None,
            lineage_kind=ExecutionLineageKind.RUN,
            status=ExecutionStatus.STARTED,
            revision=0,
            event_sequence=0,
            agent_run_sequence=0,
            result_ref=None,
            result_digest=None,
            error_code=None,
            safe_error_details={},
            created_at=now,
            updated_at=now,
        )
    )
    assert await runtime.persistence.execution.executions.get("execution", tenant_id="tenant") is not None
    await runtime.close()

    namespace_root = next(root.iterdir())
    manifest = json.loads((namespace_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "format": "linktools-ai-runtime",
        "generation": 1,
        "namespace": "selective",
    }
    assert (namespace_root / "conversation" / "records.json").is_file()
    assert not (namespace_root / "execution").exists()

    restarted = build_filesystem_runtime(
        str(root),
        namespace="selective",
        persist=frozenset({StorageDomain.CONVERSATION}),
    )
    await restarted.initialize()
    try:
        assert await restarted.persistence.conversation.sessions.get("session", tenant_id="tenant") is not None
        assert await restarted.persistence.execution.executions.get("execution", tenant_id="tenant") is None
    finally:
        await restarted.close()


def test_public_sql_and_workspace_constructors_use_engine_storage() -> None:
    assert "session_factory" not in inspect.signature(open_workspace_runtime).parameters
    assert "session_factory" not in inspect.signature(SqlStepStore).parameters
    assert "session_factory" not in inspect.signature(SqlAssetBackend).parameters
