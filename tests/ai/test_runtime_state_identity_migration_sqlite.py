#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite regression for automatic pre-composition Runtime V1 normalization."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.agent import AgentCatalog, AgentCompiler
from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, SessionStatus
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import SessionRecord
from linktools.ai.runtime.state._codec import encode_domain, encode_envelope
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._repositories import _domain_data
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import Workspace, open_workspace_runtime


def _legacy_binding() -> tuple[object, object]:
    compiler = AgentCompiler(
        model_resolver=ModelRegistry.openai(model="gpt-test").snapshot(),
        runtime_fingerprint="a" * 64,
    )
    definition = compiler.compile(AgentSpec("default"))
    catalog = AgentCatalog({"default": definition})
    return definition, compiler.bind(catalog.root_definition("default"))


def _legacy_snapshot(binding: object, legacy_digest: str) -> dict[str, object]:
    payload = dict(binding.snapshot.to_payload())
    payload.pop("agent_digest")
    spec = dict(payload["agent_spec"])
    spec["metadata"] = {}
    payload["agent_spec"] = spec
    payload["binding_digest"] = legacy_digest
    return payload


def _legacy_session_data(
    record: SessionRecord,
    legacy_digest: str,
) -> dict[str, object]:
    payload = encode_domain(record)
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields.pop("agent_digest")
    return encode_envelope(
        {
            "type": "session_record",
            "payload": {"$dataclass": "session_record", "fields": fields},
        }
    )


def _legacy_execution_data(
    record: ExecutionRecord,
    binding: object,
    legacy_digest: str,
) -> dict[str, object]:
    data = _domain_data(record)
    value = dict(data["value"])
    payload = dict(value["payload"])
    fields = dict(payload["fields"])
    fields["binding_digest"] = encode_domain(legacy_digest)
    fields["binding"] = _legacy_snapshot(binding, legacy_digest)
    payload["fields"] = fields
    value["payload"] = payload
    data["value"] = value
    return data


@pytest.mark.asyncio
async def test_open_workspace_runtime_normalizes_legacy_sqlite_session(
    tmp_path,
) -> None:
    workspace = Workspace.load(tmp_path)
    path = workspace.storage_root / "runtime.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_runtime_database(engine)
    await engine.dispose()

    definition, binding = _legacy_binding()
    legacy_digest = "f" * 64
    session_id = workspace.workspace_id
    now = datetime.now(timezone.utc)
    session = SessionRecord(
        session_id=session_id,
        tenant_id="default",
        owner_principal_id="runtime",
        agent_digest=definition.digest,
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={"linktools.ai.agent_id": "default"},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        continuation=None,
        history_quality="complete",
        history_id="history",
    )
    execution = ExecutionRecord(
        execution_id="execution",
        tenant_id="default",
        session_id=session_id,
        binding_digest=binding.digest,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.SUCCEEDED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        planning=False,
        thinking=False,
        binding=binding.snapshot,
    )

    legacy = RuntimeState.sqlite(path)
    await legacy.initialize(namespace=workspace.workspace_id, tenant_id="default")
    sessions = legacy.conversation.sessions
    executions = legacy.execution.executions
    session_record = sessions._stored(
        "session", session_id, session, state=session.status.value
    )
    session_record = replace(
        session_record,
        data=_legacy_session_data(session, legacy_digest),
    )
    execution_record = executions._stored(
        "execution", "execution", execution, state=execution.status.value
    )
    execution_record = replace(
        execution_record,
        data=_legacy_execution_data(execution, binding, legacy_digest),
    )
    await sessions.state_store.mutate(lambda tx: tx.insert_record(session_record))
    await executions.state_store.mutate(lambda tx: tx.insert_record(execution_record))
    await legacy.close()

    reopened = RuntimeState.sqlite(path)
    models = ModelRegistry.openai(model="gpt-test")
    async with open_workspace_runtime(
        workspace,
        state=reopened,
        models=models,
    ) as runtime:
        loaded = await runtime.session.get(
            session_id,
            principal=runtime.default_principal,
        )
        assert loaded is not None
        assert loaded.agent_digest == runtime.agent()._agent_digest
