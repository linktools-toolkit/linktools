#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution views keep large content out of metadata and authorization reads."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import AuthorizationAction, ExecutionLineageKind, Principal, ResourceKind, ResourceRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ExecutionManager, TaskExecutionService, project_execution_view
from linktools.ai.persistence import Persistence, PersistenceDomain
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence.commands import StoredExecutionManagementAdapter
from linktools.ai.storage import StoredPayload

from .test_execution_result_read_io import _PRINCIPAL, _seed, _service
from .test_incremental_projection_boundaries import _storage
from .test_runtime_persistence_evidence import _execution

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_metadata_reads_do_not_fetch_large_execution_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="metadata", tenant_id="tenant")
    try:
        parent = await _seed(state, output=StoredPayload.inline_json("large output " * 20000))
        child = replace(parent, execution_id="child", parent_execution_id="execution",
                        parent_invocation_id="invocation", lineage_kind=ExecutionLineageKind.SUBAGENT)
        await state.execution.executions.create_execution(child)
        read_contents = ContentArchive.read_contents

        async def binding_only(self, identities, *, codec, **kwargs):
            if codec != "execution-binding":
                raise AssertionError("Metadata read fetched execution payload content")
            return await read_contents(self, identities, codec=codec, **kwargs)

        monkeypatch.setattr(ContentArchive, "read_contents", binding_only)
        repository = state.execution.executions
        assert (await repository.get_header("execution", tenant_id="tenant")).owner_principal_id is None
        assert (await repository.get_header_fresh("execution", tenant_id="tenant")).owner_principal_id is None
        service = _service(state)
        assert await service.inspect("execution", principal=_PRINCIPAL) == project_execution_view(parent)
        assert await service.list_children("execution", principal=_PRINCIPAL) == (project_execution_view(child),)
        assert (await repository.get_view_fresh("execution", tenant_id="tenant"))[1] == project_execution_view(parent)
        with pytest.raises(AssertionError, match="Metadata read fetched execution payload content"):
            await repository.get_fresh("execution", tenant_id="tenant")
    finally:
        await state.close()


async def test_task_inspection_preserves_admitted_principal_authorization(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="metadata", tenant_id="tenant")
    try:
        record = await _seed(state)

        class OwnerPolicy:
            async def authorize(self, principal, action, resource):
                if resource.owner_principal_id != principal.principal_id:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        service = TaskExecutionService(state.execution, OwnerPolicy(), port=StoredExecutionManagementAdapter(
            state.execution, state.object_store(PersistenceDomain.EXECUTION),
        ))
        assert await service.inspect("execution", principal=_PRINCIPAL) == project_execution_view(record)
    finally:
        await state.close()


@pytest.mark.parametrize("change", ("owner", "missing"))
async def test_metadata_authorization_rechecks_changed_owner_or_disappearance(change: str) -> None:
    record = _execution()
    view = project_execution_view(record)
    original = ResourceRef(ResourceKind.EXECUTION, record.execution_id, "tenant", "principal")
    replacement = replace(original, owner_principal_id="other")

    class Port:
        value = (original, view)

        async def read_view(self, execution_id, *, tenant_id, observed=False):
            return self.value

    port = Port()

    class Policy:
        async def authorize(self, principal, action, resource):
            if resource.owner_principal_id == "other":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            port.value = (replacement, view) if change == "owner" else None

    with pytest.raises(AIError) as caught:
        await ExecutionManager(port, Policy()).load_view_authorized(
            "execution", Principal("principal", "tenant", "service"), AuthorizationAction.EXECUTION_READ,
        )
    assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED


async def test_task_wait_polls_metadata_without_loading_active_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.core import TenantAuthorizationPolicy

    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="metadata", tenant_id="tenant")
    try:
        await state.execution.executions.create_execution(_execution())

        async def forbid_full_record(*args, **kwargs):
            raise AssertionError("Active wait poll hydrated execution content")

        monkeypatch.setattr(state.execution.executions, "get_fresh", forbid_full_record)
        service = TaskExecutionService(state.execution, TenantAuthorizationPolicy(),
            port=StoredExecutionManagementAdapter(
                state.execution, state.object_store(PersistenceDomain.EXECUTION),
            ))
        with pytest.raises(AIError) as caught:
            await service.wait("execution", principal=_PRINCIPAL, timeout_seconds=0.01)
        assert caught.value.code is ErrorCode.EXECUTION_WAIT_TIMEOUT
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("local_sqlite", "sqlite"))
async def test_cold_children_snapshot_uses_summary_without_content_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="cold-metadata", tenant_id="tenant")
    parent = _execution()
    parent = replace(parent, binding=replace(parent.binding,
        output_schema={"type": "string", "description": "large binding " * 20000}))
    child = replace(parent, execution_id="child", parent_execution_id="execution",
                    parent_invocation_id="invocation", lineage_kind=ExecutionLineageKind.SUBAGENT)
    try:
        await state.execution.executions.create_execution(parent)
        await state.execution.executions.create_execution(child)
    finally:
        await state.close()
    reader = Persistence(_storage(backend, tmp_path))
    await reader.initialize(namespace="cold-metadata", tenant_id="tenant", read_only=True)
    try:
        async def reject_content(*args, **kwargs):
            raise AssertionError("metadata must not hydrate binding or payload bodies")

        monkeypatch.setattr(ContentArchive, "read_contents", reject_content)
        assert await _service(reader).list_children("execution", principal=_PRINCIPAL) == (
            project_execution_view(child),
        )
    finally:
        await reader.close()


@pytest.mark.parametrize("kind,changes", (
    ("agent", {"task_attempt": 1}),
    ("agent", {"task_deadline_at": "timestamp"}),
    ("agent", {"task_next_attempt_at": "timestamp"}),
    ("task", {"task_attempt": -1}),
    ("task", {"agent_run_seq": 1}),
    ("task", {"task_attempt": 2}),
    ("task", {"session_id": "session"}),
    ("task", {"lineage_kind": ExecutionLineageKind.SUBAGENT,
              "parent_execution_id": "parent", "parent_invocation_id": "call"}),
    ("agent", {"lineage_kind": ExecutionLineageKind.SUBAGENT,
               "parent_execution_id": None, "parent_invocation_id": "call"}),
    ("agent", {"lineage_kind": ExecutionLineageKind.SUBAGENT,
               "parent_execution_id": "parent", "parent_invocation_id": ""}),
    ("agent", {"parent_execution_id": "parent"}),
    ("agent", {"parent_invocation_id": "call"}),
))
async def test_metadata_view_rejects_invalid_consumed_execution_state(kind: str, changes: dict) -> None:
    from linktools.ai.execution import ExecutionRecord
    from linktools.ai.persistence._repository_common import project_record
    from linktools.ai.persistence.codec import (
        _decode_enveloped_domain, decode_execution_view, encode_domain, encode_envelope,
    )
    from linktools.ai.task import TaskBindingContract

    record = _execution()
    if kind == "task":
        record = replace(record, binding=TaskBindingContract("callable", 1, "none", {"kind": "json"}, None, 1, 0),
                         mode=None, planning=None, thinking=None, agent_run_seq=0, conversation=False)
    stored = project_record(namespace="metadata", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                            kind="execution", identity=(record.execution_id,), value=record)
    value = stored.data["value"]
    fields = dict(value["payload"]["fields"])
    fields.update({key: encode_domain(record.created_at if item == "timestamp" else item)
                   for key, item in changes.items()})
    wire = encode_envelope({**value, "payload": {**value["payload"], "fields": fields}})
    with pytest.raises(AIError) as full_error:
        _decode_enveloped_domain(wire, ExecutionRecord)
    with pytest.raises(AIError) as view_error:
        decode_execution_view(wire, tenant_id="tenant")
    assert full_error.value.code is view_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


async def test_metadata_identity_type_errors_use_storage_integrity_error() -> None:
    from linktools.ai.persistence._repository_common import project_record
    from linktools.ai.persistence.codec import decode_execution_header, decode_execution_view, encode_envelope

    record = _execution()
    stored = project_record(namespace="metadata", tenant_id="tenant", domain=PersistenceDomain.EXECUTION,
                            kind="execution", identity=(record.execution_id,), value=record)
    value = stored.data["value"]
    payload = value["payload"]
    wire = encode_envelope({**value, "payload": {**payload,
        "fields": {**payload["fields"], "execution_id": 1}}})
    for decode in (decode_execution_header, decode_execution_view):
        with pytest.raises(AIError) as error:
            decode(wire, tenant_id="tenant")
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
