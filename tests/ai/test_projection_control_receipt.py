#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guard-only receipts never stand in for a changed execution value."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime._local import LocalExecutionBackend

from .test_history_projection_conformance import _record
from .test_checkpoint_metadata_snapshot_io import _trace_archive
from .test_incremental_projection_boundaries import _storage
from linktools.ai.persistence._step_contracts import AgentRunRecord


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("valid", "other-store", "retired", "stale", "wrong-key", "changed-value", "old-revision", "claim", "raw-claim", "raw-generation", "raw-identity", "raw-status"))
async def test_projection_receipt_updates_only_its_exact_live_owner(case: str) -> None:
    async with Persistence.open("receipt", tenant_id="tenant", storage=PersistenceStorage.in_memory()) as state:
        repository = state.execution.executions
        initial = replace(_record(ExecutionStatus.STARTED, 0), producer_generation=1, producer_claim_id="claim")
        await repository.create_execution(initial)
        observed = await repository.get_observed(initial.execution_id, tenant_id="tenant")
        before = await repository.state_store.read_record(repository.execution_record_key(initial.execution_id))
        after = replace(before, storage_revision=before.storage_revision + 1)
        owner = SimpleNamespace(observed=observed)
        backend = LocalExecutionBackend.__new__(LocalExecutionBackend)
        backend._execution = SimpleNamespace(executions=repository)
        backend._turn_owners = {initial.execution_id: owner}
        store = repository.state_store
        claim = "claim"
        if case == "other-store":
            store = object()
        elif case == "retired":
            backend._turn_owners.clear()
        elif case == "stale":
            owner.observed = replace(observed, storage_revision=observed.storage_revision + 5)
        elif case == "wrong-key":
            before = replace(before, key=repository.execution_record_key("other"))
            after = replace(after, key=before.key)
        elif case == "changed-value":
            after = replace(after, data={**after.data, "receipt_test": True})
        elif case == "old-revision":
            after = before
        elif case == "claim":
            claim = "another"
        if case.startswith("raw-"):
            updates = {"raw-claim": {"producer_claim_id": "another"},
                       "raw-generation": {"producer_generation": 2},
                       "raw-identity": {"execution_id": "another"},
                       "raw-status": {"status": ExecutionStatus.CANCELLING}}[case]
            rewritten = await repository._stored_content(
                "execution", (initial.execution_id,), replace(initial, **updates), state=initial.status.value,
            )
            before = replace(before, data=rewritten.data)
            after = replace(after, data=before.data)
        original = owner.observed
        backend._confirm_execution_projection(initial.execution_id, owner, 1, claim, store, before, after)
        if case == "valid":
            assert owner.observed.value is observed.value
            assert owner.observed.storage_revision == after.storage_revision
        else:
            assert owner.observed is original


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("change", (False, True))
async def test_direct_archive_projection_keeps_none_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, change: bool,
) -> None:
    async with Persistence.open("projection-guard", tenant_id="tenant", storage=_storage(backend, tmp_path)) as state:
        archive = await _trace_archive(state)
        repository = state.execution.executions
        async def share(*args, **kwargs):
            if not change:
                return
            async def update(tx):
                key = repository.execution_record_key("execution")
                before = await tx.get_record(key)
                assert await tx.replace_record(replace(before, storage_revision=before.storage_revision + 1,
                    data={**before.data, "receipt_test": True}), expected_storage_revision=before.storage_revision)
            await repository.state_store.mutate(update)
        monkeypatch.setattr(archive, "share_published_content", share)
        receipt = await archive.sync_prepared_projection(
            AgentRunRecord("run"), events=(), checkpoints=(), producer_generation=1, execution_id="execution",
        )
        assert receipt is None
        stored = await repository.state_store.read_record(repository.execution_record_key("execution"))
        assert stored.data.get("receipt_test", False) is change


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", (False, True))
async def test_projection_notification_requires_direct_commit_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uncertain: bool,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.session import CreateSessionRequest
    from linktools.ai.persistence import runs
    from linktools.ai.workspace import Workspace
    from linktools.commands.ai._common import _open_local_runtime
    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    original_commit = runs.run_durable_commit
    async def commit(*args, **kwargs):
        result = await original_commit(*args, **kwargs)
        if uncertain and result.committed:
            return replace(result, error=AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN))
        return result
    monkeypatch.setattr(runs, "run_durable_commit", commit)
    original_bind = runs.RuntimeAgentRunStore.bind_step_boundary
    notifications = []
    def bind(self, run_id, writer, require_active, *, confirm_projection=None):
        def confirm(*receipt):
            notifications.append(receipt)
            confirm_projection(*receipt)
        return original_bind(self, run_id, writer, require_active,
                             confirm_projection=None if confirm_projection is None else confirm)
    monkeypatch.setattr(runs.RuntimeAgentRunStore, "bind_step_boundary", bind)
    def respond(_messages, _info):
        return ModelResponse(parts=[TextPart("complete")])
    async with _open_local_runtime(Workspace.initialize(tmp_path), model=_Models(_UsageFunctionModel(respond))) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        run = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="turn")
        assert (await run.wait(timeout_seconds=20)).result.output == {"text": "complete"}
    assert bool(notifications) is not uncertain
