#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime cancel / resume principal enforcement (§7.3) + cancel-audit (§8.2).

Phase 2 of the production-hardening plan: sensitive operations no longer act
on a bare ``run_id``. Default (production-safe) Runtime rejects cancel / resume
without a Principal; ``local_trusted_mode`` is the explicit single-tenant
escape (with a DeprecationWarning). When a Principal is presented and the run
has a tenant, ownership is enforced; the cancel-request audit fields are
populated from the trusted Principal.
"""

import asyncio
import warnings
from datetime import datetime, timezone

import pytest

from linktools.ai.errors import PrincipalAccessDeniedError
from linktools.ai.run.models import (
    RunInput,
    RunRecord,
    RunnableType,
    RunStatus,
)
from linktools.ai.runtime import Runtime
from linktools.ai.identity.principal import PrincipalContext
from linktools.ai.governance.security.authorization import ScopeAuthorization
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.jobs.models import ActorRef, ScopeSet

_NOW = datetime(2026, 7, 6, tzinfo=timezone.utc)


def _seed_run(store, run_id: str, status: RunStatus = RunStatus.RUNNING,
              task_attempt_id: "str | None" = None) -> None:
    async def _seed():
        await store.runs.create(
            RunRecord(
                id=run_id,
                root_run_id=run_id,
                parent_run_id=None,
                session_id="session-x",
                runnable_id="agent-x",
                runnable_type=RunnableType.AGENT,
                status=RunStatus.PENDING,
                input=RunInput(prompt="seed"),
                result=None,
                error=None,
                version=1,
                created_at=_NOW,
                started_at=None,
                finished_at=None,
                metadata={"tenant_id": "t1", **({"task_attempt_id": task_attempt_id}
                         if task_attempt_id is not None else {})},
            )
        )
        if status is RunStatus.PENDING:
            return
        await store.runs.transition(run_id, RunStatus.RUNNING, expected_version=1)
        if status is RunStatus.RUNNING:
            return
        await store.runs.transition(run_id, status, expected_version=2)

    asyncio.run(_seed())


def _principal(tenant_id: str = "t1", actor_id: str = "alice") -> PrincipalContext:
    return PrincipalContext(
        tenant_id=tenant_id,
        user_id=actor_id,
        actor=ActorRef(kind="user", id=actor_id),
        scopes=ScopeSet.allow_all(),
    )


# --- §7.3 default-strict gate ------------------------------------------------


def test_cancel_without_principal_denied_by_default(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)  # default strict
    _seed_run(storage, "run-1", RunStatus.RUNNING)

    with pytest.raises(PrincipalAccessDeniedError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.run(runtime.cancel("run-1"))


def test_cancel_without_principal_emits_deprecation(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)
    _seed_run(storage, "run-2", RunStatus.RUNNING)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(PrincipalAccessDeniedError):
            asyncio.run(runtime.cancel("run-2"))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_cancel_without_principal_allowed_in_local_trusted_mode(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, local_trusted_mode=True)
    _seed_run(storage, "run-3", RunStatus.RUNNING)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.run(runtime.cancel("run-3"))

    record = asyncio.run(storage.runs.get("run-3"))
    assert record is not None
    assert record.status is RunStatus.CANCELLED

# --- principal presented ----------------------------------------------------


def test_cancel_with_principal_proceeds_and_audits(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, authorization=ScopeAuthorization())
    _seed_run(storage, "run-4", RunStatus.RUNNING)

    asyncio.run(
        runtime.cancel("run-4", principal=_principal(), reason="user-requested")
    )
    record = asyncio.run(storage.runs.get("run-4"))
    assert record is not None
    assert record.status is RunStatus.CANCELLED
    # §8.2 audit: identity derived from the trusted Principal, not caller-supplied.
    assert record.cancel_requested_by == "user:alice"
    assert record.cancel_reason == "user-requested"
    assert record.cancel_requested_at is not None


def test_cancel_with_principal_emits_no_deprecation(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, authorization=ScopeAuthorization())
    _seed_run(storage, "run-5", RunStatus.RUNNING)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(runtime.cancel("run-5", principal=_principal()))
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_task_attempt_can_cancel_only_its_bound_run(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, authorization=ScopeAuthorization())
    _seed_run(storage, "run-self", task_attempt_id="attempt-1")
    principal = PrincipalContext(tenant_id="t1", user_id="alice",
        actor=ActorRef(kind="task-attempt", id="attempt-1"),
        scopes=ScopeSet.of("run.cancel:self"))
    asyncio.run(runtime.cancel("run-self", principal=principal))
    assert asyncio.run(storage.runs.get("run-self")).status is RunStatus.CANCELLED

    _seed_run(storage, "run-other", task_attempt_id="attempt-2")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(runtime.cancel("run-other", principal=principal))


def test_resume_without_principal_denied_by_default(tmp_path):
    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage)

    async def _drive():
        async for _event in runtime.resume("missing"):
            pass

    with pytest.raises(PrincipalAccessDeniedError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.run(_drive())


# --- §5.5 cross-tenant denial ----------------------------------------------


def test_cancel_cross_tenant_denied(tmp_path):
    from linktools.ai.run.definition import RunDefinitionSnapshot

    storage = FilesystemStorage(root=tmp_path)
    runtime = Runtime.build(storage=storage, local_trusted_mode=True)
    _seed_run(storage, "run-6", RunStatus.RUNNING)
    # Seed the run's definition owning it for "owner-tenant" so the principal's
    # tenant can be compared against a real resource tenant.
    asyncio.run(
        storage.run_definitions.create(
            RunDefinitionSnapshot(
                run_id="run-6",
                runnable_type="agent",
                runnable_id="agent-x",
                serialized_spec={},
                spec_fingerprint="fp",
                user_id=None,
                tenant_id="owner-tenant",
                workspace=None,
                provider_revision=None,
                created_at=_NOW,
            )
        )
    )

    # Principal from a different tenant must be rejected.
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(
            runtime.cancel("run-6", principal=_principal(tenant_id="other-tenant"))
        )
    # Run untouched.
    record = asyncio.run(storage.runs.get("run-6"))
    assert record is not None
    assert record.status is RunStatus.RUNNING

    # Same-tenant principal succeeds.
    asyncio.run(
        runtime.cancel("run-6", principal=_principal(tenant_id="owner-tenant"))
    )
    record = asyncio.run(storage.runs.get("run-6"))
    assert record is not None
    assert record.status is RunStatus.CANCELLED
