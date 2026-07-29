#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime.resume/.cancel/.decide_approval must authorize against the run's
recorded (tenant_id, user_id) -- a caller who merely knows a run_id must not
be able to act on another tenant's run. Runs are put into CLAIMED/PAUSED
state directly via the store (bypassing the model/tool/governance chain)
since only the ownership check on resume/cancel/decide_approval is under
test here."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256

import pytest

from linktools.ai.agent.codec import AgentSpecCodec
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import PrincipalAccessDeniedError
from linktools.ai.execution.commands import ClaimExecution, PauseExecution, StartExecution
from linktools.ai.execution.domain import ApprovalDecision, RunApproval, RunDefinition, RunKind, RunStatus, RunUsage, RunnableType
from linktools.ai.execution.snapshots import AgentSnapshotData
from linktools.ai.json import canonical_json_bytes
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime
from linktools.ai.governance.identity import trusted_local_principal
from tests.ai.fakes.model import make_router


def _spec() -> AgentSpec:
    return AgentSpec("agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer"))


async def _start_claimed_run(store, *, run_id: str, tenant_id: str):
    await store.create_session(session_id="s", user_id=None, tenant_id=tenant_id)
    codec = AgentSpecCodec()
    value = codec.encode(_spec())
    definition = RunDefinition(
        "agent",
        RunnableType.AGENT,
        "agent-spec.v1",
        value,
        sha256(canonical_json_bytes(value)).hexdigest(),
    )
    await store.start_run(StartExecution(run_id, "s", RunKind.USER_TURN, definition, "hi"))
    return await store.claim_run(ClaimExecution(run_id, "worker", datetime.now(timezone.utc), timedelta(minutes=5)))


async def _pause_run(store, claimed, *, approval_id: str):
    now = datetime.now(timezone.utc)
    snapshot = AgentSnapshotData((), None, RunUsage(), 0)
    approval = RunApproval(approval_id, "tc-1", "some-tool", "binding")
    return await store.pause_run(PauseExecution(claimed.id, claimed.lease.owner, claimed.lease.fence, snapshot, approval))


@pytest.mark.asyncio
async def test_cancel_rejects_cross_tenant_access(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    await _start_claimed_run(storage.execution, run_id="r1", tenant_id="t1")

    with pytest.raises(PrincipalAccessDeniedError):
        await runtime.cancel("r1", principal=trusted_local_principal(tenant_id="t2"))

    await runtime.cancel("r1", principal=trusted_local_principal(tenant_id="t1"))
    record = await storage.execution.get_run("r1")
    assert record.status is RunStatus.CANCELLING
    await runtime.aclose()


@pytest.mark.asyncio
async def test_resume_rejects_cross_tenant_access(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    claimed = await _start_claimed_run(storage.execution, run_id="r1", tenant_id="t1")
    await _pause_run(storage.execution, claimed, approval_id="a1")
    await runtime.decide_approval("r1", approval_id="a1", decision=ApprovalDecision.ALLOW, principal=trusted_local_principal(tenant_id="t1"))

    with pytest.raises(PrincipalAccessDeniedError):
        await runtime.resume("r1", principal=trusted_local_principal(tenant_id="t2"))

    assert (await runtime.resume("r1", principal=trusted_local_principal(tenant_id="t1"))) is not None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_decide_approval_rejects_cross_tenant_access(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    claimed = await _start_claimed_run(storage.execution, run_id="r1", tenant_id="t1")
    await _pause_run(storage.execution, claimed, approval_id="a1")

    with pytest.raises(PrincipalAccessDeniedError):
        await runtime.decide_approval("r1", approval_id="a1", decision=ApprovalDecision.DENY, principal=trusted_local_principal(tenant_id="t2"))
    await runtime.aclose()


@pytest.mark.asyncio
async def test_decide_approval_deny_is_terminal(tmp_path):
    storage = LocalDirectoryStorage(tmp_path)
    runtime = build_runtime(storage=storage, model_resolver=make_router())
    claimed = await _start_claimed_run(storage.execution, run_id="r1", tenant_id="t1")
    await _pause_run(storage.execution, claimed, approval_id="a1")

    record = await runtime.decide_approval("r1", approval_id="a1", decision=ApprovalDecision.DENY, principal=trusted_local_principal(tenant_id="t1"))
    assert record.status is RunStatus.CANCELLED

    with pytest.raises(PrincipalAccessDeniedError):
        await runtime.resume("r1", principal=trusted_local_principal(tenant_id="wrong-tenant"))
    await runtime.aclose()
