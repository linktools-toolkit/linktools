#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure Run transition and lease validation functions."""


from ..storage.coordination.lease import assert_active, claim, renew
from ..errors import InvalidRunTransitionError, RunConflictError, RunNotResumableError
from .domain import ALLOWED_RUN_TRANSITIONS, ApprovalDecision, RunStatus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime, timedelta
    from ..storage.coordination.lease import Lease
    from .domain import RunRecord

def assert_transition(current: RunStatus, target: RunStatus) -> None:
    if target not in ALLOWED_RUN_TRANSITIONS[current]:
        raise InvalidRunTransitionError(f"cannot transition {current.value} to {target.value}")


def assert_claimable(run: "RunRecord", now: "datetime") -> None:
    if run.status is RunStatus.PENDING:
        return
    if run.status is RunStatus.RUNNING and (run.lease.expires_at is None or run.lease.expires_at <= now):
        return
    raise RunConflictError(f"run {run.id} is not claimable")


def claim_lease(run: "RunRecord", owner: str, now: "datetime", duration: "timedelta") -> "Lease":
    assert_claimable(run, now)
    return claim(run.lease, owner=owner, now=now, duration=duration)


def renew_lease(run: "RunRecord", owner: str, fence: int, now: "datetime", duration: "timedelta") -> "Lease":
    if run.status is not RunStatus.RUNNING:
        raise RunConflictError("only running runs can be renewed")
    return renew(run.lease, owner=owner, fence=fence, now=now, duration=duration)


def assert_owner(run: "RunRecord", owner: str, fence: int, now: "datetime") -> None:
    assert_active(run.lease, owner=owner, fence=fence, now=now)


def assert_resumable(run: "RunRecord") -> None:
    if run.status is not RunStatus.PAUSED:
        raise RunNotResumableError(f"run {run.id} is not paused")


def assert_approval_decided(run: "RunRecord") -> None:
    if run.approval is not None and run.approval.decision is not ApprovalDecision.ALLOW:
        raise RunConflictError("paused execution is not approved")


__all__ = [
    "assert_active",
    "assert_approval_decided",
    "assert_claimable",
    "assert_owner",
    "assert_resumable",
    "assert_transition",
    "claim_lease",
    "renew_lease",
]
