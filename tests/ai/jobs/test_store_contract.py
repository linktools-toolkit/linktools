#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JobStore Protocol contract (plan section 12).

The domain-semantic surface every JobStore backend (file, sqlalchemy) must
implement. The full backend contract test -- exercising claim / commit /
recover against real storage -- is parametrized over backends once they land
(plan 30.1); this phase fixes the Protocol shape and proves a conforming fake
is accepted while a stripped object is rejected.
"""

from linktools.ai.jobs import ClaimedTask, TaskClaim, JobStore

import dataclasses

import pytest


class _CompleteFakeStore:
    """Implements every JobStore method -- satisfies the Protocol."""

    async def create_job(self, job, root_task):  # pragma: no cover - shape only
        ...

    async def get_job(self, job_id): ...
    async def get_task(self, task_id): ...
    async def list_tasks(self, job_id, *, status=None): ...
    async def claim(self, *, worker_id, now, lease_seconds, handlers=None): ...
    async def renew_lease(self, **kwargs): ...
    async def bind_run(self, **kwargs): ...
    async def bind_runnable(self, **kwargs): ...
    async def commit_success(self, claim, outcome): ...
    async def commit_failure(self, claim, outcome): ...
    async def request_cancel(self, job_id, *, reason=None): ...
    async def submit_signal(self, signal): ...
    async def recover_expired(self, *, now, limit=100): ...
    async def reconcile_due(self, *, now, limit=100): ...
    async def list_orphan_run_ids(self, *, limit=500): ...
    async def list_attempts(self, task_id): ...
    async def list_transitions(self, job_id): ...


class _StrippedStore:
    async def get_task(self, task_id): ...


def test_complete_store_satisfies_protocol() -> None:
    assert isinstance(_CompleteFakeStore(), JobStore)


def test_stripped_store_does_not_satisfy_protocol() -> None:
    assert not isinstance(_StrippedStore(), JobStore)


def test_claim_types_are_frozen_value_objects() -> None:
    claim = TaskClaim(task_id="t", attempt_id="a", worker_id="w", fencing_token=3)
    # A worker must carry the full claim after claiming; it is immutable.
    assert dataclasses.is_dataclass(claim)
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.fencing_token = 99  # type: ignore[misc]
    # ClaimedTask bundles the claim with the job/task/attempt snapshot.
    assert "claim" in {f.name for f in dataclasses.fields(ClaimedTask)}
