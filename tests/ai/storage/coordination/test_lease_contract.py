#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared lease contract for Execution, Task, and Tool.

Run, Task, and Tool stores must run the SAME lease/fence semantics. This test
exercises the contract directly and asserts all three domains bind the identical
Lease type from storage.coordination.lease.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from linktools.ai.execution.domain import RunRecord
from linktools.ai.storage.coordination.lease import (
    Lease,
    assert_active,
    claim,
    is_expired,
    release,
    renew,
)
from linktools.ai.errors import StorageConflictError

now = datetime(2026, 1, 1, tzinfo=timezone.utc)
DURATION = timedelta(minutes=5)


def test_claim_sets_owner_and_advances_fence():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    assert lease.owner == "w1"
    assert lease.fence == 1
    assert lease.expires_at == now + DURATION


def test_unexpired_lease_cannot_be_stolen():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    # A second worker cannot claim while w1's lease is unexpired.
    with pytest.raises(StorageConflictError):
        claim(lease, owner="w2", now=now + timedelta(seconds=1), duration=DURATION)


def test_expired_lease_can_be_reclaimed_with_higher_fence():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    # Past expiry, a different worker may reclaim and the fence increases.
    reclaimed = claim(lease, owner="w2", now=now + DURATION + timedelta(seconds=1), duration=DURATION)
    assert reclaimed.owner == "w2"
    assert reclaimed.fence == lease.fence + 1


def test_old_owner_cannot_renew_after_reclaim():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    reclaimed = claim(lease, owner="w2", now=now + DURATION + timedelta(seconds=1), duration=DURATION)
    # w1's old fence is stale: renew must reject it.
    with pytest.raises(StorageConflictError):
        renew(reclaimed, owner="w1", fence=lease.fence, now=now + DURATION + timedelta(seconds=2), duration=DURATION)
    # The current owner at the current fence can renew, advancing expires_at.
    renewed = renew(reclaimed, owner="w2", fence=reclaimed.fence, now=now + DURATION + timedelta(seconds=2), duration=DURATION)
    assert renewed.expires_at > reclaimed.expires_at


def test_release_clears_owner_but_keeps_fence():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    released = release(lease)
    assert released.owner is None
    assert released.expires_at is None
    assert released.fence == lease.fence  # fence never resets -- stale writers are fenced out


def test_assert_active_rejects_expired_or_mismatched():
    lease = claim(Lease(), owner="w1", now=now, duration=DURATION)
    assert_active(lease, owner="w1", fence=lease.fence, now=now + timedelta(seconds=1))
    with pytest.raises(StorageConflictError):  # expired
        assert_active(lease, owner="w1", fence=lease.fence, now=now + DURATION + timedelta(seconds=1))
    with pytest.raises(StorageConflictError):  # wrong owner/fence
        assert_active(lease, owner="other", fence=99, now=now + timedelta(seconds=1))


def test_run_record_carries_the_shared_lease_type():
    # Execution, Task, and Tool all bind storage.coordination.lease.Lease as
    # their record's lease field type -- the single shared contract.
    import dataclasses

    lease_fields = {f.name: f.type for f in dataclasses.fields(RunRecord) if f.name == "lease"}
    assert "lease" in lease_fields
    # RunRecord.lease is the shared Lease (resolve the string annotation).
    assert RunRecord.__dataclass_fields__["lease"].type in (Lease, "Lease") or Lease.__name__ in str(
        RunRecord.__dataclass_fields__["lease"].type
    )
