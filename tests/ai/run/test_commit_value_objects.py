#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunCommitId + ExecutionFence + RunCommitPolicy value-object tests (P5 value-object invariants).

These are the foundational invariants the rest of the Run commit idempotency
+ strict-fencing work composes on:

- RunCommitId is bounded 1..200 chars; the empty string is forbidden (the
  old default that defeated idempotent replay).
- ExecutionFence is non-empty (the empty-string bypass is the bug strict
  fencing closes).
- RunCommitPolicy is a frozen per-topology declaration of whether fencing
  is required."""

from __future__ import annotations

import pytest

from linktools.ai.run.commit import ExecutionFence, RunCommitId, RunCommitPolicy


# --- RunCommitId ------------------------------------------------------------


def test_run_commit_id_accepts_typical_value():
    cid = RunCommitId("start:run-1")
    assert cid.value == "start:run-1"
    assert str(cid) == "start:run-1"


def test_run_commit_id_rejects_empty():
    with pytest.raises(ValueError, match="1..200"):
        RunCommitId("")


def test_run_commit_id_rejects_over_200_chars():
    with pytest.raises(ValueError, match="1..200"):
        RunCommitId("x" * 201)


def test_run_commit_id_accepts_exactly_200_chars():
    # Boundary: 200 is allowed; 201 is not.
    cid = RunCommitId("x" * 200)
    assert len(cid.value) == 200


def test_run_commit_id_is_frozen():
    cid = RunCommitId("start:run-1")
    with pytest.raises(Exception):
        cid.value = "other"  # type: ignore[misc]


def test_run_commit_id_equality():
    assert RunCommitId("a") == RunCommitId("a")
    assert RunCommitId("a") != RunCommitId("b")


# --- ExecutionFence ---------------------------------------------------------


def test_execution_fence_accepts_nonempty_token():
    fence = ExecutionFence("token-xyz")
    assert fence.token == "token-xyz"
    assert str(fence) == "token-xyz"


def test_execution_fence_rejects_empty_token():
    """The empty-string bypass is the bug strict fencing closes."""
    with pytest.raises(ValueError, match="cannot be empty"):
        ExecutionFence("")


def test_execution_fence_is_frozen():
    fence = ExecutionFence("t")
    with pytest.raises(Exception):
        fence.token = "other"  # type: ignore[misc]


# --- RunCommitPolicy --------------------------------------------------------


def test_run_commit_policy_is_frozen_dataclass():
    policy = RunCommitPolicy(fencing_required=True)
    assert policy.fencing_required is True
    with pytest.raises(Exception):
        policy.fencing_required = False  # type: ignore[misc]


def test_run_commit_policy_single_process_reference_topology():
    """The single-process reference deployment does not require fencing
    (there is no competing worker to lose a lease)."""
    policy = RunCommitPolicy(fencing_required=False)
    assert policy.fencing_required is False


def test_run_commit_policy_multi_worker_topology():
    """A multi-worker deployment DOES require fencing: a worker that lost its
    lease MUST NOT commit after the fact."""
    policy = RunCommitPolicy(fencing_required=True)
    assert policy.fencing_required is True
