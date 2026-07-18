#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reliability-architecture boundary freeze for linktools.ai.

Phase 0 of the production-hardening plan
(``.docs/linktools-ai-production-hardening-plan.md``) snapshots the CURRENT
reliability-relevant architecture. Every assertion describes the code as it is
on the branch base; when a later phase legitimately changes one of these
invariants, update the snapshot in the same change so the change is visible.

Already-in-place invariants (frozen so a regression is caught):

* ``RunStatus`` carries a ``CANCELLING`` intermediate state and the transition
  table routes in-flight runs through it (§8.1 -- already implemented);
* a cooperative ``CancellationToken`` propagates ``Runtime.cancel`` into the
  agent loop / tool executor (§8.5 -- already implemented).

Snapshot invariants (a later phase WILL change these -- update here then):

* ``IdempotencyStatus`` now spans {reserved, executed, completed, failed,
  unknown} -- §9.2 EXECUTED + UNKNOWN separate "Handler returned" from
  "result committed" so a commit failure is never resolved by re-running it;
* the atomic-write helper is the private ``_atomic_write`` shared across file
  stores -- §16.1 promotes it to a public ``atomic_write_bytes``;
* ``RunRecord`` carries cancel-request audit and worker fencing/manifest fields.

(The ``Runtime.cancel`` signature snapshot -- cross-process ownership fencing
lands in §8.3 / §8.7 -- lives in ``test_security_boundaries.py`` next to the
``resume`` signature under the §7.3 Principal change.)
"""

import dataclasses
import importlib.util


# --- Run status state machine -------------------------------------------------


def test_run_status_includes_cancelling_intermediate_state() -> None:
    # §8.1 requires a CANCELLING intermediate state between RUNNING and
    # CANCELLED so cancel is never falsely advertised as complete.
    from linktools.ai.run.models import RunStatus

    values = {s.value for s in RunStatus}
    assert {"running", "cancelling", "cancelled"} <= values, values


def test_cancelling_transitions_only_to_terminal() -> None:
    # §8.1: CANCELLING -> {CANCELLED, FAILED}; it must not return to RUNNING.
    from linktools.ai.run.models import (
        ALLOWED_RUN_TRANSITIONS,
        RunStatus,
    )

    assert ALLOWED_RUN_TRANSITIONS[RunStatus.CANCELLING] == frozenset(
        {RunStatus.CANCELLED, RunStatus.FAILED}
    )


def test_in_flight_states_can_reach_cancelling() -> None:
    # §8.1: RUNNING / WAITING_APPROVAL / PAUSED may transition to CANCELLING
    # rather than jumping straight to CANCELLED.
    from linktools.ai.run.models import (
        ALLOWED_RUN_TRANSITIONS,
        RunStatus,
    )

    for src in (RunStatus.RUNNING, RunStatus.WAITING_APPROVAL, RunStatus.PAUSED):
        assert RunStatus.CANCELLING in ALLOWED_RUN_TRANSITIONS[src], src


# --- Cancellation primitive (§8.5) -------------------------------------------


def test_cancellation_token_surface_snapshot() -> None:
    # §8.5: cancellation must reach the execution points via this token.
    # Freeze the method surface so a refactor that silently guts the token
    # (e.g. drops raise_if_cancelled) is caught instead of passing vacuously.
    from linktools.ai.run.cancellation import CancellationToken

    for method in ("cancel", "is_cancelled", "raise_if_cancelled"):
        assert callable(getattr(CancellationToken, method, None)), method


# --- Tool idempotency status (§9.2 will extend) ------------------------------


def test_idempotency_status_values_snapshot() -> None:
    # §9.2 landed: EXECUTED (Handler returned, never re-run; result held as the
    # execution receipt) and UNKNOWN (commit could not be confirmed after the
    # Handler ran) now separate "side effect happened" from "result committed".
    from linktools.ai.tool.idempotency import IdempotencyStatus

    values = {s.value for s in IdempotencyStatus}
    assert values == {
        "reserved",
        "executed",
        "completed",
        "failed",
        "unknown",
    }, values


# --- File storage atomicity (§16 will promote) -------------------------------


def test_atomic_write_helper_is_the_public_module() -> None:
    # §16.1 landed: the file-store's atomic-write helper is now the public
    # ``atomic_write_bytes`` in storage/file/atomic.py (temp + fsync + os.replace
    # + parent-directory fsync). The historical private ``_atomic_write`` in
    # _util.py remains as a thin delegate so every existing store import keeps
    # working.
    from linktools.ai.storage.file import atomic, _util

    assert importlib.util.find_spec("linktools.ai.storage.file.atomic") is not None
    assert callable(atomic.atomic_write_bytes)
    # The private name still exists for back-compat and delegates to the public one.
    assert callable(_util._atomic_write)


# --- Run record shape (§8.2 / §8.7 / §13 will extend) ------------------------


def test_run_record_has_cancel_audit_and_fencing_fields() -> None:
    # §8.2 landed: cancel_requested_at / _by / reason are present (audit). Still
    # absent (snapshot so a future addition is visible): §8.7 execution_token /
    # heartbeat_at / worker_id (distributed-worker fencing) and §13 manifest_id
    # / resumability (deterministic resume).
    from linktools.ai.run.models import RunRecord

    fields = {f.name for f in dataclasses.fields(RunRecord)}
    for landed in ("cancel_requested_at", "cancel_requested_by", "cancel_reason"):
        assert landed in fields, landed
    for landed_field in (
        "execution_token",
        "heartbeat_at",
        "worker_id",
        "manifest_id",
        "resumability",
    ):
        assert landed_field in fields, landed_field
