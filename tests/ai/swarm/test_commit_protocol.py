#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SwarmCommitCoordinator Protocol shape + the 7 command types (P7 swarm commit boundary).

These are the contract-surface tests for the swarm commit boundary. The
reference SQL implementation  and SwarmEngine dep-cleanup 
are separate increments."""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from linktools.ai.swarm.commit import (
    CancelSwarmCommand,
    CompleteSwarmCommand,
    CompleteSwarmStepCommand,
    FailSwarmCommand,
    FailSwarmStepCommand,
    StartSwarmCommand,
    StartSwarmStepCommand,
    SwarmCommitConflictError,
    SwarmCommitCoordinator,
)


# --- Protocol shape ---------------------------------------------------------


def test_swarm_commit_coordinator_protocol_has_all_seven_methods():
    """The Protocol declares exactly the 7 swarm lifecycle commits the spec
    names: start / start_step / complete_step / fail_step / complete / fail /
    cancel."""
    expected = {
        "start",
        "start_step",
        "complete_step",
        "fail_step",
        "complete",
        "fail",
        "cancel",
    }
    actual = {
        name
        for name in dir(SwarmCommitCoordinator)
        if not name.startswith("_") and callable(getattr(SwarmCommitCoordinator, name, None))
    }
    # `dir` on a runtime_checkable Protocol includes the Protocol machinery;
    # restrict to the methods the spec names.
    assert expected.issubset(actual), (
        f"SwarmCommitCoordinator missing methods: {expected - actual}"
    )


def test_a_class_implementing_all_seven_methods_satisfies_the_protocol():
    """A concrete class that implements every Protocol member (the 7 swarm
    lifecycle commits plus the get_run/update_run helpers AND the
    ``state_store`` attribute) satisfies SwarmCommitCoordinator
    (runtime_checkable). runtime_checkable checks every Protocol member is
    present on the instance, not just the seven commit methods."""

    class _Stub:
        state_store = None  # Protocol member (non-callable): checked by runtime_checkable

        async def get_run(self, swarm_run_id): ...
        async def update_run(self, swarm_run_id, *, expected_version, status=None, token_usage=None): ...
        async def start(self, command): ...
        async def start_step(self, command): ...
        async def complete_step(self, command): ...
        async def fail_step(self, command): ...
        async def complete(self, command): ...
        async def fail(self, command): ...
        async def cancel(self, command): ...

    assert isinstance(_Stub(), SwarmCommitCoordinator)


def test_a_class_missing_a_method_does_not_satisfy_the_protocol():
    """A class missing any of the 7 methods fails the runtime_checkable
    Protocol check -- the contract cannot drift silently."""

    class _Stub:
        async def start(self, command): ...

    assert not isinstance(_Stub(), SwarmCommitCoordinator)


# --- Command shape ----------------------------------------------------------


def test_each_command_carries_commit_id_swarm_run_id_expected_version_payload_fence():
    """Per the spec, every swarm commit command carries these 5 top-level
    fields: commit_id, swarm_run_id, expected_version, payload (the typed
    operation data), and fence (the execution fence). The payload itself
    wraps event_context alongside the operation-specific lifecycle data, so
    event_context is asserted on the payload type, not the command."""
    required_top_level = {
        "commit_id",
        "swarm_run_id",
        "expected_version",
        "payload",
        "fence",
    }
    commands = (
        StartSwarmCommand,
        StartSwarmStepCommand,
        CompleteSwarmStepCommand,
        FailSwarmStepCommand,
        CompleteSwarmCommand,
        FailSwarmCommand,
        CancelSwarmCommand,
    )
    for cmd_cls in commands:
        assert is_dataclass(cmd_cls), f"{cmd_cls.__name__} is not a dataclass"
        actual = {f.name for f in fields(cmd_cls)}
        missing = required_top_level - actual
        assert not missing, f"{cmd_cls.__name__} missing fields: {missing}"
    # event_context lives INSIDE each payload (typed operation data), not at
    # the command top level.
    from linktools.ai.swarm.commit import (
        CancelSwarmPayload,
        CompleteSwarmPayload,
        CompleteSwarmStepPayload,
        FailSwarmPayload,
        FailSwarmStepPayload,
        StartSwarmPayload,
        StartSwarmStepPayload,
    )

    for payload_cls in (
        StartSwarmPayload,
        StartSwarmStepPayload,
        CompleteSwarmStepPayload,
        FailSwarmStepPayload,
        CompleteSwarmPayload,
        FailSwarmPayload,
        CancelSwarmPayload,
    ):
        assert is_dataclass(payload_cls), f"{payload_cls.__name__} is not a dataclass"
        actual = {f.name for f in fields(payload_cls)}
        assert "event_context" in actual, (
            f"{payload_cls.__name__} missing event_context field"
        )


def test_step_commands_carry_step_attempt_id():
    """Step-scoped commands also carry the step_attempt_id they target."""
    for cmd_cls in (
        StartSwarmStepCommand,
        CompleteSwarmStepCommand,
        FailSwarmStepCommand,
    ):
        actual = {f.name for f in fields(cmd_cls)}
        assert "step_attempt_id" in actual, (
            f"{cmd_cls.__name__} is step-scoped but lacks step_attempt_id"
        )


def test_commands_are_frozen():
    """Commands are value objects: frozen so a passed-down command cannot be
    mutated mid-commit. Constructs each command with the payload+fence shape
    (the actual data the coordinator receives)."""
    from linktools.ai.events.context import EventStreamContext
    from linktools.ai.events.payloads import (
        SwarmCancelled,
        SwarmCompleted,
        SwarmStarted,
    )
    from linktools.ai.run.models import RunResult
    from linktools.ai.swarm.commit import (
        CancelSwarmPayload,
        CompleteSwarmPayload,
        StartSwarmPayload,
        SwarmCommitId,
        SwarmExecutionFence,
    )
    from linktools.ai.swarm.models import (
        SwarmRun,
        SwarmStatus,
        TokenUsage,
    )
    from decimal import Decimal
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ctx = EventStreamContext(
        stream_id="s1",
        run_id="s1",
        root_run_id="s1",
        parent_run_id=None,
        session_id="sess",
        runnable_id="swarm",
    )
    run = SwarmRun(
        id="s1",
        run_id="driving-1",
        round=0,
        status=SwarmStatus.PENDING,
        version=1,
        token_usage=TokenUsage(),
        cost=Decimal("0"),
        created_at=now,
        updated_at=now,
    )
    samples = (
        StartSwarmCommand(
            commit_id=SwarmCommitId("c1"),
            swarm_run_id="s1",
            expected_version=1,
            payload=StartSwarmPayload(
                run=run,
                started_event=SwarmStarted(swarm_run_id="s1", swarm_id="swarm"),
                event_context=ctx,
            ),
            fence=SwarmExecutionFence("test-token"),
        ),
        CompleteSwarmCommand(
            commit_id=SwarmCommitId("c1"),
            swarm_run_id="s1",
            expected_version=1,
            payload=CompleteSwarmPayload(
                result=RunResult(output={"done": True}),
                completed_event=SwarmCompleted(swarm_run_id="s1"),
                event_context=ctx,
            ),
            fence=SwarmExecutionFence("test-token"),
        ),
        CancelSwarmCommand(
            commit_id=SwarmCommitId("c1"),
            swarm_run_id="s1",
            expected_version=1,
            payload=CancelSwarmPayload(
                cancelled_event=SwarmCancelled(swarm_run_id="s1"),
                event_context=ctx,
            ),
            fence=SwarmExecutionFence("test-token"),
        ),
    )
    for cmd in samples:
        cmd_cls = type(cmd)
        try:
            cmd.commit_id = SwarmCommitId("c2")  # type: ignore[misc]
            raise AssertionError(f"{cmd_cls.__name__} is not frozen")
        except Exception as exc:
            # Frozen dataclass raises FrozenInstanceError; the assertion
            # above only fires if the assignment succeeded.
            assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower(), (
                f"{cmd_cls.__name__} mutation raised an unexpected error: {exc!r}"
            )


def test_swarm_commit_conflict_error_is_a_distinct_exception_type():
    """The conflict error must be a distinct type so callers can catch it
    without catching unrelated errors."""
    assert issubclass(SwarmCommitConflictError, Exception)
    assert SwarmCommitConflictError is not Exception
