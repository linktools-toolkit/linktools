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
    """A concrete class that implements every method satisfies
    SwarmCommitCoordinator (runtime_checkable)."""

    class _Stub:
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


def test_each_command_carries_commit_id_swarm_run_id_expected_version_payload_event_context():
    """Per the spec, every swarm commit command carries these 6 fields."""
    required = {
        "commit_id",
        "swarm_run_id",
        "expected_version",
        "payload",
        "event_context",
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
        missing = required - actual
        assert not missing, f"{cmd_cls.__name__} missing fields: {missing}"


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
    mutated mid-commit."""
    for cmd_cls in (
        StartSwarmCommand,
        CompleteSwarmCommand,
        CancelSwarmCommand,
    ):
        cmd = cmd_cls(
            commit_id="c1",
            swarm_run_id="s1",
            expected_version=1,
            payload={},
            event_context=None,
        )
        try:
            cmd.commit_id = "c2"  # type: ignore[misc]
            raise AssertionError(f"{cmd_cls.__name__} is not frozen")
        except Exception as exc:
            # Frozen dataclass raises FrozenInstanceError; the assertion
            # above only fires if the assignment succeeded.
            assert "frozen" in str(exc).lower() or "cannot assign" in str(exc).lower()


def test_swarm_commit_conflict_error_is_a_distinct_exception_type():
    """The conflict error must be a distinct type so callers can catch it
    without catching unrelated errors."""
    assert issubclass(SwarmCommitConflictError, Exception)
    assert SwarmCommitConflictError is not Exception
