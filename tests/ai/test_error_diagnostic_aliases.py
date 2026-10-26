#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Exception identity references preserve complete failures without duplicate expansion."""

import json
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.test import TestModel

from linktools.ai.capability import TraceCapability
from linktools.ai.core import ExecutionEventType, ExecutionStatus
from linktools.ai.errors import ErrorDiagnostics
from linktools.ai.model import FallbackBinding
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup


def _raised(error: BaseException) -> BaseException:
    try:
        raise error
    except BaseException as captured:
        return captured


@pytest.mark.parametrize("omission", (None, "referenced_content_lifecycle"))
@pytest.mark.parametrize("count", (1, 3, 11, 41))
def test_shared_group_nodes_emit_frames_once_and_retain_every_branch(omission: str | None, count: int) -> None:
    error = _raised(RuntimeError("shared synthetic failure"))
    before = error.__traceback__
    group = ExceptionGroup("failed attempts", [error] * count)
    diagnostics = ErrorDiagnostics.from_exception(group, content_omission_reason=omission)
    chain = diagnostics.exception_chain
    assert len(chain) == count + 1
    assert chain[1]["path"] == "root.branch[0]"
    assert "same_exception_as" not in chain[1]
    assert [entry["same_exception_as"] for entry in chain[2:]] == ["root.branch[0]"] * (count - 1)
    assert diagnostics.traceback.count('File "') == len(list(traceback.walk_tb(before)))
    assert error.__traceback__ is before
    assert group.exceptions == (error,) * count
    assert decode_domain(encode_domain(diagnostics), ErrorDiagnostics) == diagnostics


def test_equal_messages_and_frames_do_not_merge_distinct_exceptions() -> None:
    errors = [_raised(RuntimeError("same synthetic message")) for _ in range(11)]
    diagnostics = ErrorDiagnostics.from_exception(ExceptionGroup("distinct attempts", errors))
    assert len(diagnostics.exception_chain) == 12
    assert not any("same_exception_as" in entry for entry in diagnostics.exception_chain)
    assert diagnostics.traceback.count("same synthetic message") == 11
    assert diagnostics.traceback.count('File "') == 11


@pytest.mark.parametrize("relation", ("__cause__", "__context__"))
def test_cycles_are_explicit_aliases_not_recursive_expansion(relation: str) -> None:
    first, second = RuntimeError("first"), ValueError("second")
    setattr(first, relation, second)
    setattr(second, relation, first)
    diagnostics = ErrorDiagnostics.from_exception(first)
    assert [entry["exception_type"] for entry in diagnostics.exception_chain] == [
        "RuntimeError", "ValueError", "RuntimeError",
    ]
    assert diagnostics.exception_chain[-1]["same_exception_as"] == "root"
    assert diagnostics.traceback.count("ValueError: second") == 1
    assert "same exception as root" in diagnostics.traceback


@pytest.mark.parametrize("omission", (None, "referenced_content_lifecycle"))
def test_deep_shared_groups_do_not_enter_recursive_formatter(omission: str | None) -> None:
    error: BaseException = RuntimeError("deep shared leaf")
    for _ in range(40):
        error = ExceptionGroup("shared branches", [error, error])
    diagnostics = ErrorDiagnostics.from_exception(error, content_omission_reason=omission)
    assert len(diagnostics.exception_chain) == 81
    assert sum("same_exception_as" in entry for entry in diagnostics.exception_chain) == 40
    assert diagnostics.traceback is not None
    assert len(diagnostics.traceback) < 40000


def test_chain_beyond_recursion_limit_keeps_the_last_cause() -> None:
    error: BaseException = RuntimeError("deepest unique cause")
    for _ in range(sys.getrecursionlimit() + 10):
        outer = RuntimeError("outer unique cause")
        outer.__cause__ = error
        error = outer
    diagnostics = ErrorDiagnostics.from_exception(error)
    assert len(diagnostics.exception_chain) == sys.getrecursionlimit() + 11
    assert diagnostics.exception_chain[-1]["exception_message"] == "deepest unique cause"
    assert diagnostics.traceback is not None and "deepest unique cause" in diagnostics.traceback


@pytest.mark.skipif(sys.version_info < (3, 11), reason="exception notes require Python 3.11")
def test_large_message_notes_and_nested_body_are_captured_once_and_redacted() -> None:
    class NamedFailure(RuntimeError):
        calls = 0

        def __str__(self) -> str:
            self.calls += 1
            return super().__str__()

    body = json.dumps({"detail": "BODY_MARKER_" + "z" * 100000,
                       "headers": [{"name": "Authorization", "value": "Bearer synthetic-secret"}]})
    error = NamedFailure(body)
    error.add_note("NOTE_MARKER_" + "n" * 100000)
    diagnostic = ErrorDiagnostics.from_exception(ExceptionGroup("shared body", [error] * 41))
    assert error.calls == 1
    assert diagnostic.traceback.count("BODY_MARKER_") == 1
    assert diagnostic.traceback.count("NOTE_MARKER_") == 1
    assert "n" * 100000 in diagnostic.traceback
    assert "z" * 100000 in diagnostic.traceback
    assert "synthetic-secret" not in json.dumps(diagnostic.to_payload())
    assert "NamedFailure" in diagnostic.traceback


@pytest.mark.parametrize("source", ("value = (\n", "value = [1 2]\n", None))
def test_syntax_error_locations_notes_and_original_state_survive_aliasing(source: str | None) -> None:
    if source is None:
        error = SyntaxError("multiline marker", ("synthetic-input.py", 1, 2, "first\nsecond", 2, 3))
    else:
        try:
            compile(source, "synthetic-input.py", "exec")
        except SyntaxError as caught:
            error = caught
    if sys.version_info >= (3, 11):
        error.add_note("synthetic syntax note")
    original_notes = list(error.__notes__) if sys.version_info >= (3, 11) else None
    original = (error.args, error.__traceback__, error.__cause__, original_notes)
    expected = "".join(traceback.format_exception_only(type(error), error))
    positions = (error.lineno, error.offset, error.end_lineno, error.end_offset)
    diagnostic = ErrorDiagnostics.from_exception(ExceptionGroup("syntax", [error, error]))
    assert 'File "synthetic-input.py", line 1' in diagnostic.traceback
    assert expected in diagnostic.traceback and "^" in diagnostic.traceback
    assert (error.lineno, error.offset, error.end_lineno, error.end_offset) == positions
    if sys.version_info >= (3, 11):
        assert diagnostic.traceback.count("synthetic syntax note") == 1
    current_notes = error.__notes__ if sys.version_info >= (3, 11) else None
    assert (error.args, error.__traceback__, error.__cause__, current_notes) == original


@pytest.mark.asyncio
async def test_runtime_shared_failures_preserve_attempts_terminal_event_and_reopen(tmp_path: Path) -> None:
    shared = ModelHTTPError(503, "offline", "shared synthetic provider failure")
    calls: list[int] = []

    class Provider(TestModel):
        @asynccontextmanager
        async def request_stream(
            self, messages: object, model_settings: object, model_request_parameters: ModelRequestParameters,
            run_context: object = None,
        ) -> AsyncIterator[StreamedResponse]:
            calls.append(1)
            raise shared
            yield  # pragma: no cover

    class Binding:
        provider = "offline"
        model_identity = "offline:shared-failure"
        contract = {"provider": "offline"}

        def materialize(self) -> Provider:
            return Provider()

    features = RuntimeFeatures("alias-persistence")
    features.agent("default", allow_tools=())
    features.model("default", FallbackBinding(*(Binding() for _ in range(11))))
    features.capability(TraceCapability(), scope="runtime")
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Persistence.open('alias-persistence', storage=storage) as persistence_176, Runtime.open(persistence=persistence_176, features=features) as runtime:
        result = (await runtime.agents.get().run("synthetic offline failure", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        diagnostics = result.error_diagnostics
        aliases = [entry for entry in diagnostics.exception_chain if "same_exception_as" in entry]
        assert len(aliases) == 10
        assert diagnostics.traceback.count('File "') < 200
        attempts = result.safe_error_details.get("failed_attempts")
        if attempts is not None:  # Older supported SDKs do not expose this observation.
            assert len(attempts) == 11
        events = await runtime.events.list(result.execution_id, principal=runtime.default_principal, limit=100)
        terminal = next(event for event in events.items if event.event_type == ExecutionEventType.EXECUTION_FAILED)
        assert terminal.payload["error_diagnostics"] == diagnostics.to_payload()
        principal = runtime.default_principal
    async with Persistence.open('alias-persistence', storage=storage) as persistence_190, Runtime.open(persistence=persistence_190, features=features) as runtime:
        restored = await runtime.executions.result(result.execution_id, principal=principal)
        assert restored.error_diagnostics == diagnostics
    assert len(calls) == 11


def test_unreadable_notes_and_string_leave_other_nodes_and_root_diagnostic_intact() -> None:
    class BadNotes(RuntimeError):
        @property
        def __notes__(self) -> object:
            raise RuntimeError("untrusted-note-secret")

    class BadMessage(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("untrusted-message-secret")

    bad = BadNotes("bad note owner")
    group = ExceptionGroup("root retained", [
        _raised(ValueError("before intact")), bad, _raised(BadMessage()),
        _raised(TypeError("after intact")), bad,
    ])
    diagnostic = ErrorDiagnostics.from_exception(group)
    assert diagnostic.exception_message == "root retained (5 sub-exceptions)"
    assert "before intact" in diagnostic.traceback and "after intact" in diagnostic.traceback
    assert diagnostic.traceback.count("traceback formatting unavailable: RuntimeError") == 1
    assert diagnostic.exception_chain[3]["exception_message"] == ""
    assert "untrusted-note-secret" not in repr(diagnostic)
    assert "untrusted-message-secret" not in repr(diagnostic)
    assert diagnostic.exception_chain[-1]["same_exception_as"] == "root.branch[1]"


def test_broken_syntax_position_does_not_discard_other_nodes() -> None:
    class InvalidLine:
        def __str__(self) -> str:
            raise ValueError("untrusted-position-secret")

    bad = SyntaxError("invalid position")
    bad.lineno = InvalidLine()
    diagnostic = ErrorDiagnostics.from_exception(ExceptionGroup("syntax nodes", [bad, ValueError("retained")]))
    assert "traceback formatting unavailable:" in diagnostic.traceback
    assert "ValueError: retained" in diagnostic.traceback
    assert "untrusted-position-secret" not in repr(diagnostic)


def test_custom_group_type_and_message_are_not_replaced_by_snapshot_type() -> None:
    class ProviderFailures(ExceptionGroup):
        def __str__(self) -> str:
            return "custom provider failures"

    leaf = RuntimeError("group child")
    group = ProviderFailures("original args", [leaf, leaf])
    diagnostic = ErrorDiagnostics.from_exception(group)
    assert diagnostic.exception_type == "ProviderFailures"
    assert diagnostic.exception_message == "custom provider failures"
    assert "ProviderFailures: custom provider failures" in diagnostic.traceback
    assert "Exception: custom provider failures" not in diagnostic.traceback
    assert diagnostic.traceback.count("RuntimeError: group child") == 1
    assert group.args == ("original args", [leaf, leaf])
