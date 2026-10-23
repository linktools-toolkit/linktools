#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Omitted-content diagnostics retain structure without inspecting failure data."""

import base64
import hashlib
import json
import linecache
import sys
import traceback
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

from linktools.ai.errors import ErrorDiagnostics


_REASON = "referenced_content_lifecycle"
_BODY = "synthetic-private-body-value"
_BASE64 = base64.b64encode(_BODY.encode()).decode()
_NOTE = "synthetic-private-note-value"
_LITERAL = "synthetic-private-source-literal"
_ACCESSES: list[str] = []


class _UnreadableContent:
    def __str__(self) -> str:
        _ACCESSES.append("str")
        raise AssertionError("original exception must not be formatted")

    def __getattribute__(self, name: str) -> object:
        if name in {"__notes__", "args", "body", "payload"}:
            _ACCESSES.append(name)
            raise AssertionError("original exception content must not be read")
        return super().__getattribute__(name)


class _UnreadableError(_UnreadableContent, RuntimeError):
    pass


class _UnreadableGroup(_UnreadableContent, ExceptionGroup):
    pass


def _raised_error(message: str) -> BaseException:
    try:
        hidden_local = {"body": message, "base64": _BASE64}
        raise _UnreadableError(hidden_local)
    except _UnreadableError as error:
        error.body = message
        error.payload = _BASE64
        error.__notes__ = [_NOTE]
        return error


def test_content_omission_never_formats_original_messages_notes_locals_or_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic_failure.py"
    source.write_text(f"def fail(error):\n    raise error  # {_LITERAL}\n", encoding="utf-8")
    namespace = {}
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)
    first = _raised_error(_BODY)
    second = _raised_error(_BASE64)
    first.__context__ = second
    group = _UnreadableGroup(_BODY, [first, second])
    group.__notes__ = [_NOTE]
    group.__cause__ = first
    try:
        namespace["fail"](group)
    except _UnreadableGroup as error:
        original = error
    _ACCESSES.clear()

    def forbidden_formatter(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("content-aware traceback formatting is forbidden")

    monkeypatch.setattr(traceback, "TracebackException", forbidden_formatter)
    monkeypatch.setattr(linecache, "getline", forbidden_formatter)
    monkeypatch.setattr(linecache, "getlines", forbidden_formatter)
    monkeypatch.setattr(sys, "tracebacklimit", 0, raising=False)
    diagnostics = ErrorDiagnostics.from_exception(original, content_omission_reason=_REASON)

    assert _ACCESSES == []
    assert diagnostics.exception_type == "_UnreadableGroup"
    assert diagnostics.exception_message == f"[content omitted: {_REASON}]"
    assert diagnostics.content_omission_reason == _REASON
    assert diagnostics.to_payload()["content_omission_reason"] == _REASON
    chain = diagnostics.exception_chain
    assert [entry["path"] for entry in chain] == [
        "root", "cause", "cause.context", "root.branch[0]", "root.branch[1]",
    ]
    assert [entry["relation"] for entry in chain] == ["root", "cause", "context", "branch", "branch"]
    assert chain[3]["same_exception_as"] == "cause"
    assert chain[4]["same_exception_as"] == "cause.context"
    assert f'File "{source}", line 2, in fail\n' in diagnostics.traceback
    assert "in _raised_error\n" in diagnostics.traceback
    assert "in test_content_omission_never_formats_original_messages_notes_locals_or_source\n" in diagnostics.traceback
    serialized = json.dumps(diagnostics.to_payload())
    assert all(value not in serialized for value in (_BODY, _BASE64, _NOTE, _LITERAL))


def test_content_omission_retains_deep_wide_graphs_and_cycle_references() -> None:
    leaf = ValueError(_BODY)
    error: BaseException = leaf
    for _index in range(1100):
        outer = RuntimeError(_BASE64)
        outer.__cause__ = error
        error = outer
    leaf.__cause__ = error
    group = ExceptionGroup(_BODY, [error] + [ValueError(_BODY) for _index in range(25)])
    diagnostics = ErrorDiagnostics.from_exception(group, content_omission_reason=_REASON)
    chain = diagnostics.exception_chain
    assert len(chain) == 1128
    assert chain[1102]["same_exception_as"] == "root.branch[0]"
    assert chain[-1]["path"] == "root.branch[25]"
    assert f"root.branch[25]: ValueError: [content omitted: {_REASON}]" in diagnostics.traceback
    assert _BODY not in diagnostics.traceback and _BASE64 not in diagnostics.traceback


def test_content_omission_preserves_explicit_cause_and_context_suppression() -> None:
    root = RuntimeError(_BODY)
    root.__context__ = TypeError(_BASE64)
    root.__suppress_context__ = True
    suppressed = ErrorDiagnostics.from_exception(root, content_omission_reason=_REASON)
    assert [entry["exception_type"] for entry in suppressed.exception_chain] == ["RuntimeError"]
    root.__cause__ = ValueError(_BODY)
    explicit = ErrorDiagnostics.from_exception(root, content_omission_reason=_REASON)
    assert [entry["exception_type"] for entry in explicit.exception_chain] == ["RuntimeError", "ValueError"]
    assert explicit.exception_chain[-1]["relation"] == "cause"


def test_content_omission_digest_depends_on_safe_structure_not_original_values() -> None:
    first = RuntimeError(_BODY)
    first.__notes__ = [_NOTE]
    first.__cause__ = ValueError(_BASE64)
    second = RuntimeError("different-private-body")
    second.__notes__ = ["different-private-note"]
    second.__cause__ = ValueError("different-private-base64")
    left = ErrorDiagnostics.from_exception(first, content_omission_reason=_REASON)
    right = ErrorDiagnostics.from_exception(second, content_omission_reason=_REASON)
    assert left == right
    second.__cause__ = TypeError("different-private-body")
    assert ErrorDiagnostics.from_exception(second, content_omission_reason=_REASON).cause_digest != left.cause_digest


def test_ordinary_diagnostics_keep_original_messages_notes_and_digest_semantics() -> None:
    error = RuntimeError("ordinary diagnostic message")
    error.__notes__ = ["ordinary diagnostic note"]
    error.__cause__ = ValueError("ordinary cause")
    original = ErrorDiagnostics.from_exception(error)
    explicit = ErrorDiagnostics.from_exception(error, content_omission_reason=None)
    assert original == explicit
    assert original.exception_message == "ordinary diagnostic message"
    assert original.exception_chain[-1]["exception_message"] == "ordinary cause"
    assert "ordinary diagnostic message" in original.traceback
    if sys.version_info >= (3, 11):
        assert "ordinary diagnostic note" in original.traceback
    assert original.content_omission_reason is None
    assert "content_omission_reason" not in original.to_payload()
    expected = json.dumps({
        "exception_type": "RuntimeError", "exception_message": "ordinary diagnostic message",
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert original.cause_digest == hashlib.sha256(expected.encode()).hexdigest()


def test_content_omission_reason_round_trips_without_becoming_required_on_older_records() -> None:
    from linktools.ai.runtime.state._codec import decode_domain, encode_domain

    diagnostics = ErrorDiagnostics.from_exception(_raised_error(_BODY), content_omission_reason=_REASON)
    encoded = encode_domain(diagnostics)
    assert decode_domain(encoded, ErrorDiagnostics) == diagnostics
    del encoded["fields"]["content_omission_reason"]
    restored = decode_domain(encoded, ErrorDiagnostics)
    assert restored.content_omission_reason is None
    assert restored.exception_chain == diagnostics.exception_chain
    assert restored.traceback == diagnostics.traceback


@pytest.mark.parametrize("reason", ("", " \t", 1))
def test_content_omission_requires_a_nonempty_diagnostic_label(reason: object) -> None:
    with pytest.raises(ValueError, match="nonempty string"):
        ErrorDiagnostics.from_exception(RuntimeError(_BODY), content_omission_reason=reason)
