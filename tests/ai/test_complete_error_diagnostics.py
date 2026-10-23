#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete, secret-safe failure capture and durable public reads."""

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx2 import RemoteProtocolError
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.function import FunctionModel
from linktools.core import environ
from linktools.core._logging import LoggingManager

from linktools.ai.core import ExecutionEventType, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.runtime import Runtime, RuntimeFeatures, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.storage import FilesystemObjectStore

from ._runtime_test_helpers import model_features


def _raised_failure() -> BaseException:
    try:
        try:
            raise RemoteProtocolError("incomplete chunked read " + "x" * 5000 + " END_TRANSPORT")
        except RemoteProtocolError as cause:
            raise ModelAPIError("test-model", "model connection failed") from cause
    except ModelAPIError as error:
        return error


def test_complete_chain_keeps_long_messages_frames_and_no_locals(monkeypatch: pytest.MonkeyPatch) -> None:
    hidden_local = "local-only-secret-not-in-any-exception"
    monkeypatch.setattr(sys, "tracebacklimit", 1, raising=False)
    diagnostics = ErrorDiagnostics.from_exception(_raised_failure())
    assert [cause["relation"] for cause in diagnostics.exception_chain] == ["root", "cause"]
    assert [cause["exception_type"] for cause in diagnostics.exception_chain] == [
        "ModelAPIError", "RemoteProtocolError",
    ]
    assert diagnostics.exception_chain[1]["exception_message"].endswith("END_TRANSPORT")
    assert diagnostics.traceback.count("_raised_failure") == 2
    assert "x" * 5000 in diagnostics.traceback
    assert hidden_local not in diagnostics.traceback
    assert "direct cause" in diagnostics.traceback


def test_deep_chain_cycles_and_suppressed_context_have_truthful_boundaries() -> None:
    error = RuntimeError("leaf")
    for index in range(15):
        outer = RuntimeError(f"outer-{index}")
        outer.__cause__ = error
        error = outer
    diagnostics = ErrorDiagnostics.from_exception(error)
    assert len(diagnostics.exception_chain) == 16
    assert diagnostics.exception_chain[-1]["exception_message"] == "leaf"
    error.__cause__.__cause__ = error
    assert len(ErrorDiagnostics.from_exception(error).exception_chain) == 2
    suppressed = RuntimeError("visible")
    suppressed.__context__ = RuntimeError("hidden-context")
    suppressed.__suppress_context__ = True
    visible = ErrorDiagnostics.from_exception(suppressed)
    assert len(visible.exception_chain) == 1
    assert "hidden-context" not in visible.traceback


@pytest.mark.skipif(sys.version_info < (3, 11), reason="native exception groups require Python 3.11")
def test_group_traceback_does_not_apply_display_width_or_depth_limits() -> None:
    import builtins

    error = builtins.ExceptionGroup("many", [ValueError(f"branch-{index}") for index in range(20)])
    for index in range(12):
        error = builtins.ExceptionGroup(f"level-{index}", [error])
    diagnostics = ErrorDiagnostics.from_exception(error)
    for index in range(20):
        assert f"branch-{index}" in diagnostics.traceback
    assert "max_group_depth" not in diagnostics.traceback


def test_diagnostics_filter_constructor_and_collected_nested_secrets() -> None:
    body = {
        "headers": [["Authorization", "Bearer raw-header-secret"], {"name": "Cookie", "value": "session=raw-cookie-secret"}],
        "url": "https://user:raw-url-secret@host/path?api%5Fkey=raw-query-secret&status=502",
        "payload": json.dumps({"api_key": "raw-json-secret", "status": "failed"}),
        "detail": "z" * 3000 + "END_DIAGNOSTIC",
    }
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError(json.dumps(body)))
    rendered = repr(diagnostics)
    for secret in ["raw-header-secret", "raw-cookie-secret", "raw-url-secret", "raw-query-secret", "raw-json-secret"]:
        assert secret not in rendered
    assert "END_DIAGNOSTIC" in diagnostics.exception_chain[0]["exception_message"]
    assert "END_DIAGNOSTIC" in diagnostics.traceback
    direct = ErrorDiagnostics("Failure", "api_key=manual-secret", "a" * 64, traceback="Authorization: Bearer stack-secret")
    assert "manual-secret" not in repr(direct)
    assert "stack-secret" not in repr(direct)


def test_diagnostic_summary_redacts_secrets_before_display_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = LoggingManager()
    secret = "sensitive-marker-1234567890"
    manager.register_secret(secret)
    monkeypatch.setattr(environ.logging, "redact", manager.redact)
    diagnostics = ErrorDiagnostics.from_exception(RuntimeError("x" * 2040 + secret))
    assert diagnostics.exception_message == "x" * 2040 + "***"
    assert "sensitive" not in repr(diagnostics)


def test_optional_diagnostics_round_trip_and_prior_records_remain_unknown() -> None:
    diagnostics = ErrorDiagnostics.from_exception(_raised_failure())
    encoded = encode_domain(diagnostics)
    assert decode_domain(encoded, ErrorDiagnostics) == diagnostics
    encoded["fields"].pop("exception_chain")
    encoded["fields"].pop("traceback")
    restored = decode_domain(encoded, ErrorDiagnostics)
    assert restored.exception_chain is None
    assert restored.traceback is None
    assert restored.cause_digest == diagnostics.cause_digest
    encoded["fields"]["traceback"] = 42
    with pytest.raises(AIError) as failed:
        decode_domain(encoded, ErrorDiagnostics)
    assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


class _FailingModel:
    route_id = "default"
    provider = "test"
    model_identity = "test:complete-diagnostics"
    contract = {"provider": "test", "model": "complete-diagnostics"}

    def materialize(self) -> FunctionModel:
        async def stream(*_args: object) -> AsyncIterator[str]:
            yield "partial answer"
            raise _raised_failure()

        return FunctionModel(stream_function=stream)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["filesystem", "sqlite"])
async def test_actual_failure_full_diagnostics_survive_close_and_public_reopen(tmp_path: Path, backend: str) -> None:
    def storage() -> RuntimeStorage:
        if backend == "filesystem":
            return RuntimeStorage.filesystem(tmp_path / "runtime")
        return RuntimeStorage.sqlite(tmp_path / "runtime.db", object_store=FilesystemObjectStore(tmp_path / "objects"))

    group = RuntimeFeatures("complete-diagnostics")
    group.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Runtime.open("complete-diagnostics", features=model_features(_FailingModel(), group), storage=storage()) as runtime:
        result = (await runtime.agents.get().run("test failure", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.MODEL_API_ERROR.value
        diagnostics = result.error_diagnostics
        assert diagnostics.exception_chain[-1]["exception_type"] == "RemoteProtocolError"
        assert "END_TRANSPORT" in diagnostics.traceback
        events = await runtime.events.list(result.execution_id, principal=runtime.default_principal, limit=100)
        terminal = next(event for event in events.items if event.event_type == ExecutionEventType.EXECUTION_FAILED)
        assert terminal.payload["error_diagnostics"]["traceback"] == diagnostics.traceback
        assert terminal.payload["error_diagnostics"]["exception_chain"][-1]["exception_message"].endswith("END_TRANSPORT")
        principal = runtime.default_principal
    async with RuntimeHistory.open("complete-diagnostics", storage=storage()) as history:
        restored = await history.result(result.execution_id, principal=principal)
        assert restored.error_diagnostics == diagnostics
        summary = await history.inspect_execution(result.execution_id, principal=principal)
        assert summary.error_diagnostics is None
