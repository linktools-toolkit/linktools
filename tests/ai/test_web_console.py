#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Public Web transport parity, local trust boundary, and observer lifetime."""
from ._runtime_test_helpers import model_features

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

import httpx
import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from linktools.ai.errors import ErrorDiagnostics
from linktools.ai.observe import Metrics
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability, TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.web import create_app
import linktools.commands.ai._common as common_command
from ._runtime_test_helpers import RuntimeUsageModels, _persistence_capabilities

_HEADERS = {"X-LinkTools-Console": "1"}


def _turn_text(turn):
    if "output" in turn:
        return turn["output"]["text"]
    return next(item["content"] for item in reversed(turn["items"]) if item["item_kind"] == "assistant")


def client(app: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765")



def _no_published_turn(*_args, **_kwargs):
    async def read():
        return None
    return nullcontext(read)


@pytest.mark.asyncio
async def test_session_events_expose_only_authorized_occupancy() -> None:
    from types import SimpleNamespace

    from collections.abc import AsyncIterator

    from linktools.ai.core import Principal, SessionStatus
    from linktools.ai.errors import AIError, ErrorCode

    principal = Principal("runtime", "default")

    async def watch(session_id: str, *, principal: Principal) -> AsyncIterator[SimpleNamespace]:
        assert principal.principal_id == "runtime"
        if session_id != "chosen":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        for revision, execution_id in enumerate((None, "running", None)):
            yield SimpleNamespace(status=SessionStatus.OPEN, active_execution_id=execution_id, revision=revision,
                                  metadata={"title": "private"})

    runtime = SimpleNamespace(history=None, default_principal=principal, sessions=SimpleNamespace(supports_turn_sessions=False, watch=watch))
    async with client(create_app(runtime=runtime)) as http:
        response = await http.get("/api/session/events?session_id=chosen")
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        assert [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")] == [
            {"status": "OPEN", "active_execution_id": execution_id, "revision": revision}
            for revision, execution_id in enumerate((None, "running", None))
        ]
        rejected = await http.get("/api/session/events?session_id=other")
        assert rejected.text.splitlines()[:2] == [
            "event: session_error", 'data: {"error_code": "AUTHORIZATION_DENIED"}',
        ]


@pytest.mark.asyncio
async def test_session_release_does_not_wait_for_stalled_chat_observer() -> None:
    from types import SimpleNamespace

    from linktools.ai.core import Principal, SessionStatus

    watching, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def session_watch(_session_id, *, principal):
        yield SimpleNamespace(status=SessionStatus.OPEN, active_execution_id="first", revision=1)
        await release.wait()
        yield SimpleNamespace(status=SessionStatus.OPEN, active_execution_id="next", revision=2)

    async def execution_watch(*, cursor, **_kwargs):
        assert cursor is None, "an old execution cursor cannot be passed to the new owner"
        try:
            watching.set()
            await asyncio.Future()
            yield
        finally:
            closed.set()

    async def execution_get(execution_id, *, principal):
        assert execution_id == "first" and principal.principal_id == "runtime"
        return SimpleNamespace(watch=execution_watch)

    runtime = SimpleNamespace(
        history=None, default_principal=Principal("runtime", "default"),
        sessions=SimpleNamespace(supports_turn_sessions=False, watch=session_watch, capture_published_turn=_no_published_turn), executions=SimpleNamespace(get=execution_get),
        events=SimpleNamespace(has_local_producer=lambda execution_id: execution_id == "first"),
    )
    async with client(create_app(runtime=runtime)) as http:
        request = asyncio.create_task(http.get(
            "/api/session/events?session_id=chosen&view=chat",
            headers={"Last-Event-ID": json.dumps(["earlier", "old-cursor"])},
        ))
        try:
            await asyncio.wait_for(watching.wait(), 5)
            release.set()
            response = await asyncio.wait_for(request, 5)
            states = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            assert [state["active_execution_id"] for state in states] == ["first", "next"]
            assert closed.is_set()
        finally:
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_session_disconnect_joins_observer_that_finishes_during_cancellation() -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Principal, SessionStatus
    from linktools.ai.history import ExecutionInfo

    watching, disconnected = asyncio.Event(), asyncio.Event()
    observer = None

    async def session_watch(_session_id, *, principal):
        yield SimpleNamespace(status=SessionStatus.OPEN, active_execution_id="run", revision=1)
        await asyncio.Future()

    async def execution_watch(**_kwargs):
        nonlocal observer
        observer = asyncio.current_task()
        watching.set()
        # An underlying store may finish its operation when cancellation arrives.
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            return
        yield

    async def inspect_execution(execution_id, *, principal):
        now = datetime.now(timezone.utc)
        return ExecutionInfo(
            execution_id, "agent", "default", None, ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.RUN, None, execution_id, None, "chosen", now, now, None,
        )

    async def execution_get(_execution_id, *, principal):
        return SimpleNamespace(watch=execution_watch)

    runtime = SimpleNamespace(
        history=SimpleNamespace(inspect_execution=inspect_execution),
        default_principal=Principal("runtime", "default"),
        sessions=SimpleNamespace(supports_turn_sessions=False, watch=session_watch, capture_published_turn=_no_published_turn), executions=SimpleNamespace(get=execution_get),
        events=SimpleNamespace(has_local_producer=lambda _: True),
    )
    app = create_app(runtime=runtime, shutdown_event=asyncio.Event())
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
             "scheme": "http", "path": "/api/session/events", "raw_path": b"/api/session/events",
             "query_string": b"session_id=chosen&view=chat", "headers": [],
             "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(_message):
        pass

    stream = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(watching.wait(), 5)
        disconnected.set()
        done, _ = await asyncio.wait((stream,), timeout=5)
        assert stream in done, "a disconnect must not wait for an unread observer frame"
        stream.result()
        assert observer.done()
    finally:
        disconnected.set()
        if observer is not None and not observer.done():
            observer.cancel()
        await asyncio.gather(stream, return_exceptions=True)


@pytest.mark.asyncio
async def test_two_session_streams_follow_same_turn_and_release(monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo

    from linktools.ai.storage import InMemoryCacheBackend
    from . import _runtime_test_helpers as helpers

    release = asyncio.Event()
    original = helpers._runtime_usage_model

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        await release.wait()
        return await original(messages, info)

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    features = RuntimeFeatures("web-session-events")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    storage = PersistenceStorage.in_memory(coordination=InMemoryCacheBackend(max_bytes=0))
    async with Persistence.open("web-session-events", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=model_features(RuntimeUsageModels(), features),
    ) as runtime:
        app = create_app(runtime=runtime)
        path = "/api/session/events"
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                 "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"session_id=shared&view=chat",
                 "headers": [], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}

        async def connect(frames: asyncio.Queue, disconnect: asyncio.Event) -> None:
            async def receive() -> dict[str, object]:
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                body = message.get("body")
                if isinstance(body, bytes) and body.startswith(b"data: "):
                    frames.put_nowait(json.loads(body[6:]))

            await app(dict(scope), receive, send)

        async with client(app) as http:
            created = await http.post("/api/sessions", json={"session_id": "shared", "request_id": "create"}, headers=_HEADERS)
            assert created.status_code == 201
            streams = (asyncio.Queue(), asyncio.Queue())
            disconnect = asyncio.Event()
            async with asyncio.TaskGroup() as tasks:
                for frames in streams:
                    tasks.create_task(connect(frames, disconnect))
                try:
                    initial = await asyncio.gather(*(asyncio.wait_for(frames.get(), 5) for frames in streams))
                    assert initial[0] == initial[1]
                    assert initial[0]["status"] == "OPEN" and initial[0]["active_execution_id"] is None
                    accepted = await http.post("/api/session/messages?session_id=shared", headers=_HEADERS,
                                               json={"prompt": "first", "request_id": "first"})
                    assert accepted.status_code == 202
                    identity = accepted.json()["execution_id"]
                    active = await asyncio.gather(*(asyncio.wait_for(frames.get(), 5) for frames in streams))
                    assert active[0] == active[1]
                    assert active[0]["status"] == "OPEN" and active[0]["active_execution_id"] == identity
                    assert active[0]["local_execution"] is True
                    replay = await http.post("/api/session/messages?session_id=shared", headers=_HEADERS,
                                             json={"prompt": "first", "request_id": "first"})
                    assert replay.status_code == 202 and replay.json()["execution_id"] == identity
                    collision = await http.post("/api/session/messages?session_id=shared", headers=_HEADERS,
                                                 json={"prompt": "second", "request_id": "second"})
                    assert collision.status_code == 409 and collision.json()["code"] == "SESSION_BUSY"
                    release.set()
                    idle = await asyncio.gather(*(asyncio.wait_for(frames.get(), 5) for frames in streams))
                    assert idle[0] == idle[1]
                    assert idle[0]["status"] == "OPEN" and idle[0]["active_execution_id"] is None
                    assert idle[0]["revision"] > active[0]["revision"]
                finally:
                    release.set()
                    disconnect.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(("code_name", "stage", "reason"), (
    ("STORAGE_RECOVERY_REQUIRED", "tool_receipts", "pending_tool_receipt_name_mismatch"),
    ("STORAGE_INTEGRITY_ERROR", "approval", "settled_call_frontier_mismatch"),
))
async def test_session_recover_logs_safe_blocking_reason_without_debug(
    monkeypatch: pytest.MonkeyPatch, code_name: str, stage: str, reason: str,
) -> None:
    from types import SimpleNamespace

    from linktools.ai.core import Principal
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.web import _app

    logged: list[str] = []
    monkeypatch.setattr(_app, "environ", SimpleNamespace(debug=False))
    monkeypatch.setattr(_app._logger, "warning", lambda message, *args: logged.append(message % args))
    details = {"recovery_stage": stage, "reason": reason}

    async def recover_turn(session_id, turn_id, *, principal):
        assert (session_id, turn_id) == ("private-session-id", "private-turn-id")
        raise AIError(
            ErrorCode[code_name],
            "private-exception-detail",
            safe_details=details,
            diagnostics=ErrorDiagnostics.from_exception(ValueError("private-cause")),
        )

    runtime = SimpleNamespace(
        history=None, default_principal=Principal("runtime", "default"),
        sessions=SimpleNamespace(
            _uses_turn_sessions=True, cancel_turn=None, recover_turn=recover_turn, turn_attachment_limits=None,
        ),
    )
    async with client(create_app(runtime=runtime)) as http:
        response = await http.post(
            "/api/session/recover?session_id=private-session-id",
            headers=_HEADERS, json={"request_id": "recover-request", "turn_id": "private-turn-id"},
        )

    assert response.status_code == 503
    payload = response.json()
    assert payload["code"] == code_name
    assert payload["operation_id"] == "recover-request"
    assert payload["retryable"] is False
    assert payload["safe_details"] == details
    assert len(logged) == 1
    for value in ("recover-request", code_name, *details.values()):
        assert value in logged[0]
    assert "private-" not in logged[0]
    assert "private-" not in response.text


@pytest.mark.asyncio
async def test_failed_history_debug_reports_stage_without_payload_or_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.web import _app

    environment = SimpleNamespace(debug=False)
    monkeypatch.setattr(_app, "environ", environment)
    logged: list[str] = []
    monkeypatch.setattr(_app._logger, "debug", lambda message, *args: logged.append(message % args))
    details = {"wire_type": "execution", "missing_fields": ["budget_scope_id"]}

    class History:
        has_turn_sessions = False
        tenant_id = "default"

        async def inspect_session(self, identity: str, *, principal: object) -> dict[str, object]:
            return {"session_id": identity, "status": "OPEN"}

        async def session_detail(self, identity: str, **kwargs: object) -> None:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR, "private exception prompt",
                operation_id="safe-operation", safe_details=details,
                diagnostics=ErrorDiagnostics.from_exception(ValueError("api_key=private-credential")),
            )

    async with client(create_app(history=History())) as http:
        quiet_failure = await http.get("/api/session?session_id=private-query&limit=50")
        assert quiet_failure.status_code == 503 and not logged
        environment.debug = True
        metadata = await http.get("/api/session?session_id=private-query&include_timeline=false")
        assert metadata.status_code == 200 and not logged
        failure = await http.get("/api/session?session_id=private-query&limit=50")
        assert failure.status_code == 503
        assert failure.json()["safe_details"] == details
        assert failure.json()["operation_id"] == "safe-operation"
        assert len(logged) == 1
        assert "path=/api/session phase=session.timeline code=STORAGE_INTEGRITY_ERROR" in logged[0]
        assert "operation_id=safe-operation" in logged[0]
        assert "wire_type=execution missing_fields=['budget_scope_id']" in logged[0]
        assert "private-" not in logged[0] and "private-" not in failure.text


@pytest.mark.asyncio
async def test_local_boundary_blocks_cross_site_fetches_and_unmarked_mutations() -> None:
    app = create_app()
    async with client(app) as http:
        response = await http.get("/")
        assert response.status_code == 200
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["cache-control"] == "private, no-store"
        assert (await http.get("/api/config", headers={"Sec-Fetch-Site": "cross-site"})).status_code == 403
        assert (await http.post("/api/sessions", json={})).status_code == 403
        assert (await http.options("/api/sessions", headers={"Origin": "https://evil.example", "Access-Control-Request-Headers": "x-linktools-console"})).status_code == 403
        assert (await http.post("/api/sessions", json={}, headers=_HEADERS)).status_code == 503
        assert (await http.get("/assets/missing.js")).status_code == 404
        for asset in ("message.js", "markdown-it.js"):
            resource = await http.get(f"/assets/{asset}")
            assert resource.status_code == 200 and "javascript" in resource.headers["content-type"]
            assert resource.headers["cache-control"] == "private, no-store"
        config = await http.get("/api/config", headers={"Host": "localhost:8765", "Origin": "http://localhost:8765"})
        assert config.status_code == 200
        assert config.headers["cache-control"] == "private, no-store"


@pytest.mark.asyncio
async def test_session_creation_returns_public_view_and_replays_without_duplicates() -> None:
    features = model_features(RuntimeUsageModels(), RuntimeFeatures("web-session-create"))
    async with Persistence.open("web-session-create", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.open(persistence=persistence, features=features) as runtime:
            async with client(create_app(runtime=runtime)) as http:
                request = {"session_id": "created", "request_id": "create", "title": "New session"}
                created = await http.post("/api/sessions", json=request, headers=_HEADERS)
                assert created.status_code == 201, created.text
                view = created.json()
                assert view == {
                    "session_id": "created", "agent_id": "default", "status": "OPEN",
                    "revision": 0, "cwd": None, "active_execution_id": None,
                    "metadata": {"title": "New session"}, "history_quality": "complete",
                }
                repeated = await http.post("/api/sessions", json=request, headers=_HEADERS)
                assert repeated.status_code == 201, repeated.text
                assert repeated.json() == view
                listed = await http.get("/api/sessions")
                assert listed.status_code == 200, listed.text
                assert listed.json()["items"] == [view]
                inspected = await http.get("/api/session?session_id=created&include_timeline=false")
                assert inspected.status_code == 200, inspected.text
                assert inspected.json()["session"] == view


@pytest.mark.asyncio
async def test_session_creation_does_not_require_session_read_authority() -> None:
    from linktools.ai.core import AuthorizationAction, Principal, ResourceRef, TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.session import CreateSessionRequest
    from linktools.ai.runtime import Session

    class CreateOnlyAuthorization(TenantAuthorizationPolicy):
        async def authorize(self, principal: Principal, action: AuthorizationAction, resource: ResourceRef) -> None:
            if action is not AuthorizationAction.SESSION_CREATE:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await super().authorize(principal, action, resource)

    features = model_features(RuntimeUsageModels(), RuntimeFeatures("web-create-only"))
    async with Runtime.open(
        "web-create-only", storage=PersistenceStorage.in_memory(), features=features,
        authorization=CreateOnlyAuthorization(),
    ) as runtime:
        async with client(create_app(runtime=runtime)) as http:
            request = {"session_id": "created", "request_id": "create", "title": "Create only"}
            created = await http.post("/api/sessions", json=request, headers=_HEADERS)
            assert created.status_code == 201, created.text
            assert created.json() == {
                "session_id": "created", "agent_id": "default", "status": "OPEN",
                "revision": 0, "cwd": None, "active_execution_id": None,
                "metadata": {"title": "Create only"}, "history_quality": "complete",
            }
            repeated = await http.post("/api/sessions", json=request, headers=_HEADERS)
            assert repeated.status_code == 201, repeated.text
            assert repeated.json() == created.json()
            detail = await http.get("/api/session?session_id=created&include_timeline=false")
            assert detail.status_code == 404, detail.text
        session = await runtime.sessions.create(
            "default", CreateSessionRequest(runtime.default_principal, "handle", "create-handle"),
        )
        assert isinstance(session, Session)
        with pytest.raises(AIError) as denied:
            await session.inspect()
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_session_execution_history_trace_result_and_metrics_use_runtime() -> None:
    metrics = Metrics.in_memory()
    features = RuntimeFeatures("web-test")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('web-test', storage=PersistenceStorage.in_memory()) as persistence_104, Runtime.open(persistence=persistence_104, metrics=metrics, features=model_features(RuntimeUsageModels(), features)) as runtime:
        async with client(create_app(runtime=runtime, metrics=metrics)) as http:
            request = {"session_id": "web-session", "request_id": "new-session", "title": "Review <script>"}
            first = await http.post("/api/sessions", json=request, headers=_HEADERS)
            repeated = await http.post("/api/sessions", json=request, headers=_HEADERS)
            assert first.status_code == repeated.status_code == 201
            assert first.json()["session_id"] == repeated.json()["session_id"]
            request = {"prompt": "hello", "request_id": "first-turn", "planning": False, "thinking": False}
            response = await http.post("/api/session/messages?session_id=web-session", json=request, headers=_HEADERS)
            assert response.status_code == 202, response.text
            identity = response.json()["execution_id"]
            result = (await (await runtime.executions.get(identity)).wait()).result
            assert result.output == {"text": "done"}
            repeat = await http.post("/api/session/messages?session_id=web-session", json=request, headers=_HEADERS)
            assert repeat.json()["execution_id"] == identity
            executions = (await http.get("/api/executions?session_id=web-session&limit=1")).json()
            assert [item["execution_id"] for item in executions["items"]] == [identity]
            detail = (await http.get(f"/api/executions/{identity}")).json()
            assert detail["status"] == "SUCCEEDED"
            assert detail["started_at"] and detail["terminal_at"]
            metadata = (await http.get(f"/api/executions/{identity}/history")).json()
            assert all(item["content"] is None for item in metadata["items"])
            all_items = []
            cursor = None
            while True:
                response = await http.get(f"/api/executions/{identity}/history", params={"include_content": "true", "limit": 1, **({"cursor": cursor} if cursor else {})})
                assert response.status_code == 200, response.text
                page = response.json()
                all_items.extend(page["items"])
                cursor = page["next_cursor"]
                if cursor is None:
                    break
            assert any(item["content"] == "hello" for item in all_items)
            assert any(item["content"] == "done" for item in all_items)
            trace = await http.get(f"/api/executions/{identity}/trace?agent_run_seq=1")
            assert trace.status_code == 200 and trace.json()["items"]
            models = (await http.get(f"/api/executions/{identity}/models?include_content=true")).json()
            assert models["items"][0]["model_request_seq"] == 1
            assert models["items"][0]["content_included"] is True
            assert (await http.get(f"/api/executions/{identity}/result")).json()["output"] == {"text": "done"}
            session = (await http.get("/api/session?session_id=web-session")).json()
            assert session["timeline"]["items"][0]["conversation_committed"]
            assert session["session"]["metadata"]["title"] == "Review <script>"
            await runtime.metrics.flush()
            summary = (await http.get("/api/metrics")).json()
            assert len(summary["items"]) == 12
            named = await http.get("/api/metrics?metric=linktools.model.input_tokens")
            assert named.json()["items"][0]["points"][0]["value"] == 101
            invalid = await http.get("/api/metrics?start=invalid")
            assert invalid.status_code == 400
            update = await http.post("/api/session/update?session_id=web-session", json={"request_id": "rename", "expected_revision": session["session"]["revision"], "metadata": {"title": "Updated"}}, headers=_HEADERS)
            assert update.status_code == 200, update.text
            stale = await http.post("/api/session/update?session_id=web-session", json={"request_id": "stale", "expected_revision": session["session"]["revision"], "metadata": {}}, headers=_HEADERS)
            assert stale.status_code == 409
            fork = await http.post("/api/session/fork?session_id=web-session", json={"request_id": "fork", "new_session_id": "forked"}, headers=_HEADERS)
            assert fork.status_code == 201, fork.text
            closed = await http.post("/api/session/close?session_id=web-session", json={"request_id": "close"}, headers=_HEADERS)
            assert closed.status_code == 200 and closed.json()["status"] == "CLOSED"


@pytest.mark.asyncio
async def test_reopened_console_preserves_history_and_hides_unavailable_resources(tmp_path: Path) -> None:
    from linktools.ai.core import AuthorizationAction, Principal, ResourceKind, TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode

    class Policy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)
            if (action is AuthorizationAction.SESSION_READ and resource.kind is ResourceKind.SESSION
                    and resource.id == "private" and principal.principal_id != "other-owner"):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    storage = PersistenceStorage.from_root(tmp_path)

    def features() -> RuntimeFeatures:
        result = model_features(RuntimeUsageModels(), RuntimeFeatures("web-reopen"))
        result.capability(ExecutionCapability(retain_history=True), scope="runtime")
        result.capability(StepPersistenceCapability(), scope="runtime")
        return result

    async with Persistence.open("web-reopen", storage=storage) as persistence:
        async with Runtime.open(persistence=persistence, features=features(), authorization=Policy()) as runtime:
            principal = runtime.default_principal
            other = Principal("other-owner", runtime.tenant_id, "service")
            await runtime.sessions.create_turn_session("visible", principal=principal)
            await runtime.sessions.create_turn_session("private", principal=other)
            terminal = asyncio.Event()

            def observe(frame):
                if frame["type"] == "terminal":
                    terminal.set()

            unsubscribe = runtime.sessions.observe_turn_session("visible", observe)
            try:
                visible_turn = await runtime.sessions.start_turn(
                    "visible", "visible message", agent="default", principal=principal, idempotency_key="visible",
                )
                await asyncio.wait_for(terminal.wait(), 5)
            finally:
                unsubscribe()
            visible_execution = await runtime.agents.get().start("independent execution")
            await visible_execution.wait()

    async with Persistence.open("web-reopen", storage=storage, tenant_id="other-tenant") as persistence:
        async with Runtime.open(persistence=persistence, features=features(), authorization=Policy()) as runtime:
            foreign_execution = await runtime.agents.get().start("other tenant message")
            await foreign_execution.wait()

    async with Persistence.open("web-reopen", storage=storage) as persistence:
        async with Runtime.open(persistence=persistence, features=features(), authorization=Policy()) as runtime:
            async with client(create_app(runtime=runtime, local_workspace=True)) as http:
                restored = await http.get("/api/session?session_id=visible")
                assert restored.status_code == 200, restored.text
                saved, = restored.json()["timeline"]["items"]
                assert saved["turn_id"] == visible_turn and saved["conversation_committed"]
                assert (await http.get(f"/api/executions/{visible_execution.execution_id}")).status_code == 200
                paths = [f"/api/session?session_id={identity}" for identity in ("missing", "private")]
                paths.extend(f"/api/executions/{identity}{suffix}"
                             for identity in ("missing", foreign_execution.execution_id) for suffix in ("", "/events"))
                for path in paths:
                    response = await http.get(path)
                    assert response.status_code == 404, response.text
                    assert response.json()["code"] == "AUTHORIZATION_DENIED"
                private = await runtime.history.turn_session_state("private", principal=other)
                assert private["session_id"] == "private"
                created = await http.post("/api/sessions", json={"session_id": "new", "request_id": "new"}, headers=_HEADERS)
                assert created.status_code == 201, created.text
                assert (await http.get("/api/session?session_id=new")).status_code == 200


@pytest.mark.asyncio
async def test_rejected_message_needs_new_identity_after_session_owner_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo

    from . import _runtime_test_helpers as helpers

    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0
    original = helpers._runtime_usage_model

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return await original(messages, info)

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    features = RuntimeFeatures("web-busy")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('web-busy', storage=PersistenceStorage.in_memory()) as persistence_190, Runtime.open(persistence=persistence_190, features=model_features(RuntimeUsageModels(), features)) as runtime:
        async with client(create_app(runtime=runtime)) as http:
            created = await http.post("/api/sessions", json={"session_id": "session", "request_id": "create"}, headers=_HEADERS)
            assert created.status_code == 201
            path = "/api/session/messages?session_id=session"
            first = await http.post(path, json={"prompt": "first", "request_id": "first"}, headers=_HEADERS)
            assert first.status_code == 202
            try:
                await asyncio.wait_for(entered.wait(), 10)
                request = {"prompt": "next", "request_id": "rejected"}
                busy = await http.post(path, json=request, headers=_HEADERS)
                assert busy.status_code == 409 and busy.json()["code"] == "SESSION_BUSY"
                current = await http.get("/api/session?session_id=session")
                assert current.status_code == 200
                assert current.json()["session"]["active_execution_id"] == first.json()["execution_id"]
            finally:
                release.set()
            await (await runtime.executions.get(first.json()["execution_id"])).wait()
            completed = await http.get("/api/session?session_id=session")
            assert completed.status_code == 200 and completed.json()["session"]["active_execution_id"] is None
            replay = await http.post(path, json=request, headers=_HEADERS)
            assert replay.status_code == 409 and replay.json()["code"] == "SESSION_BUSY"
            accepted = await http.post(path, json={**request, "request_id": "next-attempt"}, headers=_HEADERS)
            assert accepted.status_code == 202
            await (await runtime.executions.get(accepted.json()["execution_id"])).wait()
            assert calls == 2


@pytest.mark.asyncio
async def test_independent_clients_serialize_session_admission_and_cancel_exact_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo

    from . import _runtime_test_helpers as helpers

    entered = (asyncio.Event(), asyncio.Event())
    release = (asyncio.Event(), asyncio.Event())
    calls = 0
    original = helpers._runtime_usage_model

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        await release[index].wait()
        return await original(messages, info)

    monkeypatch.setattr(helpers, "_runtime_usage_model", model)
    features = RuntimeFeatures("web-clients")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('web-clients', storage=PersistenceStorage.in_memory()) as persistence_243, Runtime.open(persistence=persistence_243, features=model_features(RuntimeUsageModels(), features)) as runtime:
        app = create_app(runtime=runtime)
        async with client(app) as first, client(app) as second:
            clients = (first, second)
            created = await first.post("/api/sessions", json={"session_id": "shared", "request_id": "create"}, headers=_HEADERS)
            assert created.status_code == 201
            path = "/api/session/messages?session_id=shared"
            requests = ({"prompt": "first page", "request_id": "first-page"}, {"prompt": "second page", "request_id": "second-page"})
            try:
                responses = await asyncio.gather(*(http.post(path, json=request, headers=_HEADERS) for http, request in zip(clients, requests)))
                assert sorted(response.status_code for response in responses) == [202, 409]
                winner = next(index for index, response in enumerate(responses) if response.status_code == 202)
                loser = 1 - winner
                assert responses[loser].json()["code"] in {"SESSION_BUSY", "SESSION_CONFLICT"}
                owner_id = responses[winner].json()["execution_id"]
                await asyncio.wait_for(entered[0].wait(), 10)
                for http in clients:
                    session = await http.get("/api/session?session_id=shared&include_timeline=false")
                    assert session.json()["session"]["active_execution_id"] == owner_id
                assert calls == 1
                release[0].set()
                await (await runtime.executions.get(owner_id)).wait()
                rejected = await clients[loser].post(path, json=requests[loser], headers=_HEADERS)
                assert rejected.status_code == 409 and rejected.json()["code"] == responses[loser].json()["code"]
                admitted = await clients[loser].post(path, json={**requests[loser], "request_id": "explicit-next-attempt"}, headers=_HEADERS)
                assert admitted.status_code == 202
                next_id = admitted.json()["execution_id"]
                assert next_id != owner_id
                await asyncio.wait_for(entered[1].wait(), 10)
                stale_stop = await clients[winner].post(f"/api/executions/{owner_id}/cancel", json={"request_id": "old-page-stop"}, headers=_HEADERS)
                assert stale_stop.status_code == 200
                for http in clients:
                    session = await http.get("/api/session?session_id=shared&include_timeline=false")
                    assert session.json()["session"]["active_execution_id"] == next_id
                stopped = await clients[loser].post(f"/api/executions/{next_id}/cancel", json={"request_id": "current-page-stop"}, headers=_HEADERS)
                assert stopped.status_code == 200
                result = (await (await runtime.executions.get(next_id)).wait()).result
                assert result.status.value == "CANCELLED"
                for http in clients:
                    session = await http.get("/api/session?session_id=shared&include_timeline=false")
                    assert session.json()["session"]["active_execution_id"] is None
                assert calls == 2
            finally:
                for gate in release:
                    gate.set()


@pytest.mark.asyncio
async def test_execution_stream_replays_cursor_and_finishes_with_authoritative_snapshot() -> None:
    async with Persistence.open('web-stream', storage=PersistenceStorage.in_memory()) as persistence_292, Runtime.open(persistence=persistence_292, features=model_features(RuntimeUsageModels(), _persistence_capabilities(trace=True))) as runtime:
        execution = await runtime.agents.get().start("hello")
        await execution.wait()
        async with client(create_app(runtime=runtime)) as http:
            response = await http.get(f"/api/executions/{execution.execution_id}/events")
            assert response.status_code == 200
            assert {"private", "no-store", "no-transform"} <= {
                directive.strip() for directive in response.headers["cache-control"].split(",")
            }
            assert response.headers["x-accel-buffering"] == "no"
            frames = [frame for frame in response.text.split("\n\n") if frame]
            assert any(frame.startswith("event: snapshot") for frame in frames)
            cursor = [frame.split("\n", 1)[0][4:] for frame in frames if frame.startswith("id: ")][-1]
            resumed = await http.get(f"/api/executions/{execution.execution_id}/events", headers={"Last-Event-ID": cursor})
            assert resumed.status_code == 200
            for frame in resumed.text.split("\n\n"):
                if frame.startswith("id: "):
                    envelope = json.loads(frame.split("data: ", 1)[1])
                    if envelope["type"] == "event":
                        assert envelope["item"]["event"]["durable_seq"] is None
            invalid = await http.get(f"/api/executions/{execution.execution_id}/events?cursor=bad-cursor")
            assert invalid.status_code == 400


@pytest.mark.asyncio
async def test_read_only_history_requires_no_model_and_never_opens_runtime(tmp_path: Path) -> None:
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.errors import AIError, ErrorCode

    class WorkspacePolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)

    storage = PersistenceStorage.from_root(tmp_path)
    features = RuntimeFeatures("web-history")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with (
        Persistence.open("default", storage=storage) as persistence,
        Runtime.open(persistence=persistence, features=model_features(RuntimeUsageModels(), features),
                     authorization=WorkspacePolicy()) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        terminal = asyncio.Event()

        def observe(frame):
            if frame["type"] == "terminal":
                terminal.set()

        unsubscribe = runtime.sessions.observe_turn_session("session", observe)
        try:
            turn_id = await runtime.sessions.start_turn(
                "session", "saved", agent="default", principal=principal, idempotency_key="send",
            )
            await asyncio.wait_for(terminal.wait(), 5)
        finally:
            unsubscribe()
        execution = await runtime.agents.get().start("independent execution")
        await execution.wait()
        identity = execution.execution_id

    async with (
        Persistence.open("default", storage=storage, read_only=True) as persistence,
        History.open(persistence=persistence, authorization=WorkspacePolicy()) as history,
        client(create_app(history=history, local_workspace=True)) as http,
    ):
        with pytest.raises(AIError) as legacy_read:
            await history.lookup_session("session", principal=principal)
        assert legacy_read.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert (await http.get("/api/config")).json()["read_only"]
        sessions = (await http.get("/api/sessions")).json()
        assert [item["session_id"] for item in sessions["items"]] == ["session"]
        saved = await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})
        assert saved.status_code == 200, saved.text
        assert saved.json()["conversation_committed"]
        assert any(item["item_kind"] == "assistant" and item["content"] == "done"
                   for item in saved.json()["items"])
        assert (await http.get("/api/session/events?session_id=session")).status_code == 503
        assert (await http.get(f"/api/executions/{identity}/result")).json()["output"] == {"text": "done"}
        assert (await http.get(f"/api/executions/{identity}/events")).status_code == 503
        assert (await http.post(f"/api/executions/{identity}/recover",
                                json={"request_id": "recover"}, headers=_HEADERS)).status_code == 503
        assert (await http.get("/api/executions?limit=0")).status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("server_shutdown", (False, True))
async def test_leaving_stream_closes_observer_without_cancelling_execution(server_shutdown: bool) -> None:
    closed = asyncio.Event()
    cancel_called = False

    class Execution:
        execution_id = "execution"

        async def watch(self, **kwargs: object):
            try:
                yield type("Observation", (), {"event": {"pending": True}, "cursor": None})()
                await asyncio.Event().wait()
            finally:
                closed.set()

        async def cancel(self) -> None:
            nonlocal cancel_called
            cancel_called = True

    class Executions:
        async def get(self, identity: str, **kwargs: object) -> Execution:
            return Execution()

    from types import SimpleNamespace

    class RuntimeStub:
        sessions = SimpleNamespace(supports_turn_sessions=False)
        default_principal = object()
        history = None
        executions = Executions()

    shutdown = asyncio.Event()
    app = create_app(runtime=RuntimeStub(), shutdown_event=shutdown)
    sent_body = asyncio.Event()
    initial = False
    body = []

    async def receive() -> dict[str, object]:
        nonlocal initial
        if not initial:
            initial = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await sent_body.wait()
        if server_shutdown:
            await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body":
            body.append(message.get("body", b""))
            if message.get("body"):
                sent_body.set()
                if server_shutdown:
                    shutdown.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
             "scheme": "http", "path": "/api/executions/execution/events", "raw_path": b"/api/executions/execution/events",
             "query_string": b"", "headers": [(b"host", b"127.0.0.1:8765")], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}
    await asyncio.wait_for(app(scope, receive, send), 2)
    assert closed.is_set()
    assert not cancel_called
    if server_shutdown:
        assert len(body) == 2
        assert body[-1] == b""


def test_web_command_preserves_runtime_flags_and_validates_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linktools.commands.ai.web import command
    from linktools.cli import CommandError
    parser = command.create_parser()
    assert {"project", "model", "base_url", "api_key", "port"} <= {action.dest for action in parser._actions}
    assert "memory" not in {action.dest for action in parser._actions}
    with pytest.raises(CommandError, match="port"):
        command.run(parser.parse_args(["--port", "0"]))


@pytest.mark.asyncio
async def test_remote_peer_cannot_claim_a_local_host_header() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(), client=("203.0.113.9", 1000)), base_url="http://127.0.0.1:8765") as http:
        response = await http.get("/api/config")
        assert response.status_code == 403
        assert response.json()["error_code"] == "LOOPBACK_REQUIRED"


@pytest.mark.asyncio
async def test_external_host_and_origin_preserve_peer_and_request_boundaries() -> None:
    headers = {"Host": "console.example", "Origin": "https://console.example", "Sec-Fetch-Site": "same-origin"}
    async with client(create_app()) as http:
        response = await http.get("/api/config", headers=headers)
        assert response.status_code == 200
        assert (await http.get("/api/config", headers={"Host": "another.example", "Origin": "null"})).status_code == 200
        assert "access-control-allow-origin" not in response.headers
        assert (await http.post("/api/sessions", headers=headers, json={})).status_code == 403
        assert (await http.post("/api/sessions", headers={**headers, **_HEADERS}, json={})).status_code == 503
        assert (await http.get("/api/config", headers={**headers, "Sec-Fetch-Site": "cross-site"})).status_code == 403
        assert (await http.options("/api/sessions", headers=headers)).status_code == 403
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(), client=("203.0.113.9", 1000)), base_url="http://127.0.0.1:8765") as http:
        response = await http.get("/api/config", headers={**headers, "X-Forwarded-For": "127.0.0.1"})
        assert response.status_code == 403
        assert response.json()["error_code"] == "LOOPBACK_REQUIRED"


@pytest.mark.asyncio
async def test_exception_diagnostics_preserve_original_provider_text() -> None:
    from dataclasses import dataclass

    @dataclass
    class FailedView:
        error_code: str
        error_diagnostics: ErrorDiagnostics

    class History:
        has_turn_sessions = False
        tenant_id = "default"

        async def inspect_execution(self, identity: str, **kwargs: object) -> FailedView:
            try:
                raise ValueError('api_key="secret-key-value" ' + "long-diagnostic-" * 300 + " END_DIAGNOSTIC")
            except ValueError as error:
                return FailedView("MODEL_UNAVAILABLE", ErrorDiagnostics.from_exception(error))

    async with client(create_app(history=History())) as http:
        response = await http.get("/api/executions/example")
        assert response.status_code == 200
        assert "secret-key-value" in response.text
        assert response.json()["error_diagnostics"]["exception_type"] == "ValueError"
        assert response.json()["error_diagnostics"]["cause_digest"]
        diagnostics = response.json()["error_diagnostics"]
        assert 'api_key="secret-key-value"' in diagnostics["exception_message"]
        assert diagnostics["exception_chain"][0]["exception_message"].endswith("END_DIAGNOSTIC")
        assert "END_DIAGNOSTIC" in diagnostics["traceback"]
        assert "in inspect_execution" in diagnostics["traceback"]


@pytest.mark.asyncio
async def test_failed_execution_stream_preserves_diagnostics_and_user_content() -> None:
    from ._runtime_test_helpers import _UsageFunctionModel
    from .test_live_history_readback_integration import _Models

    async def fail(messages, info):
        raise ValueError("api_key=provider-secret")

    prompt = '{"error_diagnostics":{"exception_message":"user content"}}'
    async with Persistence.open('web-failed-stream', storage=PersistenceStorage.in_memory()) as persistence_464, Runtime.open(persistence=persistence_464, features=model_features(_Models(_UsageFunctionModel(fail)), _persistence_capabilities(trace=True))) as runtime:
        execution = await runtime.agents.get().start(prompt)
        await execution.wait()
        async with client(create_app(runtime=runtime)) as http:
            response = await http.get(f"/api/executions/{execution.execution_id}/events")
            assert response.status_code == 200
            assert "provider-secret" in response.text
            events = [json.loads(frame.split("data: ", 1)[1]) for frame in response.text.split("\n\n") if "data: " in frame]
            failed = next(item["item"]["event"] for item in events if item.get("type") == "event" and item["item"]["event"]["event_type"] == "EXECUTION_FAILED")
            assert failed["payload"]["error_diagnostics"]["exception_type"] == "ValueError"
            assert failed["payload"]["error_diagnostics"]["cause_digest"]
            assert failed["payload"]["error_diagnostics"]["exception_message"] == "api_key=provider-secret"
            assert failed["payload"]["error_diagnostics"]["exception_chain"]
            assert "in fail" in failed["payload"]["error_diagnostics"]["traceback"]
            history = await http.get(f"/api/executions/{execution.execution_id}/history?include_content=true")
            assert any(item["content"] == prompt for item in history.json()["items"])


def test_historical_event_diagnostics_preserve_original_full_text() -> None:
    from linktools.ai.execution import ExecutionStreamEvent
    from linktools.ai.web._app import _wire

    text = repr(json.dumps({"api_key": 'prefix"TAIL_SECRET', "status": 401}))
    event = ExecutionStreamEvent("execution", 9, "EXECUTION_FAILED", {
        "error_diagnostics": {
            "exception_type": "ValueError", "exception_message": text, "cause_digest": "a" * 64,
            "exception_chain": [{"relation": "root", "exception_type": "ValueError", "exception_message": text}],
            "traceback": text + "\n" + "long-frame-" * 400 + "END_STACK",
        },
    })
    value = _wire(event)["payload"]["error_diagnostics"]
    assert value["exception_message"] == text
    assert value["exception_chain"][0]["exception_message"] == text
    assert "401" in value["exception_message"]
    assert value["traceback"].endswith("END_STACK")
    assert len(value["traceback"]) > 4000


def test_web_cli_owns_one_runtime_and_closes_it_after_server_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from contextlib import asynccontextmanager
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.cli import CommandError
    import linktools.commands.ai._common as common_command
    import linktools.commands.ai.web as web_command
    import uvicorn

    original_open = Runtime.open
    original_assets = common_command._local_runtime_assets
    lifecycle = []
    asset_stores = []

    def create_assets(workspace):
        store = original_assets(workspace)
        asset_stores.append(store)
        return store

    @asynccontextmanager
    async def tracked_open(*args: object, **kwargs: object):
        async with original_open(*args, **kwargs) as runtime:
            lifecycle.append("open")
            try:
                yield runtime
            finally:
                lifecycle.append("close")

    class Server:
        def __init__(self, config: object) -> None:
            self.config = config

        async def serve(self) -> None:
            assert self.config.host == "127.0.0.1"
            assert self.config.proxy_headers is False
            assert asset_stores and all(store.ready for store in asset_stores)
            async with client(self.config.app) as http:
                response = await http.get("/api/config")
                assert response.status_code == 200
                assert response.json()["read_only"] is False
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(Runtime, "open", tracked_open)
    monkeypatch.setattr(common_command, "_local_runtime_assets", create_assets)
    monkeypatch.setattr(common_command._ModelOptions, "create_model_binding", lambda _self: RuntimeUsageModels())
    monkeypatch.setattr(uvicorn, "Server", Server)
    args = web_command.command.create_parser().parse_args(["--project", str(tmp_path), "--model", "fake"])
    with pytest.raises(CommandError):
        web_command.command.run(args)
    assert lifecycle == ["open", "close"]
    assert all(not store.ready for store in asset_stores)


def test_web_cli_provides_default_memory_tools_to_messages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from contextlib import asynccontextmanager
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo
    from linktools.ai.runtime.turn_clients import TurnObserver
    import linktools.commands.ai.web as web_command
    import uvicorn

    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    memory_tools = {"read_memory", "write_memory", "search_memory", "delete_memory"}

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert memory_tools <= {tool.name for tool in info.function_tools}
        returned = next((part for part in messages[-1].parts if isinstance(part, ToolReturnPart)), None)
        if returned is not None:
            return ModelResponse(parts=[TextPart(str(returned.content))])
        prompt = next(part.content for part in messages[-1].parts if isinstance(part, UserPromptPart))
        call = (ToolCallPart("write_memory", {"file": "web-notes.md", "content": "remembered from web"})
                if prompt == "remember" else ToolCallPart("read_memory", {"file": "web-notes.md"}))
        return ModelResponse(parts=[call])

    original_open = Runtime.open
    runtimes: list[Runtime] = []

    @asynccontextmanager
    async def tracked_open(*args: object, **kwargs: object):
        async with original_open(*args, **kwargs) as runtime:
            runtimes.append(runtime)
            yield runtime

    class Server:
        def __init__(self, config: object) -> None:
            self.config = config

        async def serve(self) -> None:
            async with client(self.config.app) as http:
                settings = await http.get("/api/config")
                declarations = settings.json()["capabilities"]
                assert ("agent", "default") in {(item["kind"], item["id"]) for item in declarations}
                assert memory_tools <= {item["id"] for item in declarations if item["kind"] == "tool"}
                created = await http.post("/api/sessions", headers=_HEADERS,
                    json={"session_id": "memory-session", "request_id": "create-memory-session"})
                assert created.status_code == 201, created.text
                runtime = runtimes[0]
                for prompt in ("remember", "recall"):
                    with TurnObserver(
                        runtime, "memory-session", runtime.default_principal, local_workspace=True,
                    ) as live:
                        response = await http.post("/api/session/messages?session_id=memory-session", headers=_HEADERS,
                            json={"prompt": prompt, "request_id": prompt})
                        assert response.status_code == 202 and set(response.json()) == {"turn_id"}, response.text
                        turn_id = response.json()["turn_id"]
                        async with asyncio.timeout(10):
                            async for frame in live.events(turn_id):
                                if frame["type"] != "terminal":
                                    continue
                                assert frame["state"] == "succeeded", frame
                                result = frame["output"]
                                break
                            else:
                                pytest.fail("Memory tool turn did not publish a terminal result")

                    history = await http.get("/api/session", params={"session_id": "memory-session", "turn_id": turn_id})
                    assert history.status_code == 200, history.text
                    turn = history.json()
                    assert turn["turn_id"] == turn_id and turn["conversation_committed"]
                    tool, = turn["process"]
                    assert tool["argument_preview"] == "web-notes.md"
                    assert tool["status"] == "SUCCEEDED"
                    assert "remembered from web" not in json.dumps(tool)
                    if prompt == "recall":
                        assert result["text"].strip() == "remembered from web"
                        assert _turn_text(turn).strip() == "remembered from web"

    monkeypatch.setattr(Runtime, "open", tracked_open)
    monkeypatch.setattr(common_command._ModelOptions, "create_model_binding", lambda _self: _Models(_UsageFunctionModel(respond)))
    monkeypatch.setattr(uvicorn, "Server", Server)
    args = web_command.command.create_parser().parse_args(["--project", str(tmp_path), "--model", "fake"])
    assert web_command.command.run(args) == 0


def test_web_cli_requires_a_model_before_opening_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linktools.cli import CommandError
    import linktools.commands.ai.web as web_command
    import uvicorn

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unconfigured console must not open Runtime or start HTTP")

    monkeypatch.setattr(Runtime, "open", forbidden)
    monkeypatch.setattr(uvicorn, "Server", forbidden)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    env = web_command.command.environ
    monkeypatch.setattr(env, "config", env.build_config("web-test", "LINKTOOLS_"))
    args = web_command.command.create_parser().parse_args(["--project", str(tmp_path)])
    with pytest.raises(CommandError, match="model is required"):
        web_command.command.run(args)
    assert not (tmp_path / ".linktools" / "runtime").exists()


@pytest.mark.parametrize(
    ("argument", "environment", "configured"),
    [(None, "", False), (None, " \t", True), (None, "value", True), ("", "value", True), (" \t", "", False)],
)
def test_web_settings_show_runtime_mime_types_and_connection_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argument: str | None, environment: str, configured: bool,
) -> None:
    import linktools.commands.ai.web as web_command
    import uvicorn

    config_dir = tmp_path / ".linktools"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        "model: opaque-model\ninput_types:\n  - image\n  - mime:audio/*\n"
    )

    class Server:
        def __init__(self, config: object) -> None:
            self.config = config

        async def serve(self) -> None:
            async with client(self.config.app) as http:
                response = await http.get("/api/config")
                assert "vision" not in response.json()
                from typing import get_args
                from linktools.ai.model import ImageMediaType

                assert response.json()["input_mime_types"] == sorted({*get_args(ImageMediaType), "audio/*"})
                settings = response.json()
                assert "inputs" not in settings
                assert settings["read_only"] is False
                assert settings["runtime"]["namespace"] == "default"
                assert settings["workspace"] == str(tmp_path)
                assert settings["asset_root"] == str(config_dir)
                assert settings["model"] == "opaque-model"
                storage = {item["domain"]: item for item in settings["runtime"]["storage"]}
                assert storage["execution"]["backend"] == "sqlite"
                assert storage["execution"]["path"] == str(config_dir / "runtime" / "runtime.db")
                assert storage["execution"]["object_store_paths"] == [str(config_dir / "runtime" / "objects")]
                assert settings["runtime"]["metrics"]["enabled"] is True
                assert settings["base_url_configured"] is configured
                assert settings["api_key_configured"] is configured
                assert "credential-placeholder" not in response.text
                assert "provider.invalid" not in response.text

    base_url = " https://provider.invalid/v1 " if environment == "value" else environment
    api_key = " credential-placeholder " if environment == "value" else environment
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    env = web_command.command.environ
    monkeypatch.setattr(env, "config", env.build_config("web-test", "LINKTOOLS_"))
    monkeypatch.setattr(uvicorn, "Server", Server)
    arguments = ["--project", str(tmp_path)]
    if argument is not None:
        arguments.extend(("--base-url", argument, "--api-key", argument))
    args = web_command.command.create_parser().parse_args(arguments)
    assert web_command.command.run(args) == 0
    assert (config_dir / "runtime" / "runtime.db").exists()


def test_web_cli_rejects_invalid_input_types(tmp_path: Path) -> None:
    from linktools.commands.ai.web import command
    from linktools.cli import CommandError

    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_types: image/png\n")
    args = command.create_parser().parse_args(["--project", str(tmp_path), "--model", "fake"])
    with pytest.raises(CommandError, match="input_types"):
        command.run(args)


@pytest.mark.asyncio
async def test_opaque_session_ids_round_trip_through_query_identity() -> None:
    identity = "../team/conversation?notes#你好"
    features = RuntimeFeatures("opaque-web")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('opaque-web', storage=PersistenceStorage.in_memory()) as persistence_641, Runtime.open(persistence=persistence_641, features=model_features(RuntimeUsageModels(), features)) as runtime:
        async with client(create_app(runtime=runtime)) as http:
            created = await http.post("/api/sessions", json={"session_id": identity, "request_id": "create-opaque", "title": "Opaque session"}, headers=_HEADERS)
            assert created.status_code == 201, created.text
            detail = await http.get("/api/session", params={"session_id": identity})
            assert detail.status_code == 200
            assert detail.json()["session"]["session_id"] == identity
            started = await http.post("/api/session/messages", params={"session_id": identity}, json={"prompt": "hello", "request_id": "opaque-turn"}, headers=_HEADERS)
            assert started.status_code == 202, started.text
            await (await runtime.executions.get(started.json()["execution_id"])).wait()
            closed = await http.post("/api/session/close", params={"session_id": identity}, json={"request_id": "opaque-close"}, headers=_HEADERS)
            assert closed.status_code == 200


@pytest.mark.asyncio
async def test_end_recovery_required_execution_uses_durable_cancel_and_replays_recovery(
    tmp_path: Path,
) -> None:
    from ._session_tool_test_helpers import _ToolModels, _application

    calls: list[str] = []
    started, release = asyncio.Event(), asyncio.Event()
    application = _application(
        calls, effect_policy="non_replay_safe", effect_log=tmp_path / "effects.txt",
        started=started, release=release,
    )
    try:
        application.capability(native_checkpoint(), scope="runtime")
        application.capability(TraceCapability(), scope="runtime")
        async with Persistence.open('web-end-recovery', storage=PersistenceStorage.in_memory()) as persistence_670, Runtime.open(persistence=persistence_670, features=model_features(_ToolModels(), application)) as runtime, client(create_app(runtime=runtime)) as http:
            session = await runtime.agents.get("default").create_session("session")
            execution = await session.start("inspect", idempotency_key="turn")
            await asyncio.wait_for(started.wait(), 10)
            path = f"/api/executions/{execution.execution_id}"
            stopped = await http.post(path + "/cancel", json={"request_id": "stop"}, headers=_HEADERS)
            assert stopped.status_code == 200 and stopped.json()["cancelled"] is False
            release.set()
            assert (await http.get(path)).json()["status"] == "RECOVERY_REQUIRED"
            before = await execution.model_interactions(include_content=True)
            for _ in range(2):
                accepted = await http.post(path + "/cancel", json={"request_id": "end-cancel"}, headers=_HEADERS)
                assert accepted.status_code == 200 and accepted.json()["cancelled"] is False
                assert (await http.get(path)).json()["status"] == "RECOVERY_REQUIRED"
            blocked = await http.post(path + "/recover", json={"request_id": "end-recover"}, headers=_HEADERS)
            assert blocked.status_code == 503 and blocked.json()["code"] == "TOOL_EFFECT_UNKNOWN", blocked.text
            assert (await http.get("/api/session?session_id=session&include_timeline=false")).json()["session"]["active_execution_id"] == execution.execution_id
            effects = (await http.get(path + "/recovery")).json()
            resolved = await http.post(path + "/resolve", json={
                "request_id": "resolve", "operation_id": effects[0]["operation_id"],
                "expected_fence": effects[0]["fence"], "resolution": "applied", "result": {"confirmed": True},
            }, headers=_HEADERS)
            assert resolved.status_code == 200
            for _ in range(2):
                ended = await http.post(path + "/recover", json={"request_id": "end-recover"}, headers=_HEADERS)
                assert ended.status_code == 202, ended.text
                assert (await http.get(path)).json()["status"] == "CANCELLED"
                owner = (await http.get("/api/session?session_id=session&include_timeline=false")).json()["session"]
                assert owner["active_execution_id"] is None
            assert calls == ["lookup"]
            assert (await execution.model_interactions(include_content=True)).items == before.items
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", ({}, {"memory_scope": {"application": "value"}}))
async def test_web_memory_uses_runtime_configuration(extra: dict[str, object]) -> None:
    from linktools.ai.capability import MemoryCapability

    features = model_features(RuntimeUsageModels())
    features.capability(MemoryCapability(scope="host-memory"))
    async with Runtime.open("web-memory", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        await runtime.agents.get().create_session("session")
        async with client(create_app(runtime=runtime)) as http:
            config = await http.get("/api/config")
            assert "memory_scope" not in config.json()
            payload = {"prompt": "hello", "request_id": "message", **extra}
            accepted = await http.post("/api/session/messages?session_id=session", json=payload, headers=_HEADERS)
            assert accepted.status_code == 202, accepted.text
            execution = await runtime.executions.get(accepted.json()["execution_id"])
            await execution.wait()
            record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert record.memory_scope == "host-memory"


@pytest.mark.asyncio
async def test_chat_transport_uses_public_tool_progress_and_excludes_native_content() -> None:
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, Page, Principal, SessionStatus
    from linktools.ai.execution import ExecutionStreamEvent, ExecutionTreeEvent
    from linktools.ai.session import SessionDetail, SessionTurn, SessionTurnItem, ToolProcessItem

    principal = Principal("runtime", "default")
    raw_marker = "NOT-FOR-CHAT"
    thinking = "  SDK thinking\n<literal text>  "
    process = ToolProcessItem("tool:id", "lookup", "SUCCEEDED", (2, 1), elapsed_ms=19)
    now = datetime.now(timezone.utc)

    class History:
        has_turn_sessions = False
        async def inspect_session(self, identity, **kwargs):
            return {"session_id": identity}

        async def session_detail(self, identity, **kwargs):
            return SessionDetail(await self.inspect_session(identity, **kwargs), Page((SessionTurn(
                "execution", ExecutionStatus.SUCCEEDED, now, now, "user", True,
                (SessionTurnItem(1, "thinking", thinking),
                 SessionTurnItem(2, "tool_call", raw_marker, "lookup", "id"),
                 SessionTurnItem(3, "tool_result", raw_marker),
                 SessionTurnItem(4, "assistant", "answer"),
                 SessionTurnItem(5, "tool_call", raw_marker, "lookup", "id")),
                process=(process,),
            ),)))

        async def inspect_execution(self, identity, **kwargs):
            from linktools.ai.history import ExecutionInfo
            return ExecutionInfo(
                identity, "agent", "default", None, ExecutionStatus.SUCCEEDED,
                ExecutionLineageKind.RUN, None, identity, None, "session", now, now,
                None, safe_error_details={"value": raw_marker},
            )

    class Execution:
        execution_id = "execution"

        async def watch(self, *, cursor, include_content, include_model_interactions, include_descendants):
            assert include_content and not include_model_interactions and not include_descendants
            for n, (kind, payload, depth) in enumerate((
                ("ASSISTANT_THINKING_DELTA", {"text": thinking, "signature": raw_marker}, 0),
                ("ASSISTANT_THINKING_DELTA", {"text": raw_marker}, 1),
                ("TOOL_CALL_FINISHED", {"call_id": "id", "tool_name": "lookup", "status": "SUCCEEDED",
                                        "result": raw_marker, "tool_position": [2, 1], "agent_run_seq": 1,
                                        "tool_model_request_seq": 2, "model_request_seq": 7, "tool_elapsed_ms": 19}, 0),
                ("ASSISTANT_TEXT_DELTA", {"text": "answer", "internal": raw_marker}, 0),
                ("ASSISTANT_TEXT_DELTA", {"text": raw_marker}, 1),
                ("MODEL_ATTEMPT_FINISHED", {"status": "SUCCEEDED", "provider_request": raw_marker}, 0),
            )):
                execution = "execution" if not depth else "child"
                yield ExecutionTreeEvent(
                    execution, "default",
                    ExecutionLineageKind.RUN if not depth else ExecutionLineageKind.SUBAGENT,
                    "execution" if depth else None, "execution", "invocation" if depth else None, depth,
                    ExecutionStreamEvent(execution, n + 1, kind, payload),
                    cursor=str(n + 1),
                )

    async def get_execution(identity, **kwargs):
        assert identity == "execution" and kwargs["principal"] == principal
        return Execution()

    closed = asyncio.Event()

    async def watch_session(identity, **kwargs):
        assert identity == "session" and kwargs["principal"] == principal
        yield SimpleNamespace(active_execution_id="execution", status=SessionStatus.OPEN, revision=0)
        await closed.wait()

    runtime = SimpleNamespace(history=History(), default_principal=principal,
                              sessions=SimpleNamespace(supports_turn_sessions=False, watch=watch_session, capture_published_turn=_no_published_turn),
                              events=SimpleNamespace(has_local_producer=lambda _identity: True),
                              executions=SimpleNamespace(get=get_execution))
    app = create_app(runtime=runtime)
    async with client(app) as http:
        history = await http.get("/api/session?session_id=session")
        assert history.status_code == 200, history.text
        assert raw_marker not in history.text
        turn = history.json()["timeline"]["items"][0]
        assert [(item["item_kind"], item["content"]) for item in turn["items"]] == [
            ("thinking", thinking), ("tool_call", None), ("assistant", "answer"),
        ]
        assert turn["items"][1]["tool_call_id"] == "id"
        assert turn["items"][1]["tool_name"] is None
        assert turn["process"] == [{"kind": "tool", "key": "tool:id", "tool_name": "lookup",
                                    "status": "SUCCEEDED", "position": [2, 1], "elapsed_ms": 19}]
        removed = await http.get("/api/executions/execution/events?view=chat")
        assert removed.status_code == 400
        assert removed.json()["error_code"] == "REQUEST_FIELD_INVALID"

    started = False

    async def receive():
        nonlocal started
        if not started:
            started = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await closed.wait()
        return {"type": "http.disconnect"}

    frames = []

    async def send(message):
        if message["type"] == "http.response.start":
            assert message["status"] == 200
        body = message.get("body", b"").decode()
        assert raw_marker not in body
        name = next((line[7:] for line in body.splitlines() if line.startswith("event: ")), "message")
        for line in body.splitlines():
            if line.startswith("data: "):
                value = json.loads(line[6:])
                frames.append((name, value))
                if name == "execution" and value["event"] == "snapshot":
                    closed.set()

    path = "/api/session/events"
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
             "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"session_id=session&view=chat",
             "headers": [], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}
    await asyncio.wait_for(app(scope, receive, send), 5)
    assert [name for name, _ in frames] == ["message", "thinking", "execution", "text", "execution", "execution"]
    assert frames[1][1] == {"execution_id": "execution", "text": thinking}
    assert frames[2][1]["data"]["item"] == turn["process"][0]
    assert frames[2][1]["data"]["order"] == [1, 2, 1]
    assert "elapsed_ms" not in frames[2][1]["data"]
    assert frames[3][1] == {"execution_id": "execution", "text": "answer"}
    assert frames[-1][1]["data"]["status"] == "SUCCEEDED"


@pytest.mark.asyncio
async def test_web_exact_session_turn_keeps_pending_and_recovery_distinct_without_latest_fallback() -> None:
    from linktools.ai.errors import AIError, ErrorCode

    class ExactHistory:
        has_turn_sessions = False
        tenant_id = "default"

        async def session_turn(self, identity, execution_id, **kwargs):
            assert (identity, execution_id) == ("session", "requested")
            raise AIError(self.failure)

    history = ExactHistory()
    async with client(create_app(history=history)) as http:
        for code, status in (
            (ErrorCode.SESSION_TURN_PENDING, 409),
            (ErrorCode.SESSION_HISTORY_UNAVAILABLE, 503),
            (ErrorCode.STORAGE_RECOVERY_REQUIRED, 503),
            (ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, 503),
        ):
            history.failure = code
            response = await http.get("/api/session?session_id=session&execution_id=requested")
            assert response.status_code == status, response.text
            assert response.json()["code"] == code.value


def test_compact_chat_part_boundary_and_append_share_identity() -> None:
    from linktools.ai.core import ExecutionLineageKind
    from linktools.ai.execution import ExecutionStreamEvent, ExecutionTreeEvent
    from linktools.ai.web._app import _chat_frame

    def frame(text: str, *, start: bool = False, truncated: bool = False):
        return _chat_frame(ExecutionTreeEvent(
            "run", "default", ExecutionLineageKind.RUN, None, "run", None, 0,
            ExecutionStreamEvent("run", None, "ASSISTANT_THINKING_DELTA", {
                "agent_run_seq": 1, "model_request_seq": 2, "stream_attempt_seq": 3,
                "part_index": 0, "part_start": start, "text": text, "stream_truncated": truncated,
            }),
            cursor="checkpoint",
        ))

    identity = {"execution_id": "run", "part": "1.2.3.0"}
    assert frame("", start=True) == ("thinking", {**identity, "start": True}, "checkpoint")
    assert frame(" 中 ") == ("thinking", {**identity, "text": " 中 "}, "checkpoint")
    assert frame("", truncated=True) == ("thinking", {**identity, "truncated": True}, "checkpoint")
    assert frame("") is None


@pytest.mark.asyncio
async def test_local_chat_streams_public_progress_then_reads_committed_turn_after_reopen(tmp_path: Path) -> None:
    from pydantic_ai.models.function import DeltaThinkingPart, FunctionModel

    from linktools.ai.workspace import Workspace
    from .test_live_history_readback_integration import _Models

    signature = "opaque-provider-signature"
    release, disconnect = asyncio.Event(), asyncio.Event()
    answer = "First answer "

    async def model(_messages, _info):
        yield {0: DeltaThinkingPart(content="provider-internal reasoning", signature=signature)}
        yield answer
        await release.wait()
        yield "and last"

    workspace = Workspace.initialize(tmp_path)
    async with common_command._open_local_runtime(
        workspace, model=_Models(FunctionModel(stream_function=model)),
    ) as local:
        app = create_app(runtime=local.runtime, local_workspace=True)
        path = "/api/session/events"
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"},
            "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"session_id=session", "headers": [],
            "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345),
        }
        frames: asyncio.Queue[dict[str, object]] = asyncio.Queue()

        async def receive():
            await disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] != "http.response.body":
                return
            body = message.get("body", b"")
            assert signature.encode() not in body
            lines = body.splitlines()
            for line in lines:
                if line.startswith(b"data: "):
                    frame = json.loads(line[6:])
                    if frame["type"] in ("delta", "tool", "progress", "preview_reset", "thinking_completed"):
                        assert "turn_no" not in frame and "revision" not in frame
                        assert not any(line.startswith(b"id: ") for line in lines)
                    if frame["type"] == "progress":
                        assert frame["item"]["status"] != "RUNNING"
                        assert "elapsed_ms" not in frame["item"]
                    frames.put_nowait(frame)

        async with client(app) as http:
            created = await http.post("/api/sessions", json={"session_id": "session", "request_id": "create"}, headers=_HEADERS)
            assert created.status_code == 201, created.text
            stream = asyncio.create_task(app(scope, receive, send))
            try:
                async with asyncio.timeout(10):
                    snapshot = await frames.get()
                    assert snapshot["type"] == "snapshot" and snapshot["session"]["active_turn_id"] is None
                    started = await http.post("/api/session/messages?session_id=session", json={
                        "request_id": "turn", "prompt": "hello", "agent_id": "default",
                    }, headers=_HEADERS)
                    assert started.status_code == 202, started.text
                    turn_id = started.json()["turn_id"]
                    saw_thinking = False
                    while True:
                        frame = await frames.get()
                        assert frame["turn_id"] == turn_id
                        if frame["type"] == "terminal":
                            pytest.fail(f"Turn ended before first text delta: state={frame.get('state')}, "
                                        f"code={frame.get('error_code')}")
                        if frame["type"] == "delta" and frame["delta_kind"] == "thinking":
                            assert frame["text"] == "provider-internal reasoning"
                            saw_thinking = True
                        if frame["type"] == "delta" and frame["delta_kind"] == "text":
                            assert frame["text"] == answer and saw_thinking and not release.is_set()
                            break
                    release.set()
                    while (terminal := await frames.get())["type"] != "terminal":
                        assert terminal["turn_id"] == turn_id
                    assert terminal["turn_id"] == turn_id and _turn_text(terminal) == answer + "and last"
                    thinking, assistant = terminal["items"]
                    assert thinking["item_kind"] == "thinking"
                    assert thinking["content"] == "provider-internal reasoning"
                    assert type(thinking["elapsed_ms"]) is int and thinking["elapsed_ms"] >= 0
                    assert assistant["item_kind"] == "assistant"
                    assert not stream.done()
            finally:
                release.set()
                disconnect.set()
                await stream

    async with common_command._open_local_runtime(
        Workspace.load(tmp_path), model=_Models(FunctionModel(stream_function=model)),
    ) as reopened, client(create_app(runtime=reopened.runtime, local_workspace=True)) as http:
        restored = await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})
        assert restored.status_code == 200, restored.text
        assert restored.json()["conversation_committed"]
        assert _turn_text(restored.json()) == _turn_text(terminal)
        assert restored.json()["items"] == terminal["items"]
        assert signature not in restored.text


@pytest.mark.asyncio
async def test_chat_tool_positions_match_history_when_parallel_results_finish_out_of_order(tmp_path: Path) -> None:
    from linktools.ai.capability import AgentContext
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.runtime.turn_clients import TurnObserver

    class Policy(TenantAuthorizationPolicy):
        pass
    from pydantic_ai.messages import (
        ModelMessage,
        ModelResponse,
        TextPart,
        ThinkingPart,
        ToolCallPart,
        ToolReturn,
        ToolReturnPart,
    )
    from pydantic_ai.models.function import AgentInfo

    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    release_first = asyncio.Event()

    async def lookup(context: AgentContext[None], path: str) -> ToolReturn:
        if path == "first":
            await release_first.wait()
        return ToolReturn(return_value=path, metadata=f"application metadata: {path}")

    def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        results = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if results:
            assert {part.tool_call_id: part.metadata for part in results} == {
                "z": "application metadata: first", "a": "application metadata: second",
            }
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "first"}, tool_call_id="z"),
            ThinkingPart("Between calls"),
            ToolCallPart("read_file", {"path": "second"}, tool_call_id="a"),
        ])

    features = _persistence_capabilities()
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.tool(lookup, name="read_file", effect_policy="none")
    features.agent("default", model="default", allow_tools=("read_file",))
    model_features(_Models(_UsageFunctionModel(respond)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("web-tool-order", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        created = await http.post("/api/sessions", headers=_HEADERS,
                                  json={"session_id": "tools", "request_id": "create"})
        assert created.status_code == 201, created.text
        started, finished = [], []
        live_elapsed = {}
        with TurnObserver(runtime, "tools", runtime.default_principal, local_workspace=True) as observer:
            accepted = await http.post("/api/session/messages?session_id=tools", headers=_HEADERS,
                                       json={"prompt": "lookup", "request_id": "turn", "agent_id": "default"})
            assert accepted.status_code == 202, accepted.text
            turn_id = accepted.json()["turn_id"]
            try:
                async with asyncio.timeout(10):
                    async for frame in observer.events(turn_id):
                        if frame["type"] == "terminal":
                            assert frame["state"] == "succeeded"
                            terminal = json.loads(json.dumps(frame))
                        if frame["type"] != "tool":
                            continue
                        item = frame["item"]
                        assert item.get("argument_preview") is None
                        if item["status"] == "RUNNING":
                            assert item.get("elapsed_ms") is None
                            started.append(item)
                        elif item["status"] == "SUCCEEDED":
                            elapsed = item["elapsed_ms"]
                            assert type(elapsed) is int and elapsed >= 0
                            live_elapsed[item["key"]] = elapsed
                            finished.append(item)
                            if item["key"] == "tool:a":
                                release_first.set()
            finally:
                release_first.set()

        replay = await http.get("/api/session", params={"session_id": "tools", "turn_id": turn_id})
        assert replay.status_code == 200, replay.text
        turn = replay.json()
        assert [item["key"] for item in started] == ["tool:z", "tool:a"]
        assert [item["key"] for item in finished] == ["tool:a", "tool:z"]
        assert turn["conversation_committed"]
        assert [item["item_kind"] for item in turn["items"]] == ["tool_call", "thinking", "tool_call", "assistant"]
        assert [item["tool_call_id"] for item in turn["items"] if item["item_kind"] == "tool_call"] == ["z", "a"]
        assert turn["items"] == terminal["items"]
        assert turn["process"] == terminal["process"]
        assert all(item["content"] is None for item in turn["items"] if item["item_kind"] == "tool_call")
        assert _turn_text(turn) == "done"
        assert [item["key"] for item in turn["process"]] == ["tool:z", "tool:a"]
        saved = {item["key"]: tuple(item["position"]) for item in turn["process"]}
        assert {item["key"]: tuple(item["position"]) for item in started} == saved
        assert {item["key"]: tuple(item["position"]) for item in finished} == saved
        assert {item["key"]: item["elapsed_ms"] for item in turn["process"]} == live_elapsed
        assert "application metadata:" not in replay.text

    async with Persistence.open("web-tool-order", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        restored = await http.get("/api/session", params={"session_id": "tools", "turn_id": turn_id})
        assert restored.status_code == 200, restored.text
        assert restored.json()["items"] == terminal["items"]
        assert {item["key"]: item["elapsed_ms"] for item in restored.json()["process"]} == live_elapsed


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", (True, False))
async def test_passive_session_tool_notice_requires_event_authorization(allowed: bool) -> None:
    from types import SimpleNamespace

    from linktools.ai.core import Principal, SessionStatus
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.session import project_tool_process_event

    class Feed:
        def subscribe(self, identity, on_process):
            assert identity == "session"
            self.deliver = on_process
            return lambda: None

    feed = Feed()
    end = asyncio.Event()

    async def watch(identity, *, principal):
        assert identity == "session" and principal.principal_id == "runtime"
        yield SimpleNamespace(status=SessionStatus.OPEN, active_execution_id="external", revision=1)
        await end.wait()

    async def authorize(identity, *, principal):
        assert identity == "external" and principal.principal_id == "runtime"
        process = project_tool_process_event("TOOL_CALL_FINISHED", {
            "call_id": "private", "tool_name": "private", "status": "SUCCEEDED", "tool_elapsed_ms": 42,
        })
        assert process is not None
        feed.deliver("external", process)
        asyncio.get_running_loop().call_later(0.05, end.set)
        if not allowed:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

    runtime = SimpleNamespace(
        history=None, default_principal=Principal("runtime", "default"),
        sessions=SimpleNamespace(supports_turn_sessions=False, watch=watch),
        events=SimpleNamespace(has_local_producer=lambda _: False, authorize_read=authorize),
    )
    async with client(create_app(runtime=runtime, session_notices=feed)) as http:
        response = await http.get("/api/session/events?session_id=session")
    assert response.status_code == 200, response.text
    if allowed:
        assert "event: session_process" in response.text
        payload = next(json.loads(line[6:]) for line in response.text.splitlines()
                       if line.startswith("data: ") and '"elapsed_ms"' in line)
        assert payload["item"]["elapsed_ms"] == 42
    else:
        assert "session_process" not in response.text and "private" not in response.text
    assert '"active_execution_id": "external"' in response.text



@pytest.mark.asyncio
@pytest.mark.parametrize("native", [True, False])
async def test_workspace_tool_title_requires_native_registration(tmp_path: Path, native: bool) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

    from linktools.ai.capability import ExecutionCapability, TraceCapability
    from linktools.ai.workspace import Workspace

    from ._runtime_test_helpers import _Models, _UsageFunctionModel, workspace_features

    (tmp_path / "README.md").write_text("private source contents", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("second source contents", encoding="utf-8")

    def respond(messages, _info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[
            ToolCallPart("read_file", {"path": "README.md"}, tool_call_id="simple"),
            ToolCallPart("read_file", {"path": ["README.md", "notes.txt"]}, tool_call_id="complex"),
        ])

    if native:
        features = workspace_features("tool-preview", workspace=Workspace.load(tmp_path))
    else:
        features = RuntimeFeatures()

        async def read_file(path: str | list[str]) -> str:
            return "custom source"

        features.tool(read_file, effect_policy="none")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("default", allow_tools=("read_file",))
    model_features(_Models(_UsageFunctionModel(respond)), features)
    async with Persistence.open("tool-preview", storage=PersistenceStorage.in_memory()) as state, Runtime.open(
        persistence=state, features=features,
    ) as runtime, client(create_app(runtime=runtime)) as http:
        session = await runtime.agents.get("default").create_session("preview")
        execution = await session.start("read the docs")
        observed = {}

        async def on_event(item) -> None:
            if item.event.event_type == "TOOL_CALL_STARTED":
                payload = item.event.payload
                observed[f"tool:{payload['call_id']}"] = payload.get("argument_preview")

        assert (await execution.wait(
            timeout_seconds=10, on_event=on_event, include_event_content=True,
        )).result.status.value == "SUCCEEDED"
        history = await http.get("/api/session?session_id=preview")
        assert history.status_code == 200, history.text
        turn, = history.json()["timeline"]["items"]
        assert {item["key"]: item.get("argument_preview") for item in turn["process"]} == observed == {
            "tool:simple": "README.md" if native else None, "tool:complex": None,
        }
        assert "private source contents" not in history.text and "notes.txt" not in history.text

        trace = await execution.model_interactions(include_content=True)
        first_tool, = trace.items[0].request["parameters"]["function_tools"]
        metadata = first_tool["metadata"]
        assert metadata["linktools.ai.effect_policy"] == "none"
        assert "linktools.ai.argument_preview_field" not in metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("retry", [False, True, "validation"])
async def test_chat_tool_failure_uses_exact_model_response_without_durable_audit_copy(
    tmp_path: Path, retry: bool | str,
) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

    from linktools.ai.capability import AgentContext, ToolCallFailed, ToolCallRetry
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.runtime.turn_clients import TurnObserver
    from linktools.ai.web._app import _native_turn_event_wire

    class Policy(TenantAuthorizationPolicy):
        pass

    from ._runtime_test_helpers import _Models, _UsageFunctionModel

    failure = "Use the original filename.\n  The second line is also model-visible."
    invalid_value = "input not meant for the Web tool card"
    release = asyncio.Event()
    received = []

    async def lookup(context: AgentContext[None], value: int = 0) -> str:
        assert retry != "validation", "invalid arguments must not reach the tool"
        await release.wait()
        raise (ToolCallRetry if retry else ToolCallFailed)(failure)

    def respond(messages, _info):
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                received.append(part.content)
                return ModelResponse(parts=[TextPart("adapted")])
        args = {"value": invalid_value} if retry == "validation" else {}
        return ModelResponse(parts=[ToolCallPart("lookup", args, tool_call_id="failed-lookup")])

    features = _persistence_capabilities()
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(StepPersistenceCapability(), scope="runtime")
    features.tool(lookup, effect_policy="none")
    features.agent("default", allow_tools=("lookup",))
    model_features(_Models(_UsageFunctionModel(respond)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("chat-tool-failure", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("failures", principal=principal)
        seen = []
        local_frames = []
        unsubscribe = runtime.sessions.observe_turn_session(
            "failures", local_frames.append, local_tool_feedback=True,
        )
        with TurnObserver(runtime, "failures", principal, local_workspace=True) as observer:
            turn_id = await runtime.sessions.start_turn(
                "failures", "lookup", principal=principal, agent="default", idempotency_key="turn",
            )
            try:
                async with asyncio.timeout(10):
                    async for frame in observer.events(turn_id):
                        if frame["type"] == "terminal":
                            assert frame["state"] == "succeeded"
                        if frame["type"] != "tool":
                            continue
                        item = frame["item"]
                        assert item.get("error_message") is None
                        if item["status"] == "RUNNING":
                            release.set()
                        if item["status"] == "FAILED":
                            seen.append(item["error_code"])
            finally:
                release.set()
                unsubscribe()
        if retry == "validation":
            [errors] = received
            [error] = errors
            assert error["input"] == invalid_value
            failure = f"value: {error['msg']}"
        else:
            assert received == [failure]
        assert len(seen) == 1 and seen[0]
        local_failed, = [frame for frame in local_frames if frame["type"] == "tool"
                         and frame["item"]["status"] == "FAILED"]
        assert _native_turn_event_wire(local_failed, local_tool_feedback=True)["item"]["error_message"] == failure
        assert "error_message" not in _native_turn_event_wire(local_failed)["item"]
        terminal, = [frame for frame in local_frames if frame["type"] == "terminal"]
        assert terminal["process"][0]["error_message"] == failure

    async with Persistence.open("chat-tool-failure", storage=storage) as reopened, Runtime.open(
        persistence=reopened, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        replay = await http.get("/api/session", params={"session_id": "failures", "turn_id": turn_id})
        assert replay.status_code == 200, replay.text
        item, = replay.json()["process"]
        assert item["status"] == "FAILED" and item["error_message"] == failure
        if retry == "validation":
            assert invalid_value not in item["error_message"]
        shared = await runtime.history.turn_result(
            "failures", turn_id, principal=runtime.default_principal, local_workspace=False,
        )
        assert shared["process"][0].get("error_message") is None
