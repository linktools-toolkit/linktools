#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Web transport parity, local trust boundary, and observer lifetime."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from linktools.ai.capability import CapabilityGroup
from linktools.ai.errors import ErrorDiagnostics
from linktools.ai.observe import Metrics
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.web import create_app
from ._runtime_test_helpers import RuntimeUsageModels

_HEADERS = {"X-LinkTools-Console": "1"}


def client(app: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765")


@pytest.mark.asyncio
async def test_local_origin_boundary_blocks_browser_cross_origin_and_rebinding() -> None:
    app = create_app()
    async with client(app) as http:
        response = await http.get("/")
        assert response.status_code == 200
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
        assert response.headers["cache-control"] == "private, no-store"
        for headers in (
            {"Host": "evil.example:8765"}, {"Host": "127.0.0.1:8000"},
            {"Origin": "https://evil.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"},
        ):
            assert (await http.get("/api/config", headers=headers)).status_code == 403
        assert (await http.post("/api/sessions", json={})).status_code == 403
        assert (await http.options("/api/sessions", headers={"Origin": "https://evil.example", "Access-Control-Request-Headers": "x-linktools-console"})).status_code == 403
        assert (await http.post("/api/sessions", json={}, headers=_HEADERS)).status_code == 503
        assert (await http.get("/assets/missing.js")).status_code == 404
        assert (await http.get("/api/config", headers={"Host": "localhost:8765", "Origin": "http://localhost:8765"})).status_code == 200


@pytest.mark.asyncio
async def test_session_execution_history_trace_result_and_metrics_use_runtime() -> None:
    metrics = Metrics.in_memory()
    async with Runtime.open("web-test", models=RuntimeUsageModels(), storage=RuntimeStorage.in_memory(), metrics=metrics) as runtime:
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
async def test_execution_stream_replays_cursor_and_finishes_with_authoritative_snapshot() -> None:
    async with Runtime.open("web-stream", models=RuntimeUsageModels(), storage=RuntimeStorage.in_memory()) as runtime:
        execution = await runtime.agents.get().start("hello")
        await execution.wait()
        async with client(create_app(runtime=runtime)) as http:
            response = await http.get(f"/api/executions/{execution.execution_id}/events")
            assert response.status_code == 200
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
    storage = RuntimeStorage.from_root(tmp_path)
    async with Runtime.open("default", models=RuntimeUsageModels(), storage=storage) as runtime:
        execution = await runtime.agents.get().start("saved", session_id="session")
        await execution.wait()
        identity = execution.execution_id
    async with RuntimeHistory.open("default", storage=RuntimeStorage.from_root(tmp_path)) as history:
        async with client(create_app(history=history)) as http:
            assert (await http.get("/api/config")).json()["read_only"]
            sessions = (await http.get("/api/sessions")).json()
            assert sessions["recent_only"] and sessions["items"][0]["session_id"] == "session"
            assert (await http.get(f"/api/executions/{identity}/result")).json()["output"] == {"text": "done"}
            assert (await http.get(f"/api/executions/{identity}/events")).status_code == 503
            assert (await http.post(f"/api/executions/{identity}/recover", json={"request_id": "recover"}, headers=_HEADERS)).status_code == 503
            assert (await http.get("/api/executions?limit=0")).status_code == 400


@pytest.mark.asyncio
async def test_stream_disconnect_closes_observer_without_cancelling_execution() -> None:
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

    class RuntimeStub:
        default_principal = object()
        history = None
        executions = Executions()

    app = create_app(runtime=RuntimeStub())
    sent_body = asyncio.Event()
    initial = False

    async def receive() -> dict[str, object]:
        nonlocal initial
        if not initial:
            initial = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await sent_body.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, object]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            sent_body.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
             "scheme": "http", "path": "/api/executions/execution/events", "raw_path": b"/api/executions/execution/events",
             "query_string": b"", "headers": [(b"host", b"127.0.0.1:8765")], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}
    await asyncio.wait_for(app(scope, receive, send), 2)
    assert closed.is_set()
    assert not cancel_called


def test_web_command_preserves_runtime_flags_and_validates_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from linktools.commands.ai.web import command
    from linktools.cli import CommandError
    parser = command.create_parser()
    assert {"project", "model", "base_url", "api_key", "vision", "memory", "read_only", "port"} <= {action.dest for action in parser._actions}
    with pytest.raises(CommandError, match="port"):
        command.run(parser.parse_args(["--port", "0"]))


@pytest.mark.asyncio
async def test_remote_peer_cannot_claim_a_local_host_header() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(), client=("203.0.113.9", 1000)), base_url="http://127.0.0.1:8765") as http:
        response = await http.get("/api/config")
        assert response.status_code == 403
        assert response.json()["error_code"] == "LOOPBACK_REQUIRED"


@pytest.mark.asyncio
async def test_exception_diagnostics_never_serialize_provider_credentials() -> None:
    from dataclasses import dataclass

    @dataclass
    class FailedView:
        error_code: str
        error_diagnostics: ErrorDiagnostics

    class History:
        tenant_id = "default"

        async def inspect_execution(self, identity: str, **kwargs: object) -> FailedView:
            return FailedView("MODEL_UNAVAILABLE", ErrorDiagnostics.from_exception(ValueError("api_key=secret-key-value")))

    async with client(create_app(history=History())) as http:
        response = await http.get("/api/executions/example")
        assert response.status_code == 200
        assert "secret-key-value" not in response.text
        assert response.json()["error_diagnostics"]["exception_type"] == "ValueError"
        assert response.json()["error_diagnostics"]["cause_digest"]


def test_web_cli_owns_one_runtime_and_closes_it_after_server_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from contextlib import asynccontextmanager
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.cli import CommandError
    import linktools.commands.ai.web as web_command
    import uvicorn

    original_open = Runtime.open
    lifecycle = []

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
            async with client(self.config.app) as http:
                response = await http.get("/api/config")
                assert response.status_code == 200
                assert response.json()["read_only"] is False
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

    monkeypatch.setattr(Runtime, "open", tracked_open)
    monkeypatch.setattr(web_command, "_local_runtime_models", lambda *_: RuntimeUsageModels())
    monkeypatch.setattr(uvicorn, "Server", Server)
    args = web_command.command.create_parser().parse_args(["--project", str(tmp_path), "--model", "fake"])
    with pytest.raises(CommandError):
        web_command.command.run(args)
    assert lifecycle == ["open", "close"]


@pytest.mark.parametrize("explicit", [False, True])
def test_web_cli_read_only_does_not_provision_workspace_or_open_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, explicit: bool) -> None:
    import linktools.commands.ai.web as web_command
    import uvicorn

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("read-only console must not open Runtime")

    class Server:
        def __init__(self, config: object) -> None:
            self.config = config

        async def serve(self) -> None:
            async with client(self.config.app) as http:
                response = await http.get("/api/config")
                assert response.json()["read_only"]
                assert response.json()["vision"] is True
                assert response.json()["api_key_configured"]
                assert "secret-key-value" not in response.text
                assert (await http.get("/api/sessions")).json()["items"] == []

    monkeypatch.setattr(Runtime, "open", forbidden)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("LINKTOOLS_OPENAI_VISION", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "secret-key-value")
    env = web_command.command.environ
    monkeypatch.setattr(env, "config", env.build_config("web-test", "LINKTOOLS_"))
    args = web_command.command.create_parser().parse_args(["--project", str(tmp_path)] + (["--read-only"] if explicit else []))
    assert web_command.command.run(args) == 0
    assert not (tmp_path / ".linktools").exists()


@pytest.mark.parametrize("vision", [False, True])
def test_web_vision_uses_the_shared_typed_cli_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, vision: bool) -> None:
    from linktools.ai.workspace import Workspace
    from linktools.commands.ai._common import _local_runtime_models
    from linktools.commands.ai.run import command as run_command
    from linktools.commands.ai.web import command as web_command

    monkeypatch.setenv("LINKTOOLS_OPENAI_VISION", str(vision).lower())
    env = web_command.environ
    monkeypatch.setattr(env, "config", env.build_config("web-test", "LINKTOOLS_"))
    workspace = Workspace.discover(tmp_path)
    for command, positional in ((run_command, ["prompt"]), (web_command, [])):
        args = command.create_parser().parse_args(positional + ["--model", "fake-model"])
        binding = _local_runtime_models(workspace, args).capture().resolve("default")
        assert args.vision is vision
        assert binding.vision is vision


@pytest.mark.asyncio
async def test_default_http_port_accepts_browser_canonical_host_and_origin() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(port=80)), base_url="http://127.0.0.1") as http:
        assert (await http.get("/api/config", headers={"Origin": "http://127.0.0.1"})).status_code == 200
        assert (await http.get("/api/config", headers={"Origin": "http://evil.example"})).status_code == 403


@pytest.mark.asyncio
async def test_opaque_session_ids_round_trip_through_query_identity() -> None:
    identity = "../team/conversation?notes#你好"
    async with Runtime.open("opaque-web", models=RuntimeUsageModels(), storage=RuntimeStorage.in_memory()) as runtime:
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
