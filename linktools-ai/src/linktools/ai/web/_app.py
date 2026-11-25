#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-origin HTTP delivery; Runtime owns execution and durable state."""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import ExitStack, aclosing, asynccontextmanager, nullcontext, suppress
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio
import httpx2
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from linktools.core import environ
from ..core import Page, Principal, PrincipalKind
from ..errors import AIError, ErrorCode, ErrorDiagnostics, ObservationError
from ..model import ModelClientRequest, ModelClientResponse
from ..observe import MetricAggregation, MetricQuery, Metrics, MetricWindow, RuntimeMetricsStatus
from ..history import History
from ..execution import ExecutionTreeEvent, ListExecutionRequest, TaskModelProjection, ToolEffectApplied, ToolEffectFailed, ToolEffectNotApplied
from ..session import CreateSessionRequest, ListSessionRequest
from ..execution.stream import local_execution_stream
from ..runtime import Runtime
from ..runtime.turn_clients import uses_turn_sessions
from ..session import SessionTurn, ToolProcessItem, project_tool_process_event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from ..execution import ExecutionObservationEvent
    from ..runtime import TurnNoticeTransport
    from starlette.types import ASGIApp, Receive, Scope, Send

_logger = environ.get_logger("ai.web")
_ASSETS = Path(__file__).with_name("assets")
_SUMMARY_METRICS = (
    "linktools.execution.count", "linktools.execution.failure_ratio", "linktools.execution.latency",
    "linktools.model.request.count", "linktools.model.request.failure_ratio", "linktools.model.request.latency",
    "linktools.tool.execution.count", "linktools.tool.execution.failure_ratio",
    "linktools.model.input_tokens", "linktools.model.output_tokens",
    "linktools.model.cache_read_tokens", "linktools.model.cache_write_tokens",
)
_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'",
}
_JSON_REQUEST_MAX_BYTES = 8 * 1024 * 1024
# Transport bound, not the Runtime's canonical turn limit. Allow JSON escaping and the HTTP envelope.
_TURN_JSON_REQUEST_MAX_BYTES = 512 * 1024
_TURN_SSE_LIMIT = 128
_TURN_SSE_BUFFER = 256 * 1024

_SSE_HEADERS = {
    **_HEADERS,
    "Cache-Control": _HEADERS["Cache-Control"] + ", no-transform",
    "X-Accel-Buffering": "no",
}


def _wire(value: object) -> object:
    if isinstance(value, RuntimeMetricsStatus):
        return {
            "recorders": _wire(value.recorders),
            "enabled": value.enabled,
            "accepting": value.accepting,
            "accepted": value.accepted,
            "persisted": value.persisted,
            "pending": value.pending,
            "rejected": value.rejected,
            "unconfirmed": value.unconfirmed,
            "write_failures": value.write_failures,
            "queue_size": value.queue_size,
            "queue_capacity": value.queue_capacity,
            "failure_codes": _wire(value.failure_codes),
        }
    if isinstance(value, ErrorDiagnostics):
        return _wire(value.to_payload())
    if isinstance(value, (ModelClientRequest, ModelClientResponse)):
        result = {field.name: _wire(getattr(value, field.name)) for field in fields(value) if field.name != "body"}
        result["body"] = None if value.body is None else base64.b64encode(value.body).decode("ascii")
        result["body_encoding"] = "base64"
        if isinstance(value, ModelClientResponse) and value.body is not None and value.content_encoding:
            result["decoded_text"], result["decode_error"] = _response_display(value)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, ToolProcessItem):
            if value.elapsed_ms is None:
                result.pop("elapsed_ms")
            if value.error_code is None:
                result.pop("error_code")
            if value.error_message is None:
                result.pop("error_message")
            if value.argument_preview is None:
                result.pop("argument_preview")
        return result
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    return value


def _chat_timeline(page: Page[SessionTurn]) -> Page[SessionTurn]:
    # Tool markers preserve placement; names/status come from process, not tool payloads.
    turns = []
    for turn in page.items:
        seen_calls = set()
        items = []
        for item in turn.items:
            if item.item_kind in {"assistant", "thinking"}:
                items.append(item)
            elif item.item_kind == "tool_call" and item.tool_call_id and item.tool_call_id not in seen_calls:
                seen_calls.add(item.tool_call_id)
                items.append(replace(item, content=None, tool_name=None))
        turns.append(replace(turn, items=tuple(items), safe_error_details={}))
    return replace(page, items=tuple(turns))


_CHAT_STREAM_POSITION = {"agent_run_seq", "model_request_seq", "stream_attempt_seq"}
_CHAT_TEXT_FIELDS = _CHAT_STREAM_POSITION | {"text", "part_index", "part_start", "stream_truncated"}
_CHAT_STREAM_FIELDS = {
    "ASSISTANT_TEXT_DELTA": _CHAT_TEXT_FIELDS,
    "ASSISTANT_THINKING_DELTA": _CHAT_TEXT_FIELDS,
    "ASSISTANT_PART_COMPLETED": _CHAT_STREAM_POSITION | {"part_index", "thinking_elapsed_ms"},
    "MODEL_ATTEMPT_STARTED": _CHAT_STREAM_POSITION,
    "MODEL_ATTEMPT_FINISHED": _CHAT_STREAM_POSITION | {"status"},
    "MODEL_REQUEST_FINISHED": _CHAT_STREAM_POSITION | {"status"},
    "STREAM_RESET": {"stream_truncated"},
    "EXECUTION_FAILED": set(),
    "EXECUTION_CANCELLED": set(),
}


_CHAT_DELTA_EVENTS = {"ASSISTANT_TEXT_DELTA": "text", "ASSISTANT_THINKING_DELTA": "thinking"}
_CHAT_PART_POSITION = ("agent_run_seq", "model_request_seq", "stream_attempt_seq", "part_index")

_ExecutionFrame = tuple[str, dict[str, object], str | None]
_MAX_CHAT_TURN_FRAME_BYTES = 256 * 1024


def _chat_frame(item: ExecutionTreeEvent) -> _ExecutionFrame | None:
    if item.depth != 0:
        return None
    cursor = item.cursor
    event = item.event
    process = project_tool_process_event(event.event_type, event.payload)
    if process is not None and isinstance(event.payload, Mapping):
        frame = {"type": "process", "item": _wire(process), "cursor": cursor}
        agent = event.payload.get("agent_run_seq")
        request = event.payload.get("tool_model_request_seq")
        if (process.position is not None and type(agent) is int and agent > 0
                and type(request) is int and request > 0):
            frame["order"] = (agent, request, process.position[1])
        return "message", frame, cursor
    allowed = _CHAT_STREAM_FIELDS.get(event.event_type)
    if allowed is None or not isinstance(event.payload, Mapping):
        return None
    source = event.payload
    if event.event_type == "ASSISTANT_PART_COMPLETED":
        elapsed = source.get("thinking_elapsed_ms")
        if source.get("part_type") != "thinking" or type(elapsed) is not int or elapsed < 0:
            return None
    kind = _CHAT_DELTA_EVENTS.get(event.event_type)
    if kind:
        text = source.get("text")
        start = source.get("part_start")
        truncated = source.get("stream_truncated") is True
        if not (text or start is True or truncated):
            return None
        part = tuple(source.get(key) for key in _CHAT_PART_POSITION)
        positioned = type(start) is bool and all(type(value) is int for value in part)
        # The SDK can emit text before a model attempt position is available.
        if positioned or start is None and all(value is None for value in part):
            payload: dict[str, object] = {"execution_id": item.execution_id}
            if positioned:
                payload["part"] = ".".join(map(str, part))
            if text:
                payload["text"] = text
            if start:
                payload["start"] = True
            if truncated:
                payload["truncated"] = True
            return kind, payload, cursor
    return "message", {"type": "event", "cursor": cursor, "item": {
        "execution_id": item.execution_id, "depth": 0,
        "event": {"event_type": event.event_type, "durable_seq": event.durable_seq,
                  "payload": {key: value for key, value in source.items() if key in allowed}},
    }}, cursor


def _sse_frame(event: str, data: object, cursor: str | None = None) -> str:
    prefix = "" if cursor is None else f"id: {cursor}\n"
    if event != "message":
        prefix += f"event: {event}\n"
    return prefix + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


def _session_execution_cursor(value: str | None) -> tuple[str | None, str | None]:
    if value is not None:
        try:
            parts = json.loads(value)
            if isinstance(parts, list) and len(parts) == 2 and all(isinstance(part, str) and part for part in parts):
                return parts[0], parts[1]
        except (ValueError, RecursionError):
            pass
    return None, None


def _response_display(response: ModelClientResponse) -> tuple[str | None, str | None]:
    supported = {"identity", "gzip", "deflate"}
    if find_spec("brotli") is not None or find_spec("brotlicffi") is not None:
        supported.add("br")
    encodings = {value.strip().lower() for value in response.content_encoding.split(",")}
    if not encodings <= supported:
        return None, "content_encoding_unavailable"
    try:
        decoded = httpx2.Response(200, headers={"content-encoding": response.content_encoding}, content=response.body).content
        return decoded.decode("utf-8"), None
    except (httpx2.DecodingError, UnicodeDecodeError):
        return None, "response_decode_failed"


def _json(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(_wire(value), status_code=status, headers=_HEADERS)


def _ordinary_turn_wire(turn: Mapping[str, object]) -> Mapping[str, object]:
    """Keep one copy of the final text when the Web timeline already displays it."""
    items, output = turn.get("items"), turn.get("output")
    if not isinstance(items, list) or not isinstance(output, Mapping) or set(output) != {"text"}:
        return turn
    text = output["text"]
    if not isinstance(text, str):
        return turn
    for item in reversed(items):
        if isinstance(item, Mapping) and item.get("item_kind") == "assistant":
            if item.get("content") == text:
                return {key: value for key, value in turn.items() if key != "output"}
            break
    return turn


def _text(payload: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _paging(request: Request) -> dict[str, object]:
    limit = int(request.query_params.get("limit", "50"))
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    return {"cursor": request.query_params.get("cursor") or None, "limit": limit}


def _selectors(request: Request, *, history: bool = False) -> dict[str, object]:
    keys = ("agent_run_seq", "model_request_seq", "step_index")
    if history:
        keys += ("message_seq", "part_index")
    selected: dict[str, object] = {}
    for key in keys:
        if key in request.query_params:
            selected[key] = int(request.query_params[key])
    if request.query_params.get("tool_call_id"):
        selected["tool_call_id"] = request.query_params["tool_call_id"]
    return selected


def _content(request: Request, field: str = "include_content") -> bool:
    value = request.query_params.get(field, "false")
    if value not in {"true", "false"}:
        raise ValueError(f"{field} must be true or false")
    return value == "true"


async def _body(request: Request, *, ordinary_turn: bool = False) -> dict[str, object]:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise ValueError("application/json is required")
    maximum = _TURN_JSON_REQUEST_MAX_BYTES if ordinary_turn else _JSON_REQUEST_MAX_BYTES
    if ordinary_turn and int(request.headers.get("content-length", "0")) > maximum:
        raise ValueError("request exceeds JSON body limit")
    data = bytearray()
    async for chunk in request.stream():
        if len(chunk) > maximum - len(data):
            raise ValueError("request exceeds JSON body limit")
        data.extend(chunk)
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    return payload


class _LocalBoundary:
    def __init__(self, app: "ASGIApp") -> None:
        self.app = app

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        peer = scope.get("client")
        if peer is None or peer[0] not in {"127.0.0.1", "::1"}:
            await _json({"error_code": "LOOPBACK_REQUIRED"}, 403)(scope, receive, send)
            return
        if request.headers.get("sec-fetch-site") == "cross-site":
            await _json({"error_code": "LOCAL_ORIGIN_REQUIRED"}, 403)(scope, receive, send)
            return
        if request.method not in {"GET", "HEAD"} and request.headers.get("x-linktools-console") != "1":
            await _json({"error_code": "CONSOLE_REQUEST_REQUIRED"}, 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _SSEStream(StreamingResponse):
    def __init__(self, body: AsyncGenerator[str, None], shutdown_event: asyncio.Event | None) -> None:
        super().__init__(body, media_type="text/event-stream", headers=_SSE_HEADERS)
        self._body = body
        self._shutdown_event = shutdown_event

    async def stream_response(self, send: "Send") -> None:
        if self._shutdown_event is None:
            async with aclosing(self._body):
                await super().stream_response(send)
            return

        await send({"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers})

        async def stream() -> None:
            async with aclosing(self._body):
                async for chunk in self._body:
                    await send({"type": "http.response.body", "body": chunk.encode(self.charset), "more_body": True})

        response = asyncio.create_task(stream())
        shutdown = asyncio.create_task(self._shutdown_event.wait())
        try:
            await asyncio.wait((response, shutdown), return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not response.done():
                response.cancel()
            shutdown.cancel()
            # Starlette can also cancel this scope when the browser disconnects.
            # Join both tasks without interrupting our observer's cleanup.
            with anyio.CancelScope(shield=True):
                try:
                    with suppress(asyncio.CancelledError):
                        await response
                finally:
                    with suppress(asyncio.CancelledError):
                        await shutdown
        await send({"type": "http.response.body", "body": b"", "more_body": False})


class _SessionNoticeFeed(Protocol):
    def subscribe(
        self, session_id: str, on_process: "Callable[[str, ToolProcessItem], None]",
    ) -> "Callable[[], None]": ...


class _Console:
    def __init__(
        self, runtime: Runtime | None, history: History | None, metrics: Metrics | None,
        status: Mapping[str, object], capabilities: Sequence[Mapping[str, object]],
        shutdown_event: asyncio.Event | None, session_notices: _SessionNoticeFeed | None,
        local_workspace: bool, native_sessions: bool,
    ) -> None:
        self.runtime = runtime
        self.history = runtime.history if runtime is not None else history
        self.metrics = metrics
        self.status = dict(status)
        self.capabilities = tuple(capabilities)
        self._shutdown_event = shutdown_event
        self._session_notices = session_notices
        self._local_workspace = local_workspace
        self.turn_sessions = native_sessions
        self._turn_subscribers = 0
        self.principal = runtime.default_principal if runtime is not None else Principal(
            "runtime", "default" if history is None else history.tenant_id, PrincipalKind.LOCAL_TRUSTED.value,
        )

    def require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, safe_details={"read_only": True})
        return self.runtime

    def require_history(self) -> History:
        if self.history is None:
            raise AIError(ErrorCode.EXECUTION_NOT_FOUND)
        return self.history

    @property
    def local_turn_workspace(self) -> bool:
        return self._local_workspace and self.principal.kind == PrincipalKind.LOCAL_TRUSTED.value

    async def config(self, request: Request) -> Response:
        return _json({
            **self.status,
            "runtime": self.runtime.status() if self.runtime is not None else None,
            "read_only": self.runtime is None,
            "turn_sessions": self.turn_sessions,
            **({"turn_trace": True} if self.turn_sessions and self.history is not None
               and self.history.has_turn_trace else {}),
            "capabilities": self.capabilities,
            "metric_names": _SUMMARY_METRICS,
        })

    async def sessions(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "session.create" if request.method == "POST" else "session.list"
        if request.method == "POST":
            runtime = self.require_runtime()
            payload = await _body(request, ordinary_turn=self.turn_sessions)
            request.state.operation_id = _text(payload, "request_id")
            if self.turn_sessions:
                cwd = payload.get("cwd")
                if cwd is not None and not isinstance(cwd, str):
                    raise ValueError("cwd must be a workspace-relative string or null")
                view = await runtime.sessions.create_turn_session(
                    _text(payload, "session_id"), principal=self.principal,
                    idempotency_key=request.state.operation_id,
                    metadata={"title": _text(payload, "title", "New conversation")}, cwd=cwd,
                )
                return _json(view, 201)
            view = await runtime.sessions.create_view(_text(payload, "agent_id", "default"), CreateSessionRequest(
                self.principal, _text(payload, "session_id"), request.state.operation_id,
                cwd=payload.get("cwd"), metadata={"title": _text(payload, "title", "New conversation")},
            ))
            return _json(view, 201)
        if self.turn_sessions:
            if not self.local_turn_workspace:
                return _json({"items": [], "next_cursor": None, "scope_required": True})
            return _json(await self.require_history().turn_sessions(
                principal=self.principal, local_workspace=True, **_paging(request),
            ))
        if self.runtime is not None:
            return _json(await self.runtime.sessions.list(ListSessionRequest(self.principal, **_paging(request))))
        # The read-only console lists the newest sessions by time, without pagination.
        if request.query_params.get("cursor"):
            raise ValueError("read-only recent sessions have no cursor")
        items = () if self.history is None else await self.history.scan_recent_sessions(
            principal=self.principal, limit=int(_paging(request)["limit"]),
        )
        return _json({"items": items, "next_cursor": None, "recent_only": True})

    async def session(self, request: Request) -> Response:
        history = self.require_history()
        identity = _text(request.query_params, "session_id")
        include_timeline = request.query_params.get("include_timeline", "true")
        if include_timeline not in {"true", "false"}:
            raise ValueError("include_timeline must be true or false")
        trace = request.query_params.get("trace")
        if trace is not None:
            if (trace != "true" or not self.turn_sessions or include_timeline != "true"
                    or "execution_id" in request.query_params):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "Trace requires an exact canonical turn")
            return _json(await history.turn_trace(
                identity, _text(request.query_params, "turn_id"), principal=self.principal,
                local_workspace=self.local_turn_workspace, **_paging(request),
            ))
        if self.turn_sessions:
            if "execution_id" in request.query_params:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "Identify the conversation turn with turn_id")
            if "turn_id" in request.query_params:
                turn = await history.turn_result(
                    identity, _text(request.query_params, "turn_id"), principal=self.principal,
                    local_workspace=self.local_turn_workspace,
                )
                return _json(_ordinary_turn_wire(turn))
            if include_timeline == "false":
                session = await history.turn_session_state(
                    identity, principal=self.principal, local_workspace=self.local_turn_workspace,
                )
                return _json({"session": session, "timeline": None})
            page = await history.turn_session(
                identity, principal=self.principal, local_workspace=self.local_turn_workspace, **_paging(request),
            )
            timeline = page["timeline"]
            if isinstance(timeline, Mapping) and isinstance(items := timeline.get("items"), list):
                return _json({
                    **page,
                    "timeline": {**timeline, "items": [
                        _ordinary_turn_wire(turn) if isinstance(turn, Mapping) else turn for turn in items
                    ]},
                })
            return _json(page)
        if "execution_id" in request.query_params:
            execution_id = _text(request.query_params, "execution_id")
            turn_state = request.query_params.get("turn_state", "committed")
            if include_timeline != "true" or turn_state not in {"active", "committed"}:
                raise ValueError("invalid session turn selector")
            if environ.debug:
                request.state.phase = "session.turn"
            if turn_state == "active":
                detail = await history.session_detail(identity, principal=self.principal, limit=1)
                session = detail.session
                items = tuple(turn for turn in detail.timeline.items if turn.execution_id == execution_id)
            else:
                turn = await history.session_turn(identity, execution_id, principal=self.principal)
                session = await history.inspect_session(identity, principal=self.principal)
                items = (turn,)
            return _json({"session": session, "timeline": _chat_timeline(Page(items, None))})
        if include_timeline == "true":
            if environ.debug:
                request.state.phase = "session.timeline"
            detail = await history.session_detail(identity, principal=self.principal, **_paging(request))
            return _json(replace(detail, timeline=_chat_timeline(detail.timeline)))
        if environ.debug:
            request.state.phase = "session.metadata"
        session = await history.inspect_session(identity, principal=self.principal)
        return _json({"session": session, "timeline": None})

    async def session_action(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "session.get"
        runtime = self.require_runtime()
        session_id = _text(request.query_params, "session_id")
        payload = await _body(request, ordinary_turn=self.turn_sessions)
        request_id = _text(payload, "request_id")
        action = request.path_params["action"]
        request.state.operation_id = request_id
        if environ.debug:
            request.state.phase = f"session.{action}"
        if self.turn_sessions:
            if action == "messages":
                if payload.get("planning"):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, "Ordinary turns do not support plan mode")
                files = payload.get("files", [])
                if not isinstance(files, list) or any(not isinstance(path, str) for path in files):
                    raise ValueError("files must be a list of workspace-relative paths")
                turn_id = await runtime.sessions.start_turn(
                    session_id, _text(payload, "prompt"), agent_id=_text(payload, "agent_id", "default"),
                    principal=self.principal, idempotency_key=request_id, thinking=payload.get("thinking"), files=files,
                )
                return _json({"turn_id": turn_id}, 202)
            if action in {"cancel", "recover"}:
                operation = runtime.sessions.cancel_turn if action == "cancel" else runtime.sessions.recover_turn
                accepted = await operation(session_id, _text(payload, "turn_id"), principal=self.principal)
                return _json({"accepted": accepted})
            if action in {"update", "close", "fork"}:
                revision = payload.get("expected_revision")
                if type(revision) is not int or revision < 0:
                    raise ValueError("expected_revision must be a nonnegative integer")
                if action == "fork":
                    target_id = _text(payload, "new_session_id")
                    created = await runtime.sessions.fork_turn_session(
                        session_id, target_id, principal=self.principal,
                        expected_revision=revision, idempotency_key=request_id,
                    )
                    if not created:
                        raise AIError(ErrorCode.SESSION_CONFLICT, "The source conversation changed")
                    return _json({"session_id": target_id}, 201)
                receipt: dict[str, object] = {"session_id": session_id, "revision": revision + 1}
                if action == "update":
                    metadata = payload.get("metadata")
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be an object")
                    cwd = payload.get("cwd")
                    if cwd is not None and not isinstance(cwd, str):
                        raise ValueError("cwd must be a workspace-relative string or null")
                    changed = await runtime.sessions.update_turn_session(
                        session_id, principal=self.principal, expected_revision=revision, metadata=metadata,
                        cwd=cwd, replace_cwd="cwd" in payload,
                    )
                    receipt["metadata"] = metadata
                else:
                    changed = await runtime.sessions.close_turn_session(
                        session_id, principal=self.principal, expected_revision=revision,
                    )
                    receipt["status"] = "CLOSED"
                    receipt["active_turn_id"] = None
                if not changed:
                    raise AIError(ErrorCode.SESSION_CONFLICT, "The conversation changed or has an active turn")
                return _json(receipt)
            return _json({"error_code": "ACTION_NOT_FOUND"}, 404)
        if action == "messages":
            execution = await runtime.sessions.start(
                session_id, _text(payload, "prompt"), principal=self.principal,
                files=payload.get("files", ()), idempotency_key=request_id,
                planning=payload.get("planning", False), thinking=payload.get("thinking"),
            )
            return _json({"execution_id": execution.execution_id}, 202)
        session = await runtime.sessions.get(session_id, principal=self.principal)
        if action == "close":
            return _json(await session.close(idempotency_key=request_id))
        if action == "fork":
            fork = await session.fork(_text(payload, "new_session_id"), idempotency_key=request_id)
            return _json({"session_id": fork.session_id}, 201)
        if action == "update":
            return _json(await session.update(
                expected_revision=payload["expected_revision"], metadata=payload["metadata"],
                cwd=payload.get("cwd"), idempotency_key=request_id,
            ))
        return _json({"error_code": "ACTION_NOT_FOUND"}, 404)

    def _turn_events(self, request: Request, identity: str, runtime: Runtime) -> Response:
        async def stream() -> AsyncGenerator[str, None]:
            if self._turn_subscribers >= _TURN_SSE_LIMIT:
                yield _sse_frame("session_error", {"error_code": "EXECUTION_CONCURRENCY_LIMIT_EXCEEDED"})
                return
            self._turn_subscribers += 1
            queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue(maxsize=_TURN_SSE_LIMIT)
            buffered = 0
            overflow = False
            unsubscribe = None

            def receive(value: Mapping[str, object]) -> None:
                nonlocal buffered, overflow
                if overflow:
                    return
                revision = value.get("revision")
                attachments = value.get("attachments")
                if isinstance(attachments, list):
                    value = {
                        **value,
                        "attachments": [
                            {key: item[key] for key in ("attachment_id", "source", "media_type", "size", "position", "path")
                             if key in item}
                            for item in attachments if isinstance(item, Mapping)
                        ],
                    }
                if value.get("type") == "terminal":
                    value = _ordinary_turn_wire(value)
                frame = _sse_frame("turn", value, str(revision) if isinstance(revision, int) else None)
                size = len(frame.encode("utf-8"))
                if queue.full() or buffered + size > _TURN_SSE_BUFFER:
                    overflow = True
                    while not queue.empty():
                        queue.get_nowait()
                    buffered = 0
                    queue.put_nowait(("", 0))
                else:
                    queue.put_nowait((frame, size))
                    buffered += size

            try:
                # Subscribe before taking the authorized snapshot so an
                # admission/commit racing the initial read cannot disappear.
                unsubscribe = runtime.sessions.observe_turn_session(identity, receive)
                snapshot = await self.require_history().turn_session_state(
                    identity, principal=self.principal, local_workspace=self.local_turn_workspace,
                )
                yield _sse_frame("turn", {"type": "snapshot", "session": snapshot}, str(snapshot["revision"]))
                while True:
                    try:
                        async with asyncio.timeout(15):
                            frame, size = await queue.get()
                    except TimeoutError:
                        yield ": heartbeat\n\n"
                        continue
                    buffered -= size
                    if overflow:
                        yield _sse_frame("turn", {"type": "gap"})
                        return
                    yield frame
            except AIError as error:
                _debug_error(request, error)
                yield _sse_frame("session_error", {"error_code": error.code.value})
            finally:
                if unsubscribe is not None:
                    unsubscribe()
                self._turn_subscribers -= 1

        return _SSEStream(stream(), self._shutdown_event)

    async def session_events(self, request: Request) -> Response:
        runtime = self.require_runtime()
        identity = _text(request.query_params, "session_id")
        if self.turn_sessions:
            return self._turn_events(request, identity, runtime)
        chat = request.query_params.get("view") == "chat"
        resume_id, resume_cursor = _session_execution_cursor(request.headers.get("last-event-id") if chat else None)

        async def stream() -> AsyncGenerator[str, None]:
            feed = self._session_notices
            tools: asyncio.Queue[tuple[str, ToolProcessItem]] = asyncio.Queue(maxsize=16)

            def collect(execution_id: str, item: ToolProcessItem) -> None:
                if not tools.full():
                    tools.put_nowait((execution_id, item))

            async def pump(execution_id: str, queue: asyncio.Queue[_ExecutionFrame]) -> None:
                task = asyncio.current_task()
                assert task is not None
                try:
                    if environ.debug:
                        request.state.phase = "execution.watch"
                    execution = await runtime.executions.get(execution_id, principal=self.principal)
                    watch = execution.watch(
                        cursor=resume_cursor if execution_id == resume_id else None,
                        include_content=True, include_descendants=False, include_model_interactions=False,
                    )
                    frames = self._execution_frames(request, execution_id, watch, chat=True)
                    async with aclosing(frames):
                        async for frame in frames:
                            if task.cancelling():
                                return
                            await queue.put(frame)
                except AIError as error:
                    if not task.cancelling():
                        await queue.put(self._execution_error_frame(request, error, chat=True))

            async def authorize_remote(execution_id: str) -> str | None:
                if feed is not None:
                    try:
                        await runtime.events.authorize_read(execution_id, principal=self.principal)
                    except AIError as error:
                        _debug_error(request, error)
                    else:
                        return execution_id
                return None

            unsubscribe = feed.subscribe(identity, collect) if feed is not None else None
            watch = runtime.sessions.watch(identity, principal=self.principal)
            state_task = asyncio.create_task(anext(watch, None))
            tool_task = asyncio.create_task(tools.get()) if feed is not None else None
            frame_task = producer = None
            execution_frames = None
            retiring: set[asyncio.Task] = set()
            authorized_execution = None
            last_active = last_local = watching = None
            ended_while_started = False
            occupancy: dict[str, object] | None = None
            publication = ExitStack()
            read_publication = None

            async def committed_turn_frame() -> str | None:
                nonlocal read_publication
                if read_publication is None:
                    return None
                captured = await read_publication()
                if captured is None:
                    return None
                read_publication = None
                publication.close()
                turn, = _chat_timeline(Page((captured.turn,), None)).items
                frame = _sse_frame("session_turn", {
                    "turn": _wire(turn), "revision": captured.session_revision,
                })
                return frame if len(frame.encode("utf-8")) <= _MAX_CHAT_TURN_FRAME_BYTES else None

            def finish_retiring(task: asyncio.Task) -> None:
                retiring.discard(task)
                if not task.cancelled():
                    task.exception()

            def stop_execution() -> None:
                nonlocal producer, frame_task, watching, read_publication
                publication.close()
                read_publication = None
                for task in (producer, frame_task):
                    if task is not None:
                        retiring.add(task)
                        task.add_done_callback(finish_retiring)
                        task.cancel()
                producer = frame_task = watching = None

            try:
                while True:
                    done, _ = await asyncio.wait(
                        tuple(task for task in (state_task, tool_task, frame_task, producer) if task is not None),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Completion doesn't invalidate a final frame already buffered by the pump.
                    if producer in done and not frame_task.done() and not execution_frames.empty():
                        await frame_task
                    if frame_task is not None and frame_task.done():
                        frame = frame_task.result()
                        # Session occupancy must not wait for optional observation to finish.
                        if not state_task.done() or frame[0] == "snapshot":
                            frame_task = asyncio.create_task(execution_frames.get())
                            event, data, cursor = frame
                            if event == "message":
                                # EventSource resumes from the SSE id, not the JSON body.
                                data.pop("cursor", None)
                            if event == "snapshot":
                                ended_while_started = isinstance(data, Mapping) and data.get("status") == "STARTED"
                                committed = await committed_turn_frame()
                                if committed is not None:
                                    yield committed
                            event_id = None if cursor is None else json.dumps((watching, cursor), separators=(",", ":"))
                            if event in ("text", "thinking"):
                                yield _sse_frame(event, data, event_id)
                            else:
                                yield _sse_frame("execution", {"execution_id": watching, "event": event, "data": data}, event_id)
                            continue
                    elif producer in done and not state_task.done():
                        producer.result()
                        frame_task.cancel()
                        await asyncio.gather(frame_task, return_exceptions=True)
                        frame_task = producer = None
                        if (ended_while_started and watching is not None and watching == last_active
                                and last_local and not runtime.events.has_local_producer(watching)):
                            authorized_execution = await authorize_remote(watching)
                            last_local = False
                            watching = None
                            if occupancy is not None and not state_task.done():
                                yield _sse_frame("message", {**occupancy, "local_execution": False})
                        continue
                    if not state_task.done():
                        execution_id, item = tool_task.result()
                        tool_task = asyncio.create_task(tools.get())
                        # A transport hint never grants access or identifies a session owner.
                        if execution_id == authorized_execution:
                            yield _sse_frame("session_process", {"execution_id": execution_id, "item": _wire(item)})
                        continue
                    session = state_task.result()
                    state_task = None
                    if session is None:
                        break
                    active = session.active_execution_id
                    local_execution = None
                    if active is not None and (chat or feed is not None):
                        local_execution = runtime.events.has_local_producer(active)
                        if (active, local_execution) != (last_active, last_local):
                            authorized_execution = None if local_execution else await authorize_remote(active)
                    else:
                        authorized_execution = None
                    last_active, last_local = active, local_execution
                    if watching is not None and (active != watching or not local_execution):
                        committed = await committed_turn_frame()
                        if committed is not None:
                            yield committed
                        stop_execution()
                    if chat and local_execution and active != watching:
                        read_publication = publication.enter_context(runtime.sessions.capture_published_turn(
                            identity, active, principal=self.principal,
                        ))
                        watching = active
                        ended_while_started = False
                        execution_frames = asyncio.Queue(maxsize=1)
                        producer = asyncio.create_task(pump(active, execution_frames))
                        frame_task = asyncio.create_task(execution_frames.get())
                    occupancy = {
                        "status": session.status.value,
                        "active_execution_id": active,
                        "revision": session.revision,
                    }
                    if active is not None and local_execution is not None:
                        occupancy["local_execution"] = local_execution
                    state_task = asyncio.create_task(anext(watch, None))
                    yield _sse_frame("message", occupancy)
            except AIError as error:
                _debug_error(request, error)
                yield _sse_frame("session_error", {"error_code": error.code.value})
            finally:
                publication.close()
                if unsubscribe is not None:
                    unsubscribe()
                tasks = tuple(task for task in (state_task, tool_task, frame_task, producer) if task is not None)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*retiring, *tasks, return_exceptions=True)
                await watch.aclose()

        return _SSEStream(stream(), self._shutdown_event)

    async def start_plan(self, request: Request) -> Response:
        runtime = self.require_runtime()
        payload = await _body(request)
        request_id = _text(payload, "request_id")
        request.state.operation_id = request_id
        if environ.debug:
            request.state.phase = "execution.plan"
        execution = await runtime.agents.get(_text(payload, "agent_id", "default")).start_plan(
            _text(payload, "prompt"), principal=self.principal, idempotency_key=request_id,
        )
        return _json({"execution_id": execution.execution_id}, 202)

    async def executions(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "execution.list"
        if request.query_params.get("recent", "false") not in {"true", "false"}:
            raise ValueError("recent must be true or false")
        if request.query_params.get("recent") == "true":
            if any(request.query_params.get(key) for key in ("cursor", "session_id", "agent_id", "parent_execution_id")):
                raise ValueError("recent scan does not support cursors or filters")
            items = () if self.history is None else await self.history.scan_recent_executions(
                principal=self.principal, limit=int(_paging(request)["limit"]),
            )
            return _json({"items": items, "next_cursor": None, "recent_scan": True})
        if self.history is None:
            return _json({"items": [], "next_cursor": None})
        return _json(await self.history.list_executions(ListExecutionRequest(
            self.principal, session_id=request.query_params.get("session_id"),
            agent_id=request.query_params.get("agent_id"), parent_execution_id=request.query_params.get("parent_execution_id"),
            **_paging(request),
        )))

    async def execution(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "execution.metadata"
        return _json(await self.require_history().inspect_execution(
            request.path_params["execution_id"], principal=self.principal,
        ))

    async def model_batch(self, request: Request) -> Response:
        identities = []
        for value in request.query_params.getlist("identity"):
            try:
                run, sequence = value.split(":")
                identities.append((int(run), int(sequence)))
            except ValueError:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from None
        items = await self.require_history().read_model_interactions(
            request.path_params["execution_id"], principal=self.principal,
            identities=tuple(identities), include_content=_content(request),
            include_client_requests=_content(request, "include_client_requests"),
        )
        return _json({"items": items})

    async def detail(self, request: Request) -> Response:
        history = self.require_history()
        identity = request.path_params["execution_id"]
        kind = request.path_params["kind"]
        if environ.debug:
            request.state.phase = f"execution.{kind}"
        paging = _paging(request)
        if kind == "trace":
            result = await history.trace(identity, principal=self.principal, **paging, **_selectors(request))
        elif kind == "history":
            result = await history.history(identity, principal=self.principal, **paging, include_content=_content(request), **_selectors(request, history=True))
        elif kind == "transcript":
            result = await history.transcript(identity, principal=self.principal, **paging, include_content=_content(request))
        elif kind == "models":
            raw = _content(request, "include_client_requests")
            result = await history.model_interactions(
                identity, principal=self.principal, **paging, include_content=_content(request),
                **({"include_client_requests": True} if raw else {}),
                **{key: int(request.query_params[key]) for key in ("agent_run_seq", "model_request_seq")
                   if key in request.query_params},
            )
        elif kind == "result":
            result = await history.result(identity, principal=self.principal)
        elif kind == "recovery":
            result = await self.require_runtime().executions.recovery_effects(identity, principal=self.principal)
        else:
            return _json({"error_code": "DETAIL_NOT_FOUND"}, 404)
        return _json(result)

    async def execution_action(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "execution.get"
        runtime = self.require_runtime()
        execution = await runtime.executions.get(request.path_params["execution_id"], principal=self.principal)
        payload = await _body(request)
        request_id = _text(payload, "request_id")
        action = request.path_params["action"]
        request.state.operation_id = request_id
        if environ.debug:
            request.state.phase = f"execution.{action}"
        if action == "cancel":
            return _json(await execution.cancel(idempotency_key=request_id))
        if action == "retry":
            result = await execution.retry(_text(payload, "prompt"), files=payload.get("files", ()), idempotency_key=request_id)
        elif action == "fork":
            result = await execution.fork(_text(payload, "prompt"), files=payload.get("files", ()), idempotency_key=request_id)
        elif action == "recover":
            result = await execution.recover(idempotency_key=request_id)
        elif action == "resolve":
            resolution = payload.get("resolution")
            if resolution == "applied":
                effect = ToolEffectApplied(payload.get("result"))
            elif resolution == "not_applied":
                effect = ToolEffectNotApplied()
            elif resolution == "failed":
                effect = ToolEffectFailed()
            else:
                raise ValueError("resolution must be applied, not_applied or failed")
            return _json(await execution.resolve_tool_effect(
                _text(payload, "operation_id"), expected_fence=payload["expected_fence"],
                resolution=effect, idempotency_key=request_id,
            ))
        else:
            return _json({"error_code": "ACTION_NOT_FOUND"}, 404)
        return _json({"execution_id": result.execution_id}, 202)

    async def events(self, request: Request) -> Response:
        if "view" in request.query_params:
            # The diagnostics stream must not satisfy a request for filtered Chat data.
            raise ValueError("select the session stream for the Chat view")
        if environ.debug:
            request.state.phase = "execution.watch"
        runtime = self.require_runtime()
        execution = await runtime.executions.get(
            request.path_params["execution_id"], principal=self.principal,
        )
        cursor = request.query_params.get("cursor") or request.headers.get("last-event-id") or None
        watch = execution.watch(
            cursor=cursor, include_content=True, include_descendants=True,
            include_model_interactions=True,
        )

        async def stream() -> AsyncGenerator[str, None]:
            frames = self._execution_frames(request, execution.execution_id, watch, chat=False)
            async with aclosing(frames):
                async for event, data, cursor in frames:
                    yield _sse_frame(event, data, cursor)

        return _SSEStream(stream(), self._shutdown_event)

    async def _execution_frames(
        self, request: Request, execution_id: str,
        watch: "AsyncIterator[ExecutionTreeEvent | ExecutionObservationEvent]",
        *, chat: bool,
    ) -> AsyncGenerator[_ExecutionFrame, None]:
        try:
            with local_execution_stream() if chat else nullcontext():
                async for observation in watch:
                    if chat:
                        frame = _chat_frame(observation)
                        if frame is not None:
                            yield frame
                    else:
                        item = observation.event
                        payload = {"type": "model" if isinstance(item, TaskModelProjection) else "event",
                                   "item": _wire(item), "cursor": observation.cursor}
                        yield "message", payload, observation.cursor
            # Completion of an observer is not the execution's terminal verdict.
            if environ.debug:
                request.state.phase = "execution.metadata"
            info = await self.require_history().inspect_execution(execution_id, principal=self.principal)
            if chat:
                info = replace(info, safe_error_details={}, error_diagnostics=None)
            yield "snapshot", _wire(info), None
        except AIError as error:
            yield self._execution_error_frame(request, error, chat=chat)
        finally:
            await watch.aclose()

    @staticmethod
    def _execution_error_frame(request: Request, error: AIError, *, chat: bool) -> _ExecutionFrame:
        _debug_error(request, error)
        details = error.safe_details
        if chat:
            details = {key: value for key, value in details.items() if key == "cleanup_pending"}
        payload = {"error_code": error.code.value, "safe_details": details}
        if isinstance(error, ObservationError):
            payload.update(cursor=error.cursor, origin=error.origin)
        return "observation_error", payload, None

    async def metric_query(self, request: Request) -> Response:
        if environ.debug:
            request.state.phase = "metrics.query"
        if self.metrics is None:
            return _json({"items": [], "unavailable": True})
        query = request.query_params
        start, end = query.get("start"), query.get("end")
        if bool(start) != bool(end):
            raise ValueError("start and end are required together")
        now = datetime.now(timezone.utc)
        window = MetricWindow.between(datetime.fromisoformat(start), datetime.fromisoformat(end)) if start else MetricWindow.between(now - timedelta(days=1), now)
        names = (query["metric"],) if query.get("metric") else _SUMMARY_METRICS
        values = []
        for name in names:
            values.append(await self.metrics.query(MetricQuery(
                name, window,
                aggregation=MetricAggregation(query["aggregation"]) if query.get("aggregation") else None,
                percentile=float(query["percentile"]) if query.get("percentile") else None,
                filters=json.loads(query.get("filters", "{}")),
                correlation_filters=json.loads(query.get("correlation_filters", "{}")),
                group_by=tuple(filter(None, query.get("group_by", "").split(","))),
                bucket=timedelta(seconds=int(query["bucket_seconds"])) if query.get("bucket_seconds") else None,
            )))
        return _json({"items": values})


def _operation_id(request: Request, error: AIError) -> str:
    return error.operation_id or request.scope.get("state", {}).get("operation_id") or request.path_params.get("execution_id", "web")


def _debug_error(request: Request, error: AIError) -> None:
    if environ.debug:
        state = request.scope.get("state", {})
        _logger.debug(
            "Web request failed: path=%s phase=%s code=%s operation_id=%s wire_type=%s missing_fields=%s",
            request.url.path, state.get("phase", "request"), error.code.value, _operation_id(request, error),
            error.safe_details.get("wire_type"), error.safe_details.get("missing_fields"),
        )


async def _error(request: Request, error: Exception) -> Response:
    if isinstance(error, AIError):
        _debug_error(request, error)
        code = error.code.value
        status = 404 if code in {"AUTHORIZATION_DENIED", "SESSION_NOT_FOUND", "EXECUTION_NOT_FOUND"} else (
            409 if code == "SESSION_TURN_PENDING" or any(part in code for part in ("CONFLICT", "BUSY", "CLOSED", "MISMATCH")) else (
                400 if error.code is not ErrorCode.STORAGE_RECOVERY_REQUIRED
                and any(part in code for part in ("INVALID", "REQUIRED")) else 503
            )
        )
        return _json(asdict(error.to_safe_error(operation_id=_operation_id(request, error))), status)
    if isinstance(error, (TypeError, ValueError, KeyError)):
        return _json({"error_code": "REQUEST_FIELD_INVALID", "message": "Check the request fields and paging parameters"}, 400)
    _logger.error("Web request failed: path=%s error_type=%s", request.url.path, type(error).__name__)
    return _json({"error_code": "INTERNAL_ERROR"}, 500)


async def _asset(request: Request) -> Response:
    name = request.path_params.get("name", "index.html")
    if name not in {"index.html", "app.js", "console.js", "message.js", "markdown-it.js", "style.css"}:
        return _json({"error_code": "NOT_FOUND"}, 404)
    return FileResponse(_ASSETS / name, headers=_HEADERS)


def create_app(
    *, runtime: Runtime | None = None, history: History | None = None,
    metrics: Metrics | None = None, status: Mapping[str, object] | None = None,
    capabilities: Sequence[Mapping[str, object]] = (),
    shutdown_event: asyncio.Event | None = None,
    session_notices: _SessionNoticeFeed | None = None,
    local_workspace: bool = False,
    turn_notice_transport: "TurnNoticeTransport | None" = None,
) -> "ASGIApp":
    """Build a loopback-only console; the caller owns and closes injected resources.

    Set shutdown_event before draining HTTP connections to close live observers
    without shutting down their Runtime. With no Runtime, mutations and live watches are disabled. Historical reads
    remain available without model credentials. Only an explicitly trusted local workspace can list ordinary sessions.
    The configured Conversation storage selects the implementation, not Runtime capabilities.
    An optional host-owned ordinary turn notice transport is attached during ASGI lifespan.
    Host/Origin headers are not
    restricted; a reverse proxy must enforce authentication and access control.
    The console itself does not authenticate remote users.
    """
    enabled = uses_turn_sessions(runtime) if runtime is not None else (
        history is not None and history.has_turn_sessions
    )
    lifespan = None
    if turn_notice_transport is not None:
        if runtime is None or not enabled:
            raise ValueError("turn_notice_transport requires a Runtime with ordinary turn sessions")
        bridge = runtime.sessions.bridge_turn_notices

        @asynccontextmanager
        async def turn_notice_lifespan(_app: Starlette) -> AsyncGenerator[None, None]:
            async with bridge(turn_notice_transport):
                yield

        lifespan = turn_notice_lifespan

    console = _Console(
        runtime, history, metrics, status or {}, capabilities, shutdown_event, session_notices, local_workspace, enabled,
    )
    app = Starlette(routes=[
        Route("/", _asset), Route("/assets/{name}", _asset),
        Route("/api/config", console.config),
        Route("/api/sessions", console.sessions, methods=["GET", "POST"]),
        Route("/api/session", console.session),
        Route("/api/session/events", console.session_events),
        Route("/api/session/{action}", console.session_action, methods=["POST"]),
        Route("/api/executions", console.executions),
        Route("/api/executions/plan", console.start_plan, methods=["POST"]),
        Route("/api/executions/{execution_id}", console.execution),
        Route("/api/executions/{execution_id}/events", console.events),
        Route("/api/executions/{execution_id}/models-exact", console.model_batch),
        Route("/api/executions/{execution_id}/{kind}", console.detail),
        Route("/api/executions/{execution_id}/{action}", console.execution_action, methods=["POST"]),
        Route("/api/metrics", console.metric_query),
    ], lifespan=lifespan, exception_handlers={AIError: _error, ValueError: _error, TypeError: _error, KeyError: _error, Exception: _error})
    return _LocalBoundary(app)
