#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-origin HTTP delivery; Runtime owns execution and durable state."""

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from linktools.core import environ
from ..core import Principal, PrincipalKind
from ..errors import AIError, ErrorCode, ErrorDiagnostics, ObservationError
from ..observe import MetricAggregation, MetricQuery, Metrics, MetricWindow
from ..runtime import (
    CreateSessionRequest,
    ListExecutionRequest, ListSessionRequest, Runtime, RuntimeHistory,
    TaskModelProjection, ToolEffectApplied, ToolEffectFailed, ToolEffectNotApplied,
)

if TYPE_CHECKING:
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


def _wire(value: object) -> object:
    if isinstance(value, ErrorDiagnostics):
        # Exception messages can contain provider credentials; safe details remain available.
        return {"exception_type": value.exception_type, "cause_digest": value.cause_digest}
    if is_dataclass(value) and not isinstance(value, type):
        # Recurse before asdict so diagnostic messages never enter an HTTP response.
        return {field.name: _wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire(item) for item in value]
    return value


def _json(value: object, status: int = 200) -> JSONResponse:
    return JSONResponse(_wire(value), status_code=status, headers=_HEADERS)


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


def _content(request: Request) -> bool:
    value = request.query_params.get("include_content", "false")
    if value not in {"true", "false"}:
        raise ValueError("include_content must be true or false")
    return value == "true"


async def _body(request: Request) -> dict[str, object]:
    if request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
        raise ValueError("application/json is required")
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("request exceeds 8 MiB")
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("request must be an object")
    return payload


class _LocalBoundary:
    def __init__(self, app: "ASGIApp", *, port: int) -> None:
        self.app = app
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if port == 80:
            self.hosts.update({"127.0.0.1", "localhost"})

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        peer = scope.get("client")
        if peer is None or peer[0] not in {"127.0.0.1", "::1"}:
            await _json({"error_code": "LOOPBACK_REQUIRED"}, 403)(scope, receive, send)
            return
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if host not in self.hosts or (origin is not None and origin != f"http://{host.removesuffix(':80')}"):
            await _json({"error_code": "LOCAL_ORIGIN_REQUIRED"}, 403)(scope, receive, send)
            return
        if request.headers.get("sec-fetch-site") == "cross-site":
            await _json({"error_code": "LOCAL_ORIGIN_REQUIRED"}, 403)(scope, receive, send)
            return
        if request.method not in {"GET", "HEAD"} and request.headers.get("x-linktools-console") != "1":
            await _json({"error_code": "CONSOLE_REQUEST_REQUIRED"}, 403)(scope, receive, send)
            return
        await self.app(scope, receive, send)


class _Console:
    def __init__(
        self, runtime: Runtime | None, history: RuntimeHistory | None, metrics: Metrics | None,
        status: Mapping[str, object], capabilities: Sequence[Mapping[str, object]], memory_scope: str,
    ) -> None:
        self.runtime = runtime
        self.history = runtime.history if runtime is not None else history
        self.metrics = metrics
        self.status = dict(status)
        self.capabilities = tuple(capabilities)
        self.memory_scope = memory_scope
        self.principal = runtime.default_principal if runtime is not None else Principal(
            "runtime", "default" if history is None else history.tenant_id, PrincipalKind.LOCAL_TRUSTED.value,
        )

    def require_runtime(self) -> Runtime:
        if self.runtime is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, safe_details={"read_only": True})
        return self.runtime

    def require_history(self) -> RuntimeHistory:
        if self.history is None:
            raise AIError(ErrorCode.EXECUTION_NOT_FOUND)
        return self.history

    async def config(self, request: Request) -> Response:
        return _json({**self.status, "read_only": self.runtime is None, "memory_scope": self.memory_scope,
                      "capabilities": self.capabilities, "metric_names": _SUMMARY_METRICS})

    async def sessions(self, request: Request) -> Response:
        if request.method == "POST":
            runtime = self.require_runtime()
            payload = await _body(request)
            return _json(await runtime.sessions.create(_text(payload, "agent_id", "default"), CreateSessionRequest(
                self.principal, _text(payload, "session_id"), _text(payload, "request_id"),
                cwd=payload.get("cwd"), metadata={"title": _text(payload, "title", "New conversation")},
            )), 201)
        if self.runtime is not None:
            return _json(await self.runtime.sessions.list(ListSessionRequest(self.principal, **_paging(request))))
        # RuntimeHistory currently offers recent sessions, not a read-only paged session index.
        if request.query_params.get("cursor"):
            raise ValueError("read-only recent sessions have no cursor")
        items = () if self.history is None else await self.history.recent_sessions(
            principal=self.principal, limit=int(_paging(request)["limit"]),
        )
        return _json({"items": items, "next_cursor": None, "recent_only": True})

    async def session(self, request: Request) -> Response:
        history = self.require_history()
        identity = _text(request.query_params, "session_id")
        return _json({"session": await history.inspect_session(identity, principal=self.principal),
                      "timeline": await history.session_timeline(identity, principal=self.principal, **_paging(request))})

    async def session_action(self, request: Request) -> Response:
        runtime = self.require_runtime()
        session = await runtime.sessions.get(_text(request.query_params, "session_id"), principal=self.principal)
        payload = await _body(request)
        request_id = _text(payload, "request_id")
        action = request.path_params["action"]
        if action == "messages":
            execution = await session.start(
                _text(payload, "prompt"), files=payload.get("files", ()), idempotency_key=request_id,
                memory_scope=_text(payload, "memory_scope", self.memory_scope),
                planning=payload.get("planning", False), thinking=payload.get("thinking", False),
            )
            return _json({"execution_id": execution.execution_id}, 202)
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

    async def executions(self, request: Request) -> Response:
        if self.history is None:
            return _json({"items": [], "next_cursor": None})
        return _json(await self.history.list_executions(ListExecutionRequest(
            self.principal, session_id=request.query_params.get("session_id"),
            agent_id=request.query_params.get("agent_id"), parent_execution_id=request.query_params.get("parent_execution_id"),
            **_paging(request),
        )))

    async def execution(self, request: Request) -> Response:
        return _json(await self.require_history().inspect_execution(
            request.path_params["execution_id"], principal=self.principal,
        ))

    async def detail(self, request: Request) -> Response:
        history = self.require_history()
        identity = request.path_params["execution_id"]
        kind = request.path_params["kind"]
        paging = _paging(request)
        if kind == "trace":
            result = await history.trace(identity, principal=self.principal, **paging, **_selectors(request))
        elif kind == "history":
            result = await history.history(identity, principal=self.principal, **paging, include_content=_content(request), **_selectors(request, history=True))
        elif kind == "transcript":
            result = await history.transcript(identity, principal=self.principal, **paging, include_content=_content(request))
        elif kind == "models":
            result = await history.model_interactions(identity, principal=self.principal, **paging, include_content=_content(request))
        elif kind == "result":
            result = await history.result(identity, principal=self.principal)
        elif kind == "recovery":
            result = await self.require_runtime().executions.recovery_effects(identity, principal=self.principal)
        else:
            return _json({"error_code": "DETAIL_NOT_FOUND"}, 404)
        return _json(result)

    async def execution_action(self, request: Request) -> Response:
        runtime = self.require_runtime()
        execution = await runtime.executions.get(request.path_params["execution_id"], principal=self.principal)
        payload = await _body(request)
        request_id = _text(payload, "request_id")
        action = request.path_params["action"]
        if action == "cancel":
            return _json(await execution.cancel(idempotency_key=request_id))
        if action == "retry":
            result = await execution.retry(_text(payload, "prompt"), files=payload.get("files", ()), idempotency_key=request_id)
        elif action == "fork":
            result = await execution.fork(_text(payload, "prompt"), files=payload.get("files", ()), idempotency_key=request_id)
        elif action == "recover":
            result = await execution.recover()
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
        execution = await self.require_runtime().executions.get(
            request.path_params["execution_id"], principal=self.principal,
        )
        cursor = request.query_params.get("cursor") or request.headers.get("last-event-id") or None
        watch = execution.watch(cursor=cursor, include_content=True, include_model_interactions=True)

        async def stream() -> AsyncIterator[str]:
            try:
                async for observation in watch:
                    item = observation.event
                    payload = {"type": "model" if isinstance(item, TaskModelProjection) else "event",
                               "item": _wire(item), "cursor": observation.cursor}
                    prefix = "" if observation.cursor is None else f"id: {observation.cursor}\n"
                    yield prefix + "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                # Completion of an observer is not the execution's terminal verdict.
                info = await self.require_history().inspect_execution(execution.execution_id, principal=self.principal)
                yield "event: snapshot\ndata: " + json.dumps(_wire(info), ensure_ascii=False) + "\n\n"
            except AIError as error:
                payload = {"error_code": error.code.value, "safe_details": error.safe_details}
                if isinstance(error, ObservationError):
                    payload.update(cursor=error.cursor, origin=error.origin)
                yield "event: observation_error\ndata: " + json.dumps(payload) + "\n\n"
            finally:
                await watch.aclose()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={**_HEADERS, "X-Accel-Buffering": "no"})

    async def metric_query(self, request: Request) -> Response:
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


async def _error(request: Request, error: Exception) -> Response:
    if isinstance(error, AIError):
        code = error.code.value
        status = 404 if code in {"AUTHORIZATION_DENIED", "SESSION_NOT_FOUND", "EXECUTION_NOT_FOUND"} else (
            409 if any(part in code for part in ("CONFLICT", "BUSY", "CLOSED", "MISMATCH")) else (
                400 if any(part in code for part in ("INVALID", "REQUIRED")) else 503
            )
        )
        return _json(asdict(error.to_safe_error(operation_id=request.path_params.get("execution_id", "web"))), status)
    if isinstance(error, (TypeError, ValueError, KeyError)):
        return _json({"error_code": "REQUEST_FIELD_INVALID", "message": "Check the request fields and paging parameters"}, 400)
    _logger.error("Web request failed: path=%s error_type=%s", request.url.path, type(error).__name__)
    return _json({"error_code": "INTERNAL_ERROR"}, 500)


async def _asset(request: Request) -> Response:
    name = request.path_params.get("name", "index.html")
    if name not in {"index.html", "app.js", "console.js", "style.css"}:
        return _json({"error_code": "NOT_FOUND"}, 404)
    return FileResponse(_ASSETS / name, headers=_HEADERS)


def create_app(
    *, runtime: Runtime | None = None, history: RuntimeHistory | None = None,
    metrics: Metrics | None = None, status: Mapping[str, object] | None = None,
    capabilities: Sequence[Mapping[str, object]] = (), memory_scope: str = "default", port: int = 8765,
) -> "ASGIApp":
    """Build a loopback-only console; the caller owns and closes injected resources.

    With no Runtime, mutations and live watches are disabled. Historical reads
    remain available without model credentials. Never mount this local-trust app
    as a remotely authenticated service.
    """
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    console = _Console(runtime, history, metrics, status or {}, capabilities, memory_scope)
    app = Starlette(routes=[
        Route("/", _asset), Route("/assets/{name}", _asset),
        Route("/api/config", console.config),
        Route("/api/sessions", console.sessions, methods=["GET", "POST"]),
        Route("/api/session", console.session),
        Route("/api/session/{action}", console.session_action, methods=["POST"]),
        Route("/api/executions", console.executions),
        Route("/api/executions/{execution_id}", console.execution),
        Route("/api/executions/{execution_id}/events", console.events),
        Route("/api/executions/{execution_id}/{kind}", console.detail),
        Route("/api/executions/{execution_id}/{action}", console.execution_action, methods=["POST"]),
        Route("/api/metrics", console.metric_query),
    ], exception_handlers={AIError: _error, ValueError: _error, TypeError: _error, KeyError: _error, Exception: _error})
    return _LocalBoundary(app, port=port)
