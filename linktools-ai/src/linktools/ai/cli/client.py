#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The single backend entry point for the ``lt ai`` console.

The thin console commands operate Runtime + Storage exclusively through
:class:`RuntimeClient`. :class:`LocalRuntimeClient` is the in-process
implementation that owns the project bundle (Runtime + Storage + registries).

:meth:`LocalRuntimeClient.run_stream` streams the engine's live events (model
text deltas, tool calls) as they are produced, via a queue-backed
:class:`StreamingRunLiveSink` wired into the Runtime by
:func:`build_runtime_client`. For bundles built without a sink (some tests),
it falls back to running the scalar ``Runtime.run`` then replaying the recorded
trace. Either path yields the same ``text``/``tool``/``paused``/``failed``/
``cancelled`` event contract the consumers render.

Local vs. Remote share the same interface; a remote (HTTP) client is not wired
up in this build and ``build_runtime_client(remote=...)`` fails explicitly
rather than pretending to support it (no fake implementation)."""

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable
import asyncio
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from linktools.cli import CommandError
from .runtime import build_cli_runtime, load_agent_spec

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ..execution.query import ExecutionDetailView
    from ..governance.identity import PrincipalContext
    from .runtime import CliRuntimeBundle
    from collections.abc import AsyncIterator


def trusted_local_principal(*, tenant_id: str = "local") -> "PrincipalContext":
    """Build the principal a local CLI acts as. Imported lazily so this
    module imports cleanly without the governance package at collection time."""
    from ..governance.identity import trusted_local_principal as _principal

    return _principal(tenant_id=tenant_id)


@dataclass(slots=True)
class _ApprovalView:
    """Approval-request detail rendered by the console."""

    approval_id: str
    run_id: str
    tool_name: str
    arguments: "Mapping[str, Any]"
    reason: "str | None"


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


__all__ = [
    "RunRequest",
    "DoctorCheck",
    "DoctorReport",
    "RuntimeClient",
    "LocalRuntimeClient",
    "build_runtime_client",
    "new_run_id",
    "validate_session_id",
]


# --------------------------------------------------------------------------- #
# Request / report dataclasses
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunRequest:
    """One run to start. ``run_id`` is caller-minted (see :func:`new_run_id`) so
    the caller can cancel the exact run it started."""

    prompt: str
    session_id: str = "main"
    agent_id: "str | None" = None
    run_id: "str | None" = None


@dataclass(slots=True)
class DoctorCheck:
    """One doctor verdict."""

    label: str
    ok: bool
    detail: "str | None" = None


@dataclass(slots=True)
class DoctorReport:
    """The aggregate doctor result. ``failed`` drives the non-zero exit code."""

    checks: "list[DoctorCheck]" = field(default_factory=list)

    @property
    def failed(self) -> "list[DoctorCheck]":
        return [c for c in self.checks if not c.ok]


# --------------------------------------------------------------------------- #
# Storage / run helpers (moved out of the command layer)
# --------------------------------------------------------------------------- #


def new_run_id() -> str:
    """Mint a run id the caller owns so it can cancel/resume a run by id.

    The runtime accepts a caller-supplied ``run_id`` everywhere it accepts a
    prompt; minting it in the caller (instead of letting the runtime generate
    one internally) lets the console cancel an in-flight run on Ctrl+C."""
    return str(uuid.uuid4())


def validate_session_id(session_id: str) -> str:
    """Reject session ids that could escape the sessions directory when joined
    into a filesystem path. Raises CommandError with a clear message; returns
    session_id unchanged if it's safe."""
    is_unsafe = (
        not session_id
        or "/" in session_id
        or "\\" in session_id
        or session_id in (".", "..")
    )
    if is_unsafe:
        raise CommandError(
            f'invalid session id "{session_id}": must not contain path separators or ".."'
        )
    return session_id


# --------------------------------------------------------------------------- #
# RuntimeClient protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class RuntimeClient(Protocol):
    """The backend surface used by the console commands."""

    async def run_stream(
        self, request: "RunRequest"
    ) -> "AsyncIterator[Mapping[str, Any]]": ...

    async def cancel(self, run_id: str) -> None: ...

    async def get_approval(self, approval_id: str) -> Any: ...

    async def doctor(self) -> DoctorReport: ...


# --------------------------------------------------------------------------- #
# LocalRuntimeClient
# --------------------------------------------------------------------------- #


class LocalRuntimeClient:
    """In-process ``RuntimeClient`` over a project bundle.

    Owns the Runtime + Storage + registries so the console does not need to
    know how ``build_runtime`` is wired. Operates the backend as the
    local user principal. ``run_stream`` streams the engine's live events
    (model text deltas, tool calls) as they are produced via a queue-backed
    :class:`StreamingRunLiveSink` wired into the Runtime; the recorded trace is
    only consulted to classify the terminal outcome (paused/cancelled/failed)
    after the live stream ends."""

    def __init__(self, bundle: "CliRuntimeBundle") -> None:
        self._bundle = bundle
        self._principal = trusted_local_principal()
        # The live-event sink the Runtime publishes into. Set when the bundle
        # was built with one (build_runtime_client wires a StreamingRunLiveSink);
        # None for bundles built without (e.g. tests constructing a bundle
        # directly), in which case run_stream falls back to trace replay.
        self._live = getattr(bundle, "live_events", None)
        self._pending_approvals: "dict[str, tuple[str, str]]" = {}

    @property
    def bundle(self) -> "CliRuntimeBundle":
        return self._bundle

    @property
    def principal(self) -> "PrincipalContext":
        return self._principal

    async def run_stream(
        self, request: "RunRequest"
    ) -> "AsyncIterator[Mapping[str, Any]]":
        spec = await load_agent_spec(self._bundle, request.agent_id)
        session_id = validate_session_id(request.session_id)
        run_id = request.run_id or new_run_id()

        async def _drive() -> "object":
            return await self._bundle.runtime.run(
                spec,
                request.prompt,
                principal=self._principal,
                session_id=session_id,
                execution_id=run_id,
            )

        async for event in self._stream_run(_drive, run_id, session_id):
            yield event

    async def _stream_run(
        self,
        drive: "Callable[[], Awaitable[object]]",
        run_id: str,
        session_id: str,
    ) -> "AsyncIterator[Mapping[str, Any]]":
        """Drive one run and yield its events.

        When a live sink is wired (build_runtime_client always wires one),
        the engine publishes text/tool events as they happen; consume them
        live from the queue while the run task runs, then classify the
        terminal outcome. With no sink (a bundle built directly, e.g. some
        tests), fall back to the scalar-run-then-replay-trace path so the
        consumer still sees the same events, just not streamed live."""
        if self._live is None:
            async for event in self._replay_stream(drive, run_id, session_id):
                yield event
            return
        sink = self._live
        queue = sink.attach(run_id)
        task = asyncio.ensure_future(drive())
        run_exc: "BaseException | None" = None
        try:
            # Consume live events as the engine publishes them. The engine does
            # not signal end-of-stream itself, so race each queue get against
            # the run task: when the task finishes, drain any remaining queued
            # events then stop.
            while True:
                get_task = asyncio.ensure_future(queue.get())
                done, _pending = await asyncio.wait(
                    {get_task, task}, return_when=asyncio.FIRST_COMPLETED
                )
                if get_task in done:
                    yield get_task.result()
                if task in done:
                    # Run finished: drain anything the engine published after
                    # the last event we yielded, then stop consuming.
                    get_task.cancel()
                    while not queue.empty():
                        yield queue.get_nowait()
                    break
                # Only the get completed (event yielded) and the run is still
                # going: loop to await the next event.
            await task
        except asyncio.CancelledError:
            task.cancel()
            run_exc = asyncio.CancelledError()
            raise
        except BaseException as exc:
            run_exc = exc
            task.cancel()
        finally:
            sink.detach(run_id)
        if run_exc is not None:
            yield {
                "type": "failed",
                "error_type": type(run_exc).__name__,
                "message": str(run_exc),
            }
            return
        try:
            result = task.result()
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed event
            yield {
                "type": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            return
        # result is an ExecutionResultView for COMPLETED, or None for PAUSED/
        # CANCELLED. The live stream already carried the model text/tools; we
        # only emit the terminal classification here.
        if result is None:
            detail = await self._get_run_detail(run_id)
            if detail is not None and detail.status == "paused":
                yield self._paused_event(run_id, detail)
            else:
                yield {"type": "cancelled", "run_id": run_id}
            return
        yield {"type": "completed", "run_id": run_id}

    async def _replay_stream(
        self,
        drive: "Callable[[], Awaitable[object]]",
        run_id: str,
        session_id: str,
    ) -> "AsyncIterator[Mapping[str, Any]]":
        """No-sink fallback: run to completion, then replay the recorded trace."""
        try:
            result = await drive()
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed event
            yield {
                "type": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            return
        if result is None:
            detail = await self._get_run_detail(run_id)
            if detail is not None and detail.status == "paused":
                yield self._paused_event(run_id, detail)
            else:
                yield {"type": "cancelled", "run_id": run_id}
            return
        async for event in self._trace_events(run_id):
            yield event

    async def cancel(self, run_id: str) -> None:
        await self._bundle.runtime.cancel(run_id, principal=self._principal)
        self._pending_approvals.pop(run_id, None)

    async def get_approval(self, approval_id: str) -> Any:
        """Read one approval request by id for the console."""
        run_id = self._run_id_for_approval(approval_id)
        if run_id is None:
            return None
        detail = await self._get_run_detail(run_id)
        if detail is None:
            return None
        tool = detail.tool_calls[-1] if detail.tool_calls else None
        return _ApprovalView(
            approval_id=approval_id,
            run_id=run_id,
            tool_name=tool.tool_name if tool else "",
            arguments=tool.arguments if tool else {},
            reason=None,
        )

    async def doctor(self) -> DoctorReport:
        """Run every project/Runtime check against the bundle and return the
        structured verdict; the console only renders it."""
        from ..agent.mcp.env import expand_env_mapping
        from ..agent.skill.private import resolve_skill_agent_path

        bundle = self._bundle
        project = bundle.project
        report = DoctorReport()

        def ok(label: str) -> None:
            report.checks.append(DoctorCheck(label=label, ok=True))

        def fail(label: str, detail: str) -> None:
            report.checks.append(DoctorCheck(label=label, ok=False, detail=detail))

        ok("project config")
        ok(f"default agent: {project.default_agent}")

        # Agents parse.
        agent_ids = await bundle.agents.list_ids()
        if project.default_agent not in agent_ids:
            fail("default agent", f"{project.default_agent!r} not in agents")
        for agent_id in agent_ids:
            try:
                await bundle.agents.get(agent_id)
                ok(f"agent: {agent_id}")
            except Exception as exc:
                fail(f"agent: {agent_id}", str(exc))

        # Skills + skill-private agents (path safety on each agents/*.md).
        for skill_id in await bundle.skill_index.list_ids():
            try:
                info = await bundle.skill_index.get(skill_id)
                ok(f"skill: {skill_id}")
            except Exception as exc:
                fail(f"skill: {skill_id}", str(exc))
                continue
            if info is None:
                continue
            for agent_path in info.list_private_agents():
                rel = agent_path.relative_to(info.root)
                try:
                    resolve_skill_agent_path(
                        skill_root=info.root, instruction_path=str(rel)
                    )
                    ok(f"skill agent: {skill_id}/{rel}")
                except Exception as exc:
                    fail(f"skill agent: {skill_id}/{rel}", str(exc))

        # MCP env expansion (fail-on-missing).
        for mcp_id in await bundle.mcp.list_ids():
            try:
                spec = await bundle.mcp.get(mcp_id)
                expand_env_mapping(spec.env)
                ok(f"MCP: {mcp_id}")
            except Exception as exc:
                fail(f"MCP: {mcp_id}", str(exc))

        # Runtime inspects cleanly for the default agent (feature validation;
        # the facade's inspect() now keys on a run_id, not a bare spec).
        try:
            default_spec = await bundle.agents.get(project.default_agent)
            await bundle.assembler.validate_features(default_spec)
            ok("runtime inspect")
        except Exception as exc:
            fail("runtime inspect", str(exc))

        # Storage writable.
        try:
            project.state_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=project.state_root):
                pass
            ok("storage writable")
        except Exception as exc:
            fail("storage writable", str(exc))

        try:
            from ..acp.errors import require_sdk

            require_sdk()
            ok("ACP SDK exact version")
            ok("ACP schema baseline")
        except Exception as exc:
            fail("ACP SDK exact version", str(exc))
        try:
            from ..acp.persistence import AcpSessionRepository
            from ..acp.process_lock import ProjectProcessLock

            repository = AcpSessionRepository(project.state_root)
            repository.root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=repository.root):
                pass
            lock = ProjectProcessLock(project.state_root / "acp" / "doctor.lock")
            lock.acquire(project_root=project.root)
            lock.release()
            ok("ACP process lock availability")
            ok("ACP session metadata writable")
        except Exception as exc:
            fail("ACP process lock availability", str(exc))
        try:
            from ..execution.live_events import AssistantTextDelta, ExecutionCompleted, ExecutionEventHub

            async def hub_check() -> None:
                hub = ExecutionEventHub()
                subscription = await hub.subscribe("doctor")
                await hub.publish("doctor", AssistantTextDelta(execution_id="doctor", text="ok"))
                await hub.close("doctor", ExecutionCompleted(execution_id="doctor"))
                if (await subscription.__anext__()).execution_id != "doctor":
                    raise RuntimeError("event hub identity mismatch")

            await hub_check()
            ok("execution event hub")
        except Exception as exc:
            fail("execution event hub", str(exc))

        return report

    # -- run detail ------------------------------------------------------- #

    async def _get_run_detail(self, run_id: str) -> "ExecutionDetailView | None":
        """The recorded detail for one run (status, interactions, tool calls,
        final output, usage) via the Runtime facade. None when the run is not
        visible to this principal or does not exist."""
        from ..execution.query import ExecutionDetailView

        try:
            detail = await self._bundle.runtime.inspect(
                run_id=run_id, principal=self._principal
            )
        except Exception:
            return None
        return detail if isinstance(detail, ExecutionDetailView) else None

    # -- internals -------------------------------------------------------- #

    async def _trace_events(self, run_id: str) -> "AsyncIterator[Mapping[str, Any]]":
        """Re-emit a completed run's recorded trace as stream events.

        Walks the model interactions in sequence order, yielding a ``text``
        event per model text part and a ``tool`` event (start+end) per tool
        call, so the consumer renders the real trajectory rather than only the
        final scalar output. ``final_output`` is emitted only as a fallback
        when no interaction carried a text part -- otherwise the model's text
        is already rendered and re-emitting the scalar output would duplicate
        it."""
        detail = await self._get_run_detail(run_id)
        if detail is None:
            return
        emitted_calls: "set[str]" = set()
        emitted_text = False
        for interaction in detail.interactions:
            response = interaction.response or {}
            for part in response.get("parts", ()):
                if not isinstance(part, Mapping):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    content = part.get("content") or part.get("text") or ""
                    if content:
                        emitted_text = True
                        yield {"type": "text", "text": str(content)}
                elif ptype == "tool_call":
                    call_id = str(part.get("call_id") or "")
                    name = str(part.get("tool_name") or "?")
                    if call_id and call_id not in emitted_calls:
                        emitted_calls.add(call_id)
                        yield {
                            "type": "tool",
                            "id": call_id,
                            "tool_call_id": call_id,
                            "name": name,
                            "phase": "start",
                        }
        # Tool results (end phase + ok/detail) come from the tool_calls view.
        for call in detail.tool_calls:
            ok = call.status in {"ok", "success", "completed"}
            yield {
                "type": "tool",
                "id": call.call_id,
                "tool_call_id": call.call_id,
                "name": call.tool_name,
                "phase": "end",
                "ok": ok,
                "detail": _truncate(str(call.result))
                if call.result is not None
                else None,
            }
        if (
            not emitted_text
            and detail.final_output is not None
            and str(detail.final_output).strip()
        ):
            yield {"type": "text", "text": str(detail.final_output)}

    def _paused_event(
        self, run_id: str, detail: "ExecutionDetailView"
    ) -> "Mapping[str, Any]":
        tool = detail.tool_calls[-1] if detail.tool_calls else None
        approval_id = f"{run_id}-approval"
        self._pending_approvals[approval_id] = (run_id, tool.tool_name if tool else "")
        return {
            "type": "paused",
            "run_id": run_id,
            "approval_id": approval_id,
            "tool_name": tool.tool_name if tool else "",
        }

    def _run_id_for_approval(self, approval_id: "str | None") -> "str | None":
        if not approval_id:
            return None
        entry = self._pending_approvals.get(approval_id)
        return entry[0] if entry else None

# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def build_runtime_client(
    *,
    remote: "str | None" = None,
    model: "str | None" = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    with_model: bool = True,
    project: "str | Path | None" = None,
) -> "RuntimeClient":
    """Build the backend client for the current project.

    ``with_model=False`` builds the bundle without registering a model so
    ``doctor`` works in a freshly-initialized project with no API key.
    ``project`` overrides where project discovery starts (the
    ``--project`` flag); the default is cwd. When ``with_model=True``, the
    caller passes already-resolved base_url/model/api_key (typically from
    ConfigAction at parse time). A non-None ``remote`` fails explicitly
    (HttpRuntimeClient is deferred)."""
    if remote is not None:
        raise CommandError("remote Runtime client is not supported in this build")
    from linktools.core import environ

    from .project import load_project

    start: "Path | None" = Path(project) if project else None
    cli_project = load_project(data_root=environ.get_data_path("ai"), start=start)
    if with_model:
        from ..model.registry import RuntimeModelConfig, model_registry
        from ..model.resolver import ModelResolver

        config = RuntimeModelConfig(
            model_type="standard",
            protocol="openai",
            model=model or "",
            base_url=base_url,
            api_key=api_key,
            auth_token=None,
            timeout_seconds=300,
            raw={},
        )
        model_registry.register(config.model_type, config=config)
        resolver: "object | None" = ModelResolver(registry=model_registry)
    else:
        resolver = None
    # A queue-backed live sink so run_stream streams model text/tool events as
    # they happen, instead of waiting for the run to finish.
    from ..execution.live_events import StreamingRunLiveSink

    live = StreamingRunLiveSink()
    bundle = build_cli_runtime(
        project=cli_project, model_resolver=resolver, live_events=live
    )
    return LocalRuntimeClient(bundle)
