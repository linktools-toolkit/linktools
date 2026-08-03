#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The single backend entry point for the ``lt ai`` console and TUI.

Both the thin console commands and the Textual TUI operate the Runtime + Storage
exclusively through :class:`RuntimeClient`. :class:`LocalRuntimeClient` is the
in-process implementation that owns the project bundle (Runtime + Storage +
registries); the TUI never touches persistence implementations or registries
directly. :class:`FakeRuntimeClient` is the shared double both console and TUI
tests drive.

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
    from ..runtime import Runtime, RuntimeStorage
    from .runtime import CliRuntimeBundle
    from ..execution.domain import RunRecord
    from ..execution.session import SessionRecord
    from collections.abc import AsyncIterator


def trusted_local_principal(*, tenant_id: str = "local") -> "PrincipalContext":
    """Build the principal a local CLI/TUI acts as. Imported lazily so this
    module imports cleanly without the governance package at collection time."""
    from ..governance.identity import trusted_local_principal as _principal

    return _principal(tenant_id=tenant_id)


@dataclass(slots=True)
class _SessionSummary:
    """A lightweight session record for the sidebar (the store has no
    enumeration API, so the client tracks what it created)."""

    id: str
    tenant_id: "str | None" = None

    @property
    def status(self) -> "_Status":
        return _Status()


@dataclass(slots=True)
class _RunSummary:
    """A lightweight run record for the sidebar."""

    id: str
    session_id: str
    tenant_id: "str | None" = None
    status: str = "completed"


@dataclass(slots=True)
class _Status:
    value: str = "active"


@dataclass(slots=True)
class _ApprovalView:
    """Approval-request detail rendered by the approval modal."""

    approval_id: str
    run_id: str
    tool_name: str
    arguments: "Mapping[str, Any]"
    reason: "str | None"


@dataclass(slots=True)
class _InspectionView:
    """Agent-feature summary rendered by the context panel."""

    id: str
    tools: "tuple[str, ...]"
    skills: "tuple[str, ...]"
    mcp_servers: "tuple[str, ...]"
    extensions: "tuple[str, ...]"
    features: "tuple[str, ...]"


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


__all__ = [
    "RunRequest",
    "DoctorCheck",
    "DoctorReport",
    "RuntimeClient",
    "LocalRuntimeClient",
    "FakeRuntimeClient",
    "build_runtime_client",
    "new_run_id",
    "validate_session_id",
    "ensure_session",
    "resolve_approval",
    "list_sessions",
    "list_runs",
    "list_pending_approvals",
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
    one internally) is what lets the console cancel an in-flight run on Ctrl+C
    and what ``continue`` later addresses."""
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


async def ensure_session(storage: "RuntimeStorage", session_id: str) -> None:
    """Get-or-create a session record.

    ``Runtime.run`` requires a pre-existing session when a ``session_id`` is
    supplied (it does not auto-create one). This mirrors the ``session_id=None``
    branch exactly by creating the ``SessionRecord`` up-front when the id is
    unseen."""
    if await storage.execution.get_session(session_id) is None:
        await storage.execution.create_session(
            session_id=session_id, user_id=None, tenant_id=None
        )


async def resolve_approval(
    runtime: "Runtime",
    execution_id: str,
    approval_id: str,
    *,
    approved: bool,
    principal: "PrincipalContext",
) -> "RunRecord":
    """Resolve a pending approval through the Runtime facade.

    ``ALLOW`` leaves the run PAUSED (resumable via ``resume_stream``); ``DENY``
    is terminal. The facade's ``decide_approval`` owns the state transition."""
    from ..execution.domain import ApprovalDecision

    decision = ApprovalDecision.ALLOW if approved else ApprovalDecision.DENY
    return await runtime.decide_approval(
        execution_id,
        approval_id=approval_id,
        decision=decision,
        principal=principal,
    )


async def list_sessions(_storage: "RuntimeStorage") -> list:
    """The execution store exposes no session-enumeration API; the client
    tracks the sessions it created in-process (see LocalRuntimeClient)."""
    return []


async def list_runs(_storage: "RuntimeStorage") -> list:
    """The execution store exposes no run-enumeration API; the client tracks
    the runs it started in-process (see LocalRuntimeClient)."""
    return []


async def list_pending_approvals(_storage: "RuntimeStorage") -> list:
    """No enumeration of pending approvals on the store; the client surfaces
    approvals via the streamed pause event instead."""
    return []


# --------------------------------------------------------------------------- #
# RuntimeClient protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class RuntimeClient(Protocol):
    """The only backend surface the console/TUI may use."""

    async def run_stream(
        self, request: "RunRequest"
    ) -> "AsyncIterator[Mapping[str, Any]]": ...

    async def resume_stream(
        self, run_id: str
    ) -> "AsyncIterator[Mapping[str, Any]]": ...

    async def cancel(self, run_id: str) -> None: ...

    async def approve(self, approval_id: str) -> None: ...

    async def reject(self, approval_id: str, reason: "str | None" = None) -> None: ...

    async def list_sessions(self) -> list: ...

    async def get_session(self, session_id: str) -> "SessionRecord | None": ...

    async def get_session_messages(
        self, session_id: str
    ) -> "tuple[tuple[Any, ...], ...]": ...

    async def list_session_turns(self, session_id: str) -> list: ...

    async def list_runs(self) -> list: ...

    async def get_run(self, run_id: str) -> "RunRecord | None": ...

    async def get_run_detail(self, run_id: str) -> Any: ...

    async def list_approvals(self) -> list: ...

    async def get_approval(self, approval_id: str) -> Any: ...

    async def list_agents(self) -> "tuple[str, ...]": ...

    async def list_skills(self) -> "tuple[str, ...]": ...

    async def list_mcp_servers(self) -> "tuple[str, ...]": ...

    async def inspect(self, agent_id: "str | None") -> Any: ...

    async def doctor(self) -> DoctorReport: ...


# --------------------------------------------------------------------------- #
# LocalRuntimeClient
# --------------------------------------------------------------------------- #


class LocalRuntimeClient:
    """In-process ``RuntimeClient`` over a project bundle.

    Owns the Runtime + Storage + registries so neither the console nor the TUI
    has to know how ``build_runtime`` is wired. Operates the backend as the
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
        # Sessions/runs the client itself started this process. The execution
        # store exposes no enumeration API, so the sidebar reflects only what
        # this CLI session produced (a remote/shared store would list more).
        self._known_sessions: "dict[str, SessionRecord]" = {}
        self._known_runs: "dict[str, RunRecord]" = {}
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
        self._remember_session(session_id)

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

    async def resume_stream(self, run_id: str) -> "AsyncIterator[Mapping[str, Any]]":
        async def _drive() -> "object":
            return await self._bundle.runtime.resume(run_id, principal=self._principal)

        first = True
        async for event in self._stream_run(_drive, run_id, run_id):
            if first:
                # Mark the resume boundary for the consumer before the events.
                yield {"type": "resumed", "run_id": run_id}
                first = False
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
        queue = sink.attach()
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
            sink.detach()
        if run_exc is not None:
            self._remember_run(run_id, session_id)
            yield {
                "type": "failed",
                "error_type": type(run_exc).__name__,
                "message": str(run_exc),
            }
            return
        try:
            result = task.result()
        except Exception as exc:  # noqa: BLE001 - surfaced as a failed event
            self._remember_run(run_id, session_id)
            yield {
                "type": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            return
        self._remember_run(run_id, session_id)
        # result is an ExecutionResultView for COMPLETED, or None for PAUSED/
        # CANCELLED. The live stream already carried the model text/tools; we
        # only emit the terminal classification here.
        if result is None:
            detail = await self.get_run_detail(run_id)
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
            self._remember_run(run_id, session_id)
            yield {
                "type": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            return
        self._remember_run(run_id, session_id)
        if result is None:
            detail = await self.get_run_detail(run_id)
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

    async def approve(self, approval_id: str) -> None:
        run_id = self._run_id_for_approval(approval_id)
        if run_id is None:
            raise CommandError(f"unknown approval {approval_id}")
        await resolve_approval(
            self._bundle.runtime,
            run_id,
            approval_id,
            approved=True,
            principal=self._principal,
        )
        self._pending_approvals.pop(run_id, None)

    async def reject(self, approval_id: str, reason: "str | None" = None) -> None:
        run_id = self._run_id_for_approval(approval_id)
        if run_id is None:
            raise CommandError(f"unknown approval {approval_id}")
        await resolve_approval(
            self._bundle.runtime,
            run_id,
            approval_id,
            approved=False,
            principal=self._principal,
        )
        await self._bundle.runtime.cancel(run_id, principal=self._principal)
        self._pending_approvals.pop(run_id, None)

    async def list_sessions(self) -> list:
        # Combine sessions created this process with every persisted session
        # the store knows about (the store's public list_all_sessions).
        records = list(self._known_sessions.values())
        seen = {getattr(r, "id", None) for r in records}
        try:
            for record in await self._bundle.storage.execution.list_all_sessions():
                if record.id not in seen:
                    records.append(record)
                    seen.add(record.id)
        except Exception:
            pass
        return records

    async def get_session(self, session_id: str) -> "SessionRecord | None":
        sid = validate_session_id(session_id)
        cached = self._known_sessions.get(sid)
        if cached is not None:
            return cached
        record = await self._bundle.storage.execution.get_session(sid)
        if record is not None:
            self._known_sessions[sid] = record
        return record

    async def get_session_messages(
        self, session_id: str
    ) -> "tuple[tuple[Any, ...], ...]":
        """Per-turn message deltas for a session (audit history), as returned
        by ``Runtime.get_session_messages``. Each inner tuple is one turn's
        recorded model messages; the caller decides how to fold them into the
        conversation. Empty list when the session is unknown or has no turns."""
        sid = validate_session_id(session_id)
        try:
            views = await self._bundle.runtime.get_session_messages(
                session_id=sid, principal=self._principal
            )
        except Exception:
            return ()
        return tuple(tuple(view.messages) for view in views)

    async def list_runs(self) -> list:
        # Combine runs created this process with every persisted run the store
        # knows about (the store's public list_all_runs).
        records = list(self._known_runs.values())
        seen = {getattr(r, "id", None) for r in records}
        try:
            for record in await self._bundle.storage.execution.list_all_runs():
                if record.id not in seen:
                    records.append(record)
                    seen.add(record.id)
        except Exception:
            pass
        return records

    async def list_session_turns(self, session_id: str) -> list:
        """Persisted turns for a session (input, status, run_id, sequence) via
        the store's ``list_session_turns``. Empty when the session is unknown."""
        sid = validate_session_id(session_id)
        try:
            page = await self._bundle.storage.execution.list_session_turns(sid)
        except Exception:
            return []
        return list(page.items)

    async def get_run(self, run_id: str) -> "RunRecord | None":
        cached = self._known_runs.get(run_id)
        if cached is not None:
            return cached
        record = await self._bundle.storage.execution.get_run(run_id)
        if record is not None:
            self._known_runs[run_id] = record
        return record

    async def list_approvals(self) -> list:
        # Materialize lightweight view objects for the pending approvals this
        # client surfaced via pause events.
        from ..execution.domain import RunApproval

        views = []
        for approval_id, (run_id, _tool) in self._pending_approvals.items():
            views.append(
                RunApproval(
                    approval_id=approval_id,
                    run_id=run_id,
                    tool_name="",
                    tool_call_id="",
                    binding_fingerprint="",
                )
            )
        return views

    async def get_approval(self, approval_id: str) -> Any:
        """One approval request by id -- the only way the console/TUI reads
        approval detail (no direct run approval access)."""
        run_id = self._run_id_for_approval(approval_id)
        if run_id is None:
            return None
        detail = await self.get_run_detail(run_id)
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

    async def list_agents(self) -> "tuple[str, ...]":
        return await self._bundle.agents.list_ids()

    async def list_skills(self) -> "tuple[str, ...]":
        return await self._bundle.skill_index.list_ids()

    async def list_mcp_servers(self) -> "tuple[str, ...]":
        return await self._bundle.mcp.list_ids()

    async def inspect(self, agent_id: "str | None") -> Any:
        spec = await load_agent_spec(self._bundle, agent_id)
        # inspect() is a no-side-effect resolution: assemble the spec's
        # features without executing, returning the summary the context panel
        # renders. The facade's inspect() keys on a persisted run_id (which
        # does not exist for a bare agent lookup), so resolve features here.
        try:
            from ..agent.assembly.provider import AgentFeatureContext

            context = AgentFeatureContext(
                agent_id=spec.id,
                execution_id="inspect",
                root_execution_id="inspect",
                parent_execution_id=None,
                session_id="inspect",
                tenant_id=self._principal.tenant_id,
                user_id=self._principal.user_id,
                workspace=None,
                sandbox=self._bundle.runtime.sandbox,
                events=None,
            )
            assembly = await self._bundle.runtime.assembler.assemble(spec, context)
        except Exception:
            return None
        return _InspectionView(
            id=spec.id,
            tools=tuple(t.descriptor.name for t in assembly.tools),
            skills=tuple(),
            mcp_servers=tuple(),
            extensions=tuple(),
            features=tuple(assembly.feature_owners.keys()),
        )

    async def doctor(self) -> DoctorReport:
        """Run every project/Runtime check against the bundle and return the
        structured verdict; the console/TUI only renders it."""
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

        return report

    # -- run detail (public) --------------------------------------------- #

    async def get_run_detail(self, run_id: str) -> "ExecutionDetailView | None":
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
        detail = await self.get_run_detail(run_id)
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

    def _remember_session(self, session_id: str) -> None:
        if session_id not in self._known_sessions:
            self._known_sessions[session_id] = _SessionSummary(
                id=session_id, tenant_id=self._principal.tenant_id
            )

    def _remember_run(self, run_id: str, session_id: str) -> None:
        self._known_runs[run_id] = _RunSummary(
            id=run_id,
            session_id=session_id,
            tenant_id=self._principal.tenant_id,
            status="running",
        )


# --------------------------------------------------------------------------- #
# FakeRuntimeClient -- shared double for console + TUI tests
# --------------------------------------------------------------------------- #


class FakeRuntimeClient:
    """In-memory ``RuntimeClient`` double. Records every call and replays
    canned stream events so the console/TUI layers can be tested without a real
    Runtime, model, or filesystem."""

    def __init__(
        self,
        *,
        stream_events: "list[Mapping[str, Any]] | None" = None,
        resume_events: "list[Mapping[str, Any]] | None" = None,
        stream_error: "BaseException | None" = None,
        sessions: "list | None" = None,
        runs: "list | None" = None,
        approvals: "list | None" = None,
        agents: "tuple[str, ...]" = (),
        skills: "tuple[str, ...]" = (),
        mcp_servers: "tuple[str, ...]" = (),
        inspection: Any = None,
        doctor_report: "DoctorReport | None" = None,
        run_record: Any = None,
        run_detail: Any = None,
        session_record: Any = None,
        session_messages: "tuple[tuple[Any, ...], ...] | None" = None,
        session_turns: "list | None" = None,
        approval: Any = None,
    ) -> None:
        self._stream_events = list(stream_events or [])
        self._resume_events = list(resume_events or [])
        self._stream_error = stream_error
        self._sessions = sessions or []
        self._runs = runs or []
        self._approvals = approvals or []
        self._approval = approval
        self._agents = agents
        self._skills = skills
        self._mcp_servers = mcp_servers
        self._inspection = inspection
        self._doctor_report = doctor_report or DoctorReport()
        self._run_record = run_record
        self._run_detail = run_detail
        self._session_record = session_record
        self._session_messages = session_messages or ()
        self._session_turns = session_turns or []
        # Call recordings.
        self.cancel_calls: "list[str]" = []
        self.approve_calls: "list[str]" = []
        self.reject_calls: "list[tuple[str, str | None]]" = []
        self.resume_calls: "list[str]" = []
        self.run_requests: "list[RunRequest]" = []
        self.last_run_id: "str | None" = None

    async def run_stream(
        self, request: "RunRequest"
    ) -> "AsyncIterator[Mapping[str, Any]]":
        self.run_requests.append(request)
        self.last_run_id = request.run_id
        for event in self._stream_events:
            yield event
        if self._stream_error is not None:
            raise self._stream_error

    async def resume_stream(self, run_id: str) -> "AsyncIterator[Mapping[str, Any]]":
        self.resume_calls.append(run_id)
        for event in self._resume_events:
            yield event

    async def cancel(self, run_id: str) -> None:
        self.cancel_calls.append(run_id)

    async def approve(self, approval_id: str) -> None:
        self.approve_calls.append(approval_id)

    async def reject(self, approval_id: str, reason: "str | None" = None) -> None:
        self.reject_calls.append((approval_id, reason))

    async def list_sessions(self) -> list:
        return list(self._sessions)

    async def get_session(self, session_id: str) -> Any:
        return self._session_record

    async def get_session_messages(
        self, session_id: str
    ) -> "tuple[tuple[Any, ...], ...]":
        return self._session_messages

    async def list_session_turns(self, session_id: str) -> list:
        return list(self._session_turns)

    async def list_runs(self) -> list:
        return list(self._runs)

    async def get_run(self, run_id: str) -> Any:
        return self._run_record

    async def get_run_detail(self, run_id: str) -> Any:
        return self._run_detail

    async def list_approvals(self) -> list:
        return list(self._approvals)

    async def get_approval(self, approval_id: str) -> Any:
        return self._approval

    async def list_agents(self) -> "tuple[str, ...]":
        return self._agents

    async def list_skills(self) -> "tuple[str, ...]":
        return self._skills

    async def list_mcp_servers(self) -> "tuple[str, ...]":
        return self._mcp_servers

    async def inspect(self, agent_id: "str | None") -> Any:
        return self._inspection

    async def doctor(self) -> DoctorReport:
        return self._doctor_report


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
    ``doctor``/``inspect``/listings work in a freshly-initialized project with no
    API key. ``project`` overrides where project discovery starts (the
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
