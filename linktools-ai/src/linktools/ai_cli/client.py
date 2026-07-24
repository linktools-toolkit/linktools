#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The single backend entry point for the ``lt ai`` console and TUI.

Both the thin console commands and the Textual TUI operate the Runtime + Storage
exclusively through :class:`RuntimeClient`. :class:`LocalRuntimeClient` is the
in-process implementation that owns the project bundle (Runtime + Storage +
registries) and translates Runtime dict-events into the protocol surface; the
TUI never touches ``FilesystemStorage`` / registries directly.
:class:`FakeRuntimeClient` is the shared double both console and TUI tests drive.

Local vs. Remote share the same interface; a remote (HTTP) client is not wired
up in this build and ``build_runtime_client(remote=...)`` fails explicitly
rather than pretending to support it (no fake implementation)."""

import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from linktools.ai.agent.approval import ApprovalStatus
from linktools.ai.session.models import SessionRecord, SessionStatus
from linktools.ai.storage.facade import FilesystemStorage
from linktools.cli import CommandError
from linktools.system import get_user

from .runtime import (
    CliRuntimeBundle,
    build_cli_runtime,
    load_agent_spec,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


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


async def ensure_session(storage: FilesystemStorage, session_id: str) -> None:
    """Get-or-create a session record.

    ``Runtime.run`` / ``Runtime.run_stream`` require a pre-existing session when
    a ``session_id`` is supplied (they do not auto-create). This mirrors the
    ``session_id=None`` branch exactly by creating the ``SessionRecord``
    up-front when the id is unseen."""
    if await storage.sessions.get(session_id) is None:
        now = datetime.now(timezone.utc)
        await storage.sessions.create(
            SessionRecord(
                id=session_id,
                parent_id=None,
                # The local CLI is single-principal: sessions it creates are
                # unowned (None/None) and only re-openable by an unowned caller.
                user_id=None,
                tenant_id=None,
                status=SessionStatus.ACTIVE,
                version=1,
                created_at=now,
                updated_at=now,
            )
        )


async def resolve_approval(
    storage: FilesystemStorage,
    approval_id: str,
    *,
    approved: bool,
    reason: "str | None",
) -> "int | None":
    """Resolve a pending approval request.

    Raises ``CommandError`` if the request is missing or no longer pending. The
    ``expected_version`` from the read request fences a concurrent resolve."""
    request = await storage.approvals.get(approval_id)
    if request is None:
        raise CommandError(f'approval "{approval_id}" not found')
    if request.status != ApprovalStatus.PENDING:
        raise CommandError(
            f'approval "{approval_id}" is already {request.status.value}'
        )
    resolved_by = get_user() or "cli"
    if approved:
        await storage.approvals.approve(
            approval_id,
            expected_version=request.version,
            resolved_by=resolved_by,
        )
    else:
        await storage.approvals.reject(
            approval_id,
            expected_version=request.version,
            resolved_by=resolved_by,
            reason=reason,
        )
    return 0


async def list_sessions(storage: FilesystemStorage) -> list:
    """Enumerate every session record by scanning the sessions directory.

    ``SessionStore`` exposes no ``list()``, so we glob the on-disk layout
    (``<root>/sessions/{id}/record.json``) and rehydrate each record through the
    store's own ``get()`` to keep deserialization encapsulated."""
    root = Path(storage.root) / "sessions"
    records = []
    for record_path in sorted(root.glob("*/record.json")):
        session_id = record_path.parent.name
        record = await storage.sessions.get(session_id)
        if record is not None:
            records.append(record)
    return records


async def list_runs(storage: FilesystemStorage) -> list:
    """Enumerate every run record by scanning the runs directory.

    ``RunStore`` exposes no whole-store ``list()`` (only ``list_children`` of a
    parent); we glob ``<root>/runs/*.json`` and rehydrate each record through
    ``storage.runs.get()``."""
    root = Path(storage.root) / "runs"
    records = []
    for record_path in sorted(root.glob("*.json")):
        record = await storage.runs.get(record_path.stem)
        if record is not None:
            records.append(record)
    return records


async def list_pending_approvals(storage: FilesystemStorage) -> list:
    """Enumerate pending approval requests across all runs.

    ``ApprovalStore`` has no whole-store listing API; we glob
    ``<root>/approvals/requests/*.json``, rehydrate each request via the store's
    ``get()``, and keep only pending ones."""
    root = Path(storage.root) / "approvals" / "requests"
    requests = []
    for request_path in sorted(root.glob("*.json")):
        approval_id = request_path.stem
        request = await storage.approvals.get(approval_id)
        if request is not None and request.status == ApprovalStatus.PENDING:
            requests.append(request)
    return requests


# --------------------------------------------------------------------------- #
# RuntimeClient protocol
# --------------------------------------------------------------------------- #


@runtime_checkable
class RuntimeClient(Protocol):
    """The only backend surface the console/TUI may use."""

    async def run_stream(
        self, request: RunRequest
    ) -> "AsyncIterator[Mapping[str, Any]]": ...

    async def resume_stream(
        self, run_id: str
    ) -> "AsyncIterator[Mapping[str, Any]]": ...

    async def cancel(self, run_id: str) -> None: ...

    async def approve(self, approval_id: str) -> None: ...

    async def reject(self, approval_id: str, reason: "str | None" = None) -> None: ...

    async def list_sessions(self) -> list: ...

    async def get_session(self, session_id: str): ...

    async def list_runs(self) -> list: ...

    async def get_run(self, run_id: str): ...

    async def list_approvals(self) -> list: ...

    async def get_approval(self, approval_id: str): ...

    async def list_agents(self) -> "tuple[str, ...]": ...

    async def list_skills(self) -> "tuple[str, ...]": ...

    async def list_mcp_servers(self) -> "tuple[str, ...]": ...

    async def inspect(self, agent_id: "str | None"): ...

    async def doctor(self) -> DoctorReport: ...


# --------------------------------------------------------------------------- #
# LocalRuntimeClient
# --------------------------------------------------------------------------- #


class LocalRuntimeClient:
    """In-process ``RuntimeClient`` over a project bundle.

    Owns the Runtime + Storage + registries so neither the console nor the TUI
    has to know how ``build_runtime`` is wired."""

    def __init__(self, bundle: CliRuntimeBundle) -> None:
        self._bundle = bundle

    @property
    def bundle(self) -> CliRuntimeBundle:
        return self._bundle

    async def run_stream(
        self, request: RunRequest
    ) -> "AsyncIterator[Mapping[str, Any]]":
        spec = await load_agent_spec(self._bundle, request.agent_id)
        session_id = validate_session_id(request.session_id)
        await ensure_session(self._bundle.storage, session_id)
        run_id = request.run_id or new_run_id()
        async for event in self._bundle.runtime.run_stream(
            spec, request.prompt, session_id=session_id, run_id=run_id
        ):
            yield event

    async def resume_stream(self, run_id: str) -> "AsyncIterator[Mapping[str, Any]]":
        async for event in self._bundle.runtime.resume(run_id):
            yield event

    async def cancel(self, run_id: str) -> None:
        await self._bundle.runtime.cancel(run_id)

    async def approve(self, approval_id: str) -> None:
        await resolve_approval(
            self._bundle.storage, approval_id, approved=True, reason=None
        )

    async def reject(self, approval_id: str, reason: "str | None" = None) -> None:
        await resolve_approval(
            self._bundle.storage, approval_id, approved=False, reason=reason
        )

    async def list_sessions(self) -> list:
        return await list_sessions(self._bundle.storage)

    async def get_session(self, session_id: str):
        return await self._bundle.storage.sessions.get(validate_session_id(session_id))

    async def list_runs(self) -> list:
        return await list_runs(self._bundle.storage)

    async def get_run(self, run_id: str):
        return await self._bundle.storage.runs.get(run_id)

    async def list_approvals(self) -> list:
        return await list_pending_approvals(self._bundle.storage)

    async def get_approval(self, approval_id: str):
        """One approval request by id -- the only way the console/TUI reads
        approval detail (no direct ApprovalStore access)."""
        return await self._bundle.storage.approvals.get(approval_id)

    async def list_agents(self) -> "tuple[str, ...]":
        return await self._bundle.agents.list_ids()

    async def list_skills(self) -> "tuple[str, ...]":
        return await self._bundle.skill_index.list_ids()

    async def list_mcp_servers(self) -> "tuple[str, ...]":
        return await self._bundle.mcp.list_ids()

    async def inspect(self, agent_id: "str | None"):
        spec = await load_agent_spec(self._bundle, agent_id)
        return await self._bundle.runtime.inspect(spec)

    async def doctor(self) -> DoctorReport:
        """Run every project/Runtime check against the bundle and return the
        structured verdict; the console/TUI only renders it."""
        from linktools.ai.mcp.env import expand_env_mapping
        from linktools.ai.skill.private import resolve_skill_agent_path

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
                expand_env_mapping(getattr(spec, "env", None))
                ok(f"MCP: {mcp_id}")
            except Exception as exc:
                fail(f"MCP: {mcp_id}", str(exc))

        # Runtime inspects cleanly for the default agent.
        try:
            default_spec = await bundle.agents.get(project.default_agent)
            await bundle.runtime.inspect(default_spec)
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
        session_record: Any = None,
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
        self._session_record = session_record
        # Call recordings.
        self.cancel_calls: "list[str]" = []
        self.approve_calls: "list[str]" = []
        self.reject_calls: "list[tuple[str, str | None]]" = []
        self.resume_calls: "list[str]" = []
        self.run_requests: "list[RunRequest]" = []
        self.last_run_id: "str | None" = None

    async def run_stream(
        self, request: RunRequest
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

    async def get_session(self, session_id: str):
        return self._session_record

    async def list_runs(self) -> list:
        return list(self._runs)

    async def get_run(self, run_id: str):
        return self._run_record

    async def list_approvals(self) -> list:
        return list(self._approvals)

    async def get_approval(self, approval_id: str):
        return self._approval

    async def list_agents(self) -> "tuple[str, ...]":
        return self._agents

    async def list_skills(self) -> "tuple[str, ...]":
        return self._skills

    async def list_mcp_servers(self) -> "tuple[str, ...]":
        return self._mcp_servers

    async def inspect(self, agent_id: "str | None"):
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
        from linktools.ai.model.registry import RuntimeModelConfig, model_registry
        from linktools.ai.model.resolver import ModelResolver

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
    bundle = build_cli_runtime(project=cli_project, model_resolver=resolver)
    return LocalRuntimeClient(bundle)
