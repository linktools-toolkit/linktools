#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunCoordinator: the single application service for a Run's lifecycle --
create/start, pause, approve/reject, resume, cancel, commit (via
commit_coordinator), terminal convergence. The SOLE owner of RunRecord
creation/transition, checkpoint/session/approval persistence, execution
claim/heartbeat/fencing, and commit retry/recovery. AgentEngine (the
``agent_engine`` dep) is a pure execution loop that returns an
AgentExecutionOutcome and writes NO Store state itself.

Runtime is a thin facade delegating every run-lifecycle method here."""

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any, Mapping

from .cancellation import CancellationToken
from .commit import (
    AcknowledgeCancelRunCommand,
    ApprovalRequestData,
    CompleteRunCommand,
    ExecutionFence,
    FailRunCommand,
    PauseRunCommand,
    RunCommitId,
    StartRunCommand,
)
from .dispatch import ChildRunHandle
from .lifecycle import prepare_run
from .models import RunErrorInfo, RunInput, RunRecord, RunStatus
from .preparation import RunPreparationCoordinator
from ..agent.models import (
    AgentCancelled,
    AgentCompleted,
    AgentExecutionOutcome,
    AgentFailed,
    AgentInput,
    AgentPaused,
    CompiledAgent,
)
from ..clock import Clock
from ..errors import RunConflictError, SwarmError
from ..events.context import EventStreamContext, append_event
from ..events.payloads import (
    RunCancelled as RunCancelledEvent,
    RunCompleted as RunCompletedEvent,
    RunFailed as RunFailedEvent,
    RunPaused as RunPausedEvent,
    RunResumed as RunResumedEvent,
    RunStarted as RunStartedEvent,
)
from ..session.reader import SessionReader
from ..swarm.models import (
    SwarmCompleted as SwarmCompletedType,
    SwarmFailed as SwarmFailedType,
)
from ..swarm.spec import SwarmSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..agent.engine import AgentEngine
    from ..agent.spec import AgentSpec
    from ..governance.security.emitter import SecurityEventSanitizer
    from ..identity.principal import PrincipalContext
    from ..observability.metrics import ObservabilityMetrics
    from ..run.controller import RunController
    from ..run.live_events import RunLiveEventHub, RunLiveEventSink, SecurityEventSink
    from ..runtime.persistence.facade import Storage
    from ..swarm.engine import SwarmEngine
    from .commit import RunCommitCoordinator


# How often the Coordinator renews its execution claim while a run is in
# flight. A renewal failure (lost fencing) cancels the engine task so a stale
# worker can never commit a terminal side effect.
_HEARTBEAT_INTERVAL_SECONDS = 10.0


class _EventStoreEventSink:
    """Interim Coordinator-owned event sink: appends a domain event (a
    SecurityEventSink payload such as ToolExposureApplied, or a swarm lifecycle
    event such as SwarmStarted) to the EventStore under the Run's stream, so the
    audit trail is preserved while AgentEngine/SwarmEngine are Store-free. This
    is the single-``emit`` shape RunCoordinator passes in; the durable,
    fencing-token-bound sink that replaces it lands with the event-sink
    decoupling."""

    def __init__(self, event_store: Any, context: Any) -> None:
        self._event_store = event_store
        self._context = context

    async def emit(self, event: Any) -> None:
        await append_event(
            self._event_store,
            EventStreamContext.from_run_context(self._context),
            event,
        )


class _FencedSecurityEventSink:
    """Per-execution SecurityEventSink that delegates to a FencedRunEventWriter.

    Every emit verifies (within the writer's storage-specific UoW) that the
    presented ExecutionFence still matches the RunRecord's stored
    execution_token. A stale/empty/missing fence raises RunFenceLostError
    BEFORE the event lands; that error propagates back through AgentEngine to
    the Coordinator, which routes the run into fail/fencing-loss convergence
    rather than letting the security-sensitive action that triggered the emit
    proceed."""

    def __init__(self, writer: Any, context: Any, fence: "ExecutionFence") -> None:
        self._writer = writer
        self._context = context
        self._fence = fence

    async def emit(self, event: Any) -> None:
        await self._writer.append_security(
            context=self._context,
            fence=self._fence,
            event=event,
        )


class RunCoordinator:
    def __init__(
        self,
        *,
        storage: "Storage",
        compiler: Any,
        agent_engine: "AgentEngine",
        swarm_engine: "SwarmEngine",
        commit_coordinator: "RunCommitCoordinator",
        run_controller: "RunController | None",
        live_event_hub: "RunLiveEventHub",
        session_reader: SessionReader,
        preparation: RunPreparationCoordinator,
        clock: Clock,
        authorization: Any = None,
        settings: Any = None,
        schema_registry: Any = None,
        metrics: "ObservabilityMetrics | None" = None,
        model_resolver: Any = None,
        fenced_event_writer: Any = None,
    ) -> None:
        self._storage = storage
        self._compiler = compiler
        self._agent_engine = agent_engine
        self._swarm_engine = swarm_engine
        self._commit_coordinator = commit_coordinator
        self._run_controller = run_controller
        self._live_event_hub = live_event_hub
        self._session_reader = session_reader
        self._prepare = preparation
        self._clock = clock
        self._authorization = authorization
        self._settings = settings
        self._schema_registry = schema_registry
        self._metrics = metrics
        # FencedRunEventWriter (Protocol) -- when supplied, every security
        # event the Coordinator forwards to AgentEngine routes through a
        # per-execution sink that verifies the claiming execution's fence
        # BEFORE appending, so a stale worker cannot land a security event
        # after losing its lease. None keeps the legacy unfenced sink for
        # tests/local-mode single-process runs.
        self._fenced_event_writer = fenced_event_writer
        self._model_resolver = model_resolver
        # One-time crash-recovery guard: the File coordinator's journal is
        # replayed before the first run/resume so an interrupted pause/complete
        # is made consistent. No-op for coordinators without recovery (SQL) and
        # idempotent (recovery discards each journal it resolves).
        self._recovery_done = False
        self._recovery_lock = asyncio.Lock()

    async def _ensure_recovered(self) -> None:
        if self._recovery_done:
            return
        async with self._recovery_lock:
            if self._recovery_done:
                return
            recover = getattr(self._commit_coordinator, "recover_incomplete_commits", None)
            if recover is not None:
                await recover()
            self._recovery_done = True

    async def run(
        self,
        spec: "AgentSpec | SwarmSpec",
        prompt: str,
        *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
        agents: "Mapping[str, AgentSpec] | None" = None,
        context_metadata: "Mapping[str, Any] | None" = None,
    ):
        await self._ensure_recovered()
        prepared = await prepare_run(
            storage=self._storage,
            spec=spec,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            context_metadata=context_metadata,
        )

        if isinstance(spec, SwarmSpec):
            if agents is None:
                raise SwarmError("agents mapping is required to run a SwarmSpec")
            run_input = RunInput(prompt=prompt)
            # prepare + snapshot (the immutable run-definition snapshot lives in
            # the Coordinator now, before the start command -- SwarmEngine no
            # longer writes it), then the atomic driving-Run start, then the
            # swarm lifecycle template: claim/heartbeat -> SwarmEngine.execute ->
            # outcome switch -> commit. SwarmEngine owns only the SwarmRun/
            # strategy; the driving Run is the Coordinator's.
            await self._prepare.prepare_swarm_run(
                spec=spec, members=agents, context=prepared.context
            )
            started = await self._start_run(prepared.context, run_input)
            return await self._drive_swarm(
                spec,
                run_input,
                prepared.context,
                agents=agents,
                running_version=started.version,
            )

        compiled = await self._compiler.compile(spec)
        # Persist the immutable run-definition snapshot AFTER compile (single
        # owner) so the resolved model bundle's revision is captured in the
        # manifest -- resume refuses if the provider config has since drifted.
        await self._prepare.prepare_agent_run(
            spec=spec,
            context=prepared.context,
            model_bundle=compiled.model_bundle,
        )
        run_input = RunInput(prompt=prompt)
        # Atomic Run start: commit_coordinator.start creates the RUNNING record
        # AND appends RunStarted in one commit -- never create_and_start_run()
        # followed by a separate append_event(RunStarted) (the old two-step
        # left a crash window between the record and the event). RunStarted is
        # always emitted, exactly once, with no skip flag.
        started = await self._start_run(prepared.context, run_input)
        return await self._drive_agent(
            compiled,
            prepared.context,
            run_input,
            resuming=False,
            message_history=(),
            running_version=started.version,
        )

    async def _authorize_sensitive(
        self,
        run_id: str,
        principal: "PrincipalContext | None",
        *,
        action: str,
    ) -> None:
        """Gate shared by sensitive operations (cancel, resume): require a
        Principal, default-deny without one (unless local_trusted_mode), and
        enforce tenant ownership. Delegates to run.sensitive so this module
        stays free of the deprecation-warning token."""
        from .sensitive import authorize_sensitive_operation

        await authorize_sensitive_operation(
            storage=self._storage,
            local_trusted_mode=self._settings.local_trusted_mode,
            run_id=run_id,
            principal=principal,
            action=action,
            authorization=self._authorization,
        )

    async def cancel(
        self,
        run_id: str,
        *,
        principal: "PrincipalContext | None" = None,
        reason: "str | None" = None,
    ) -> None:
        """Cancel an in-flight Run.

        Two paths, depending on whether a live asyncio.Task is registered with
        the RunController:

        * **In-flight task registered** -- the run is actually being driven by
          RunCoordinator (execute_pure). Transition the store to CANCELLING
          (distinguishes "cancel requested" from "actually cancelled"), then
          call ``run_controller.cancel(run_id)`` which (a) sets the
          CancellationToken so execute_pure's next execution-point check raises
          CancelledError, and (b) calls ``task.cancel()`` so any hanging await
          inside the model call also unblocks.

          If the record is already CANCELLING, skip straight to
          ``run_controller.cancel(run_id)`` -- re-transitioning is not a legal
          edge, so repeated cancel() calls are idempotent.

        * **No in-flight task** -- there is nothing to actually stop, so the
          store goes directly to CANCELLED.

        Idempotent: a Run already in a terminal status is a no-op. Raises
        :class:`RunNotFoundError` when the run does not exist;
        :class:`PrincipalAccessDeniedError` when no ``principal`` is supplied
        and the Runtime is not in ``local_trusted_mode``."""
        from datetime import datetime, timezone

        from ..errors import RunConflictError, RunNotFoundError
        from .models import RunStatus

        storage = self._storage
        controller = self._run_controller
        # Gate before any state change (and before revealing run state), so
        # the sensitive op never acts on a bare id.
        await self._authorize_sensitive(run_id, principal, action="cancel")
        record = await storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status in (
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        ):
            return

        # Cancel-request audit. The timestamp is always recorded; the identity
        # (cancel_requested_by) and reason are None when no Principal was
        # supplied (trusted-local cancel) -- there is no trusted identity then.
        cancel_at = datetime.now(timezone.utc)
        cancel_by = principal.resolved_by if principal is not None else None
        audit = {
            "cancel_requested_at": cancel_at,
            "cancel_requested_by": cancel_by,
            "cancel_reason": reason,
        }

        in_flight = controller is not None and controller.get_token(run_id) is not None
        if in_flight:
            if record.status == RunStatus.CANCELLING:
                await controller.cancel(run_id)
                return

            try:
                await storage.runs.transition(
                    run_id,
                    RunStatus.CANCELLING,
                    expected_version=record.version,
                    **audit,
                )
            except RunConflictError:
                fresh = await storage.runs.get(run_id)
                if fresh is None:
                    raise RunNotFoundError(f"run not found: {run_id}")
                if fresh.status in (
                    RunStatus.SUCCEEDED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                ):
                    return
                if fresh.status == RunStatus.CANCELLING:
                    await controller.cancel(run_id)
                    return
                raise

            await controller.cancel(run_id)
        else:
            # A worker-owned run must be acknowledged by that worker before
            # it can claim the terminal state. Legacy records without fencing
            # metadata retain the old seeded/local behavior for back-compat.
            target = RunStatus.CANCELLING if record.worker_id else RunStatus.CANCELLED
            await storage.runs.transition(
                run_id, target, expected_version=record.version, **audit
            )
        self._metrics.counter("run_cancellation_requested_total")

    async def run_stream(
        self,
        spec: "AgentSpec | SwarmSpec",
        prompt: str,
        *,
        session_id: "str | None" = None,
        run_id: "str | None" = None,
        user_id: "str | None" = None,
        tenant_id: "str | None" = None,
        context_metadata: "Mapping[str, Any] | None" = None,
    ) -> "AsyncIterator[dict]":
        """Streaming variant of :meth:`run`. Only ``AgentSpec`` is supported --
        a ``SwarmSpec`` raises :class:`SwarmError` because swarm streaming is not
        implemented. Session resolution mirrors :meth:`run` exactly."""
        if isinstance(spec, SwarmSpec):
            raise SwarmError("run_stream does not support SwarmSpec")

        await self._ensure_recovered()
        prepared = await prepare_run(
            storage=self._storage,
            spec=spec,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            context_metadata=context_metadata,
        )

        compiled = await self._compiler.compile(spec)
        # Persist the immutable run-definition snapshot AFTER compile (single
        # owner) so the resolved model bundle's revision is captured in the
        # manifest for drift detection on resume.
        await self._prepare.prepare_agent_run(
            spec=spec,
            context=prepared.context,
            model_bundle=compiled.model_bundle,
        )
        run_input = RunInput(prompt=prompt)
        # Atomic Run start -- see run(). RunStarted always emitted, exactly once.
        started = await self._start_run(prepared.context, run_input)
        async for event in self._drive_agent_stream(
            compiled,
            prepared.context,
            run_input,
            resuming=False,
            message_history=(),
            running_version=started.version,
        ):
            yield event

    async def approve(
        self,
        approval_id: str,
        *,
        principal: "PrincipalContext",
        expected_version: int,
    ):
        """Approve through the Principal-bound service, never a caller id."""
        from ..agent.approval_service import ApprovalService

        return await ApprovalService(self._storage.approvals, self._authorization).approve(
            approval_id, principal=principal, expected_version=expected_version
        )

    async def reject(
        self,
        approval_id: str,
        *,
        principal: "PrincipalContext",
        expected_version: int,
        reason: "str | None" = None,
    ):
        from ..agent.approval_service import ApprovalService

        return await ApprovalService(self._storage.approvals, self._authorization).reject(
            approval_id,
            principal=principal,
            expected_version=expected_version,
            reason=reason,
        )

    async def resume(
        self,
        run_id: str,
        *,
        principal: "PrincipalContext | None" = None,
    ) -> "AsyncIterator[dict]":
        """Resume a paused Run from its immutable persisted definition. Loads
        the RunDefinitionSnapshot, restores the ORIGINAL spec + identity (not a
        caller-supplied one), verifies the spec fingerprint, deserializes the
        checkpoint, transitions WAITING_APPROVAL -> RUNNING, and re-enters
        :meth:`AgentEngine.run_stream`.

        Yields ``{"type": "resumed", "run_id": run_id}`` first, then the same
        dict-event shape ``run_stream`` yields. Raises :class:`RunNotFoundError`
        when the run/checkpoint/snapshot does not exist;
        :class:`InvalidRunTransitionError` when the run is not WAITING_APPROVAL
        or the spec fingerprint does not match;
        :class:`PrincipalAccessDeniedError` when no ``principal`` is supplied
        and the Runtime is not in ``local_trusted_mode``."""
        from ..agent.approval import ApprovalStatus
        from ..agent.checkpoint import deserialize_messages
        from ..errors import (
            InvalidRunTransitionError,
            RunNotFoundError,
            RunNotResumableError,
        )
        from .definition import deserialize_agent_spec, spec_fingerprint
        from .manifest import (
            DefaultManifestResolver,
            Resumability,
            manifest_from_dict,
        )
        from .models import RunStatus

        await self._ensure_recovered()
        storage = self._storage
        # Gate before revealing run state.
        await self._authorize_sensitive(run_id, principal, action="resume")
        # 1. Read RunRecord. 2. Require WAITING_APPROVAL.
        record = await storage.runs.get(run_id)
        if record is None:
            raise RunNotFoundError(f"run not found: {run_id}")
        if record.status != RunStatus.WAITING_APPROVAL:
            raise InvalidRunTransitionError(
                f"cannot resume run in status {record.status}"
            )

        # 3. Read snapshot. 4. Recompute + verify fingerprint.
        snapshot = await storage.run_definitions.get(run_id)
        if snapshot is None:
            raise RunNotFoundError(f"no run-definition snapshot for run: {run_id}")
        if snapshot.resumability == Resumability.NON_RESUMABLE.value:
            # A run marked NON_RESUMABLE at creation cannot be resumed
            # deterministically -- refuse up-front rather than silently
            # re-resolving a drifted environment.
            raise RunNotResumableError(
                f"run {run_id}: marked non-resumable; cannot resume"
            )
        spec = deserialize_agent_spec(
            snapshot.serialized_spec,
            schema_registry=self._schema_registry,
        )
        if spec_fingerprint(spec) != snapshot.spec_fingerprint:
            raise InvalidRunTransitionError(
                f"run {run_id}: spec fingerprint mismatch -- the persisted "
                f"definition was tampered with or serialized incorrectly"
            )

        # 4b. Manifest drift check: re-resolve the current environment against
        # the persisted manifest and refuse if the provider revision drifted
        # between prepare and resume -- never silently fall back to the latest
        # config. Skipped for snapshots with no recorded manifest.
        if snapshot.manifest:
            from ..model.policy import ModelPolicy  # noqa: PLC0415 (lazy import)

            persisted_manifest = manifest_from_dict(dict(snapshot.manifest))

            async def _current_model_revision(name: str) -> "str | None":
                # Re-resolve ONLY the pinned model name (no fallbacks) so a
                # missing primary surfaces as "unresolvable" rather than
                # silently resolving to a fallback and reporting "drifted".
                try:
                    bundle = self._model_resolver.resolve(
                        ModelPolicy(primary=name, fallbacks=())
                    )
                except Exception:
                    return None
                return getattr(bundle, "revision", None)

            await DefaultManifestResolver(_current_model_revision).resolve(
                persisted_manifest, spec=spec
            )

        # 5. Latest checkpoint. 6. approval_id from checkpoint metadata.
        checkpoint = await storage.checkpoints.latest(run_id)
        if checkpoint is None:
            raise RunNotFoundError(f"no checkpoint for run: {run_id}")
        approval_id = (checkpoint.metadata or {}).get("approval_id")
        # 7-8. Query ApprovalRequest; require APPROVED (fail-closed). A run may
        # only resume after explicit approval -- PENDING/REJECTED/missing all
        # refuse, leaving the run WAITING_APPROVAL (no state change yet).
        if not approval_id:
            raise InvalidRunTransitionError(
                f"run {run_id}: checkpoint has no approval_id; cannot resume"
            )
        approval = await storage.approvals.get(approval_id)
        if approval is None:
            raise InvalidRunTransitionError(
                f"run {run_id}: approval {approval_id} not found; cannot resume"
            )
        if approval.status is not ApprovalStatus.APPROVED:
            raise InvalidRunTransitionError(
                f"run {run_id}: approval {approval_id} is {approval.status.value}, "
                f"not APPROVED; cannot resume"
            )

        # 9. Deserialize checkpoint. 10. Spec restored above. 11. Compile.
        # ALL of 1-11 must succeed BEFORE the CAS transition (step 13): a
        # compile failure or a tampered checkpoint must leave the run
        # WAITING_APPROVAL, not RUNNING.
        messages = deserialize_messages(checkpoint.payload)
        compiled = await self._compiler.compile(spec)
        # 12. Construct the full context, restoring the ORIGINAL identity from
        # the snapshot (user/tenant/workspace) + lineage from the record.
        from ..runtime.assembly.lifecycle import create_run_context

        context = create_run_context(
            run_id=run_id,
            session_id=record.session_id,
            runnable_id=record.runnable_id,
            runnable_type=record.runnable_type,
            user_id=snapshot.user_id,
            tenant_id=snapshot.tenant_id,
            workspace=snapshot.workspace,
            root_run_id=record.root_run_id,
            parent_run_id=record.parent_run_id,
        )
        # 13. CAS WAITING_APPROVAL -> RUNNING (only after every check + compile).
        # The returned record's version is the RUNNING version the terminal
        # commit (complete/pause/fail) must target -- claim/heartbeat do NOT
        # bump the version, so this is still authoritative at commit time.
        resumed_record = await storage.runs.transition(
            run_id,
            RunStatus.RUNNING,
            expected_version=record.version,
        )
        # 14. Resume execution via the pure loop. The ORIGINAL user prompt is
        # carried through so the complete commit persists a real USER message
        # (not an empty one). ``resuming=True`` + the checkpointed
        # ``message_history`` make execute_pure resume the graph from the
        # paused state (the CAS above already moved the record RUNNING, so
        # _drive_agent_stream does NOT re-start the run).
        yield {"type": "resumed", "run_id": run_id}
        async for event in self._drive_agent_stream(
            compiled,
            context,
            RunInput(prompt=record.input.prompt or ""),
            resuming=True,
            message_history=tuple(messages),
            running_version=resumed_record.version,
        ):
            yield event

    # ------------------------------------------------------------------
    # Agent execution core (the Coordinator's run template): atomic start,
    # execution claim
    # + heartbeat + fencing, session-history load, AgentEngine.execute_pure,
    # outcome -> commit command, heartbeat/register teardown.
    # ------------------------------------------------------------------

    async def _start_run(
        self, context: Any, run_input: RunInput
    ) -> RunRecord:
        """Atomically create the RUNNING RunRecord + append RunStarted in one
        commit_coordinator.start call. The single Run-start path for every
        top-level entry point (run/run_stream)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        record = RunRecord(
            id=context.run_id,
            root_run_id=context.root_run_id,
            parent_run_id=context.parent_run_id,
            session_id=context.session_id,
            runnable_id=context.runnable_id,
            runnable_type=context.runnable_type,
            status=RunStatus.RUNNING,
            input=run_input,
            result=None,
            error=None,
            version=1,
            created_at=now,
            started_at=now,
            finished_at=None,
        )
        commit = await self._commit_coordinator.start(
            StartRunCommand(
                record=record,
                started_event=RunStartedEvent(
                    run_id=context.run_id, runnable_id=context.runnable_id
                ),
                event_context=EventStreamContext.from_run_context(context),
                commit_id=RunCommitId(f"start:{context.run_id}"),
            )
        )
        return commit.record

    async def open_child_run(
        self,
        parent_context: Any,
        session_policy: Any,
        metadata: "Mapping[str, Any]",
    ) -> ChildRunHandle:
        """Allocate a child run's id + session id + lineage (pure -- NO store
        writes). The Coordinator is the sole id authority: a caller (swarm
        strategy, subagent executor) must not mint child run ids, build child
        RunContexts, create SessionRecords, or write RunDefinitions itself.

        Allocating the id separately from ``dispatch_child`` lets a caller
        record the child run id on its own domain state BEFORE execution starts
        (the swarm strategy writes ``task.active_run_id`` so cancel/recover can
        locate the in-flight child). A crash between this call and
        ``dispatch_child`` leaves no orphan RunRecord/SessionRecord."""
        child_run_id = str(uuid.uuid4())
        if session_policy.kind == "shared":
            session_id = parent_context.session_id
            needs_create = False
        elif session_policy.session_id_format:
            session_id = session_policy.session_id_format.format(
                child_run_id=child_run_id, **metadata
            )
            needs_create = True
        else:
            session_id = str(uuid.uuid4())
            needs_create = True
        root_run_id = (
            parent_context.root_run_id
            or parent_context.run_id
            or child_run_id
        )
        return ChildRunHandle(
            run_id=child_run_id,
            session_id=session_id,
            root_run_id=root_run_id,
            parent_run_id=parent_context.run_id,
            parent_session_id=session_policy.parent_session_id,
            user_id=parent_context.user_id,
            tenant_id=parent_context.tenant_id,
            workspace=getattr(parent_context, "workspace", None),
            session_needs_create=needs_create,
        )

    async def dispatch_child(self, request: Any) -> Any:
        """Child-run entry for CoordinatorRunDispatcher (Swarm/Subagent). Owns
        the full child lifecycle the strategy/subagent executor used to perform
        itself: scratch-session create, RunContext build (from the handle's
        lineage), resumable snapshot prepare, atomic start, claim/heartbeat/
        fencing, execute_pure, terminal commit. Returns the RunResult; raises
        RunPaused on a pause and propagates on failure, matching the
        RunDispatcher contract."""
        from .context import RunContext
        from .models import RunnableType
        from ..session.models import SessionRecord, SessionStatus
        from datetime import datetime, timezone

        handle = request.handle
        compiled = request.compiled_agent
        run_input = request.input
        context = RunContext(
            run_id=handle.run_id,
            root_run_id=handle.root_run_id,
            parent_run_id=handle.parent_run_id,
            session_id=handle.session_id,
            runnable_id=request.metadata.get("runnable_id") or compiled.spec.id,
            runnable_type=RunnableType.AGENT,
            user_id=handle.user_id,
            tenant_id=handle.tenant_id,
            workspace=handle.workspace,
        )
        if handle.session_needs_create:
            now = datetime.now(timezone.utc)
            await self._storage.sessions.create(
                SessionRecord(
                    id=handle.session_id,
                    parent_id=handle.parent_session_id,
                    user_id=handle.user_id,
                    tenant_id=handle.tenant_id,
                    status=SessionStatus.ACTIVE,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        # A child run gets the same resumable snapshot as a top-level run: if a
        # child tool pauses on approval, Runtime.resume(child_run_id) can
        # restore its spec + identity.
        await self._prepare.prepare_agent_run(spec=compiled.spec, context=context)
        started = await self._start_run(context, run_input)
        return await self._drive_agent(
            compiled,
            context,
            run_input,
            resuming=False,
            message_history=(),
            running_version=started.version,
        )

    async def _drive_agent(
        self,
        compiled: CompiledAgent,
        context: Any,
        run_input: RunInput,
        *,
        resuming: bool,
        message_history: tuple,
        running_version: "int | None",
    ) -> Any:
        """Non-streaming agent execution: claim + heartbeat + register, drive
        execute_pure to a terminal outcome, commit it, tear down. Returns the
        committed RunResult. Live events go to a NullRunLiveEventSink (no
        consumer for a non-streaming run)."""
        from ..run.live_events import NullRunLiveEventSink

        async with self._claim_and_fence(context) as (cancellation, token, _live):
            outcome = await self._execute_and_commit(
                compiled,
                context,
                run_input,
                resuming=resuming,
                message_history=message_history,
                running_version=running_version,
                cancellation=cancellation,
                token=token,
                live_events=NullRunLiveEventSink(),
            )
        return self._result_from_outcome(outcome, context.run_id)

    async def _drive_swarm(
        self,
        spec: Any,
        run_input: RunInput,
        context: Any,
        *,
        agents: Any,
        running_version: int,
    ) -> Any:
        """Swarm lifecycle template: claim/heartbeat -> SwarmEngine.execute ->
        outcome switch -> commit the driving Run. SwarmEngine owns only the
        SwarmRun/strategy; the driving RunRecord is the Coordinator's. Returns
        the aggregate RunResult on completion."""
        async with self._claim_and_fence(context) as (cancellation, token, _live):
            try:
                outcome = await self._swarm_engine.execute(
                    spec,
                    run_input,
                    context,
                    agents=agents,
                    cancellation=cancellation,
                )
            except asyncio.CancelledError:
                # Explicit cancellation: SwarmEngine already transitioned its
                # SwarmRun CANCELLED; converge the driving Run via the same
                # acknowledge_cancel commit command the agent path uses (with
                # the fencing token + a RunCancelled event), best-effort.
                if cancellation.is_cancelled():
                    await self._acknowledge_swarm_cancel(context, token)
                raise
            await self._commit_swarm_outcome(context, outcome, token, running_version)
        return self._swarm_result_from_outcome(outcome)

    async def _commit_swarm_outcome(
        self,
        context: Any,
        outcome: Any,
        token: str,
        running_version: int,
    ) -> None:
        """Converge the driving Run from a SwarmExecutionOutcome. Reuses the
        commit_coordinator (the same atomic path the agent complete/fail uses)
        so the driving Run's transition + event append in one commit. The
        fencing token is threaded into every command, mirroring the agent
        path -- a stale worker that lost the claim is rejected on mismatch."""
        run_id = context.run_id
        event_ctx = EventStreamContext.from_run_context(context)
        current = await self._storage.runs.get(run_id)
        expected_version = current.version if current is not None else running_version
        # SwarmCompleted -> complete; SwarmFailed -> fail; SwarmPaused leaves
        # the driving Run RUNNING (the swarm is paused, not the driving Run).
        if isinstance(outcome, SwarmCompletedType):
            await self._commit_coordinator.complete(
                CompleteRunCommand(
                    run_id=run_id,
                    session_id=context.session_id,
                    expected_version=expected_version,
                    execution_fence=ExecutionFence(token) if token else None,
                    messages=tuple(outcome.aggregate_messages),
                    checkpoint_payload=b"",
                    result=outcome.result,
                    completed_event=RunCompletedEvent(run_id=run_id),
                    event_context=event_ctx,
                    commit_id=RunCommitId(f"swarm-complete:{run_id}"),
                )
            )
        elif isinstance(outcome, SwarmFailedType):
            await self._commit_coordinator.fail(
                FailRunCommand(
                    run_id=run_id,
                    expected_version=expected_version,
                    execution_fence=ExecutionFence(token) if token else None,
                    error=outcome.error,
                    failed_event=RunFailedEvent(
                        run_id=run_id,
                        error_type=outcome.error.error_type,
                        message=outcome.error.message,
                    ),
                    event_context=event_ctx,
                    commit_id=RunCommitId(f"swarm-fail:{run_id}"),
                )
            )

    async def _acknowledge_swarm_cancel(
        self, context: Any, token: str
    ) -> None:
        """Converge the driving Run CANCELLING -> CANCELLED via the same
        acknowledge_cancel commit command the agent path uses: fenced by the
        execution token and persisting a RunCancelled event atomically with
        the transition. Best-effort -- a version/transition conflict (the Run
        already converged, or is not yet CANCELLING) is swallowed so the
        CancelledError that brought us here still propagates."""
        from ..errors import RunConflictError, InvalidRunTransitionError

        run_id = context.run_id
        try:
            current = await self._storage.runs.get(run_id)
            if current is None or current.status is RunStatus.CANCELLED:
                return
            await self._commit_coordinator.acknowledge_cancel(
                AcknowledgeCancelRunCommand(
                    run_id=run_id,
                    expected_version=current.version,
                    execution_fence=ExecutionFence(token) if token else None,
                    cancelled_event=RunCancelledEvent(run_id=run_id),
                    event_context=EventStreamContext.from_run_context(context),
                    commit_id=RunCommitId(f"swarm-ack-cancel:{run_id}"),
                )
            )
        except (RunConflictError, InvalidRunTransitionError):
            return

    @staticmethod
    def _swarm_result_from_outcome(outcome: Any) -> Any:
        """The non-streaming run() contract returns a RunResult. A completed
        swarm returns its aggregate result; a failed swarm raises (the driving
        Run is already FAILED); a paused swarm returns None (the driving Run
        stays RUNNING)."""
        if isinstance(outcome, SwarmCompletedType):
            return outcome.result
        if isinstance(outcome, SwarmFailedType):
            raise RuntimeError(
                f"swarm failed ({outcome.error.error_type}): {outcome.error.message}"
            )
        return None

    async def _drive_agent_stream(
        self,
        compiled: CompiledAgent,
        context: Any,
        run_input: RunInput,
        *,
        resuming: bool,
        message_history: tuple,
        running_version: "int | None",
    ) -> "AsyncIterator[dict]":
        """Streaming agent execution: open a live-event handle, drive
        execute_pure + commit in a background task while yielding live events
        to the caller, then surface the terminal outcome as a final event."""
        handle = await self._live_event_hub.open(context.run_id)
        cancellation = CancellationToken()
        engine_task = asyncio.create_task(
            self._stream_drive(
                compiled,
                context,
                run_input,
                resuming=resuming,
                message_history=message_history,
                running_version=running_version,
                cancellation=cancellation,
                handle=handle,
            )
        )
        try:
            async for event in handle.events():
                yield event
        finally:
            # On early consumer exit (GeneratorExit / external task cancel),
            # stop the engine so the run converges. Force-cancel the task IN
            # ADDITION to the cooperative flag: if the engine is currently
            # suspended in handle.publish() on a saturated bounded queue, the
            # flag is never re-checked there, so only task.cancel() unblocks it
            # -- otherwise the engine task would be orphaned mid-publish.
            if not engine_task.done():
                cancellation.cancel()
                engine_task.cancel()
            # Always await INSIDE the finally so a normal-completion exception
            # surfaces and the task is never orphaned. On the early-exit path
            # the await re-raises the CancelledError we just injected; suppress
            # it so the original exit exception (GeneratorExit/CancelledError)
            # propagates cleanly.
            try:
                await engine_task
            except asyncio.CancelledError:
                pass

    async def _stream_drive(
        self,
        compiled: CompiledAgent,
        context: Any,
        run_input: RunInput,
        *,
        resuming: bool,
        message_history: tuple,
        running_version: "int | None",
        cancellation: CancellationToken,
        handle: Any,
    ) -> None:
        """Background body of _drive_agent_stream: runs the claim/heartbeat/
        execute/commit/teardown core with the live handle as the event sink,
        then closes the handle so the streaming consumer's ``async for`` ends,
        and emits the terminal outcome as a final dict event."""
        from ..run.live_events import NullRunLiveEventSink

        try:
            async with self._claim_and_fence(context, cancellation=cancellation) as (
                _cancellation,
                token,
                _live,
            ):
                outcome = await self._execute_and_commit(
                    compiled,
                    context,
                    run_input,
                    resuming=resuming,
                    message_history=message_history,
                    running_version=running_version,
                    cancellation=_cancellation,
                    token=token,
                    live_events=handle,
                )
            await handle.publish(self._terminal_event(outcome, context.run_id))
        finally:
            await handle.close()

    @contextlib.asynccontextmanager
    async def _claim_and_fence(
        self, context: Any, *, cancellation: "CancellationToken | None" = None
    ):
        """Claim the execution (fencing token), register the driving task +
        cancellation token with the RunController, and start a heartbeat task
        that renews the claim. Lost fencing (heartbeat failure) cancels the
        token so execute_pure's next boundary check converges to AgentCancelled
        and the terminal commit is fenced out. Yields (cancellation, token,
        live_events_sink_stub) -- the live sink is owned by the caller."""
        if cancellation is None:
            cancellation = CancellationToken()
        run_id = context.run_id
        token = uuid.uuid4().hex
        worker_id = f"agent-worker:{uuid.uuid4().hex}"
        claim = getattr(self._storage.runs, "claim_execution", None)
        if claim is not None:
            await claim(run_id, worker_id=worker_id, execution_token=token)
        heartbeat = getattr(self._storage.runs, "heartbeat_execution", None)
        heartbeat_task: "asyncio.Task | None" = None
        if heartbeat is not None and claim is not None:
            heartbeat_task = asyncio.create_task(
                self._heartbeat(run_id, worker_id, token, cancellation)
            )
        if self._run_controller is not None:
            current = asyncio.current_task()
            if current is not None:
                await self._run_controller.register(run_id, current, cancellation)
        try:
            yield cancellation, token, None
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            if self._run_controller is not None:
                await self._run_controller.unregister(run_id)

    async def _heartbeat(
        self,
        run_id: str,
        worker_id: str,
        token: str,
        cancellation: CancellationToken,
    ) -> None:
        """Renew the execution claim on a fixed interval. A renewal failure
        means fencing was lost -- cancel the engine task so a stale worker
        cannot commit a terminal side effect."""
        heartbeat = getattr(self._storage.runs, "heartbeat_execution", None)
        if heartbeat is None:
            return
        while True:
            await self._clock.sleep(_HEARTBEAT_INTERVAL_SECONDS)
            if cancellation.is_cancelled():
                return
            try:
                await heartbeat(run_id, worker_id=worker_id, execution_token=token)
            except Exception:  # noqa: BLE001 - lost fencing, stop renewing
                cancellation.cancel()
                return

    async def _execute_and_commit(
        self,
        compiled: CompiledAgent,
        context: Any,
        run_input: RunInput,
        *,
        resuming: bool,
        message_history: tuple,
        running_version: "int | None",
        cancellation: CancellationToken,
        token: str,
        live_events: Any,
    ) -> AgentExecutionOutcome:
        """Load session history (new-run only), build AgentInput, drive
        execute_pure, and commit the resulting outcome via commit_coordinator
        (complete/pause/fail/cancel). The fencing token is carried on every
        terminal command so a stale worker's commit is rejected."""
        security_events = self._build_security_event_sink(context, token)
        if not resuming:
            message_history = await self._session_reader.load_message_history(
                context.session_id
            )
            message_history = tuple(message_history)
        agent_input = AgentInput(
            prompt=run_input.prompt,
            metadata=run_input.metadata,
            message_history=message_history,
            resuming=resuming,
        )
        outcome = await self._agent_engine.execute_pure(
            compiled,
            agent_input,
            context,
            cancellation=cancellation,
            live_events=live_events,
            security_events=security_events,
        )
        await self._commit_outcome(context, outcome, token, running_version)
        return outcome

    def _build_security_event_sink(self, context: Any, token: str) -> Any:
        """Construct the per-execution SecurityEventSink the engine consumes.

        When a FencedRunEventWriter is wired (production), every event routes
        through it -- the writer reads the RunRecord, verifies the presented
        fence matches the stored execution_token, and only then appends. A
        stale fence raises RunFenceLostError BEFORE the event lands, so the
        security-sensitive action that triggered the emit does NOT proceed.

        When no writer is wired (local-mode / tests), the sink appends
        directly without fencing -- the unfenced legacy path."""
        ctx = EventStreamContext.from_run_context(context)
        if self._fenced_event_writer is None or not token:
            return _EventStoreEventSink(self._storage.events, context)
        return _FencedSecurityEventSink(
            self._fenced_event_writer, ctx, ExecutionFence(token)
        )

    async def _commit_outcome(
        self,
        context: Any,
        outcome: AgentExecutionOutcome,
        token: str,
        running_version: "int | None",
    ) -> None:
        """Converge a terminal outcome to its commit command. The expected
        version is read LIVE from the store (not the stale ``running_version``
        from start/resume): a concurrent cancel may have moved the record
        RUNNING -> CANCELLING between drive-start and commit, so the version
        captured at start would conflict. The execution-token fencing the
        commit_coordinator enforces is the real cross-worker guard; the
        expected_version is only the optimistic CAS against current state."""
        run_id = context.run_id
        event_ctx = EventStreamContext.from_run_context(context)
        current = await self._storage.runs.get(run_id)
        expected_version = current.version if current is not None else (running_version or 0)
        if isinstance(outcome, AgentCompleted):
            await self._commit_coordinator.complete(
                CompleteRunCommand(
                    run_id=run_id,
                    session_id=context.session_id,
                    expected_version=expected_version,
                    messages=tuple(outcome.messages),
                    checkpoint_payload=outcome.checkpoint_payload,
                    result=outcome.result,
                    completed_event=RunCompletedEvent(run_id=run_id),
                    event_context=event_ctx,
                    execution_fence=ExecutionFence(token) if token else None,
                    commit_id=RunCommitId(f"complete:{run_id}"),
                )
            )
        elif isinstance(outcome, AgentPaused):
            await self._commit_coordinator.pause(
                PauseRunCommand(
                    run_id=run_id,
                    expected_version=expected_version,
                    approval_request=ApprovalRequestData(
                        tenant_id=context.tenant_id or "local",
                        approval_id=outcome.request.approval_id,
                        tool_call_id=outcome.request.tool_call_id,
                        tool_name=outcome.request.tool_name or "",
                        reason=outcome.request.reason or "",
                        arguments=outcome.request.arguments,
                        binding=outcome.request.binding,
                    ),
                    checkpoint_payload=outcome.checkpoint_payload,
                    paused_event=RunPausedEvent(
                        run_id=run_id,
                        reason=(
                            outcome.request.reason
                            or f"approval required: {outcome.request.approval_id}"
                        ),
                    ),
                    event_context=event_ctx,
                    execution_fence=ExecutionFence(token) if token else None,
                    commit_id=RunCommitId(f"pause:{run_id}:{outcome.request.approval_id}"),
                )
            )
        elif isinstance(outcome, AgentFailed):
            await self._commit_coordinator.fail(
                FailRunCommand(
                    run_id=run_id,
                    expected_version=expected_version,
                    execution_fence=ExecutionFence(token) if token else None,
                    error=outcome.error,
                    failed_event=RunFailedEvent(
                        run_id=run_id,
                        error_type=outcome.error.error_type,
                        message=outcome.error.message,
                    ),
                    event_context=event_ctx,
                    commit_id=RunCommitId(f"fail:{run_id}"),
                )
            )
        elif isinstance(outcome, AgentCancelled):
            await self._commit_coordinator.acknowledge_cancel(
                AcknowledgeCancelRunCommand(
                    run_id=run_id,
                    expected_version=expected_version,
                    execution_fence=ExecutionFence(token) if token else None,
                    cancelled_event=RunCancelledEvent(run_id=run_id),
                    event_context=event_ctx,
                    commit_id=RunCommitId(f"ack-cancel:{run_id}"),
                )
            )
        else:  # pragma: no cover - discriminated union is exhaustive
            raise SwarmError(f"unknown agent execution outcome: {type(outcome).__name__}")

    @staticmethod
    def _result_from_outcome(outcome: AgentExecutionOutcome, run_id: str) -> Any:
        """The non-streaming run() contract: return RunResult on completion;
        raise the control-flow signal a non-completed outcome represents so the
        caller (Runtime.run) observes it -- RunPaused for a pause,
        asyncio.CancelledError for a controller/external cancellation, and a
        reconstructed exception for a failure (the run is already FAILED in the
        store; the caller still needs to see that the run did not succeed)."""
        import asyncio as _asyncio

        from ..errors import RunPaused

        if isinstance(outcome, AgentCompleted):
            return outcome.result
        if isinstance(outcome, AgentPaused):
            raise RunPaused(
                run_id=run_id,
                approval_id=outcome.request.approval_id,
                tool_call_id=outcome.request.tool_call_id,
                tool_name=outcome.request.tool_name,
                reason=outcome.request.reason,
                arguments=dict(outcome.request.arguments),
                binding=dict(outcome.request.binding),
            )
        if isinstance(outcome, AgentCancelled):
            raise _asyncio.CancelledError(f"run {run_id} cancelled")
        # AgentFailed: the precise exception type is not carried on the outcome,
        # so raise a RuntimeError carrying the redacted message. The typed error
        # is already persisted on the RunRecord.
        raise RuntimeError(
            f"run {run_id} failed ({outcome.error.error_type}): {outcome.error.message}"
        )

    @staticmethod
    def _terminal_event(outcome: AgentExecutionOutcome, run_id: str = "") -> dict:
        if isinstance(outcome, AgentCompleted):
            return {"type": "completed", "run_id": run_id}
        if isinstance(outcome, AgentPaused):
            return {
                "type": "paused",
                "run_id": run_id,
                "approval_id": outcome.request.approval_id,
            }
        if isinstance(outcome, AgentFailed):
            return {
                "type": "failed",
                "run_id": run_id,
                "error_type": outcome.error.error_type,
                "message": outcome.error.message,
            }
        return {"type": "cancelled", "run_id": run_id}
