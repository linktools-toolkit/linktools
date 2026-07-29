"""Application orchestration for Run execution.

ExecutionService wires a compiled Agent through the AgentEngine and persists the
lifecycle transitions. Agent construction goes exclusively through
``AgentCompiler.compile`` (never a direct ``PydanticAgent``), and the
RunDefinition is encoded/decoded by ``AgentSpecCodec`` so every spec field --
tools, middleware, output type -- round-trips across pause/resume.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from uuid import uuid4

from ..agent.codec import AgentSpecCodec
from ..agent.compiler import AgentCompiler
from ..agent.engine import AgentEngine
from ..agent.models import AgentCancelled, AgentCompleted, AgentFailed, AgentInput, AgentPaused
from ..agent.spec import AgentSpec
from ..errors import PrincipalAccessDeniedError, StorageError
from ..json import canonical_json_bytes
from .commands import AbortExecution, AcknowledgeCancellation, ClaimExecution, CompleteExecution, DecideApproval, FailExecution, HeartbeatExecution, PauseExecution, RequestCancellation, ResumeExecution, StartExecution
from .domain import ApprovalDecision, RunApproval, RunDefinition, RunError, RunKind, RunRecord, RunStatus, RunnableType
from .context import RunContext
from .cancellation import CancellationToken
from .controller import RunController
from .query import ExecutionResultView
from .session import SessionRecord
from .snapshots import AgentSnapshotData
from .store import ExecutionStore
from . import trace_codec
from .trace_collector import SemanticTraceCollector


class _Events:
    async def emit(self, event: object) -> None:
        return None

    async def publish(self, event: object) -> None:
        return None


def _definition(spec: AgentSpec, codec: AgentSpecCodec) -> RunDefinition:
    value = codec.encode(spec)
    return RunDefinition(spec.id, RunnableType.AGENT, "agent-spec.v1", value, sha256(canonical_json_bytes(value)).hexdigest())


def _snapshot(outcome: object) -> AgentSnapshotData:
    if isinstance(outcome, (AgentCompleted, AgentPaused, AgentCancelled, AgentFailed)):
        snapshot = outcome.snapshot
    else:
        snapshot = None
    if not isinstance(snapshot, AgentSnapshotData):
        raise StorageError("agent outcome did not provide a canonical snapshot")
    return snapshot


def decode_model_messages(messages: tuple[object, ...]) -> tuple[object, ...]:
    return trace_codec.decode_model_messages(messages)


_LEASE_DURATION = timedelta(minutes=5)
_HEARTBEAT_INTERVAL = min(max(_LEASE_DURATION / 3, timedelta(seconds=1)), timedelta(seconds=10))


def _ownership_mismatch(owned: "SessionRecord | RunRecord", user_id: str | None, tenant_id: str | None) -> bool:
    # An unowned session/run (None principal) stays single-principal and
    # reusable; only a principal that conflicts with the recorded owner is
    # rejected. Shared by session reuse (run()) and by resume()/cancel()/
    # decide_approval() so a caller who knows a run_id cannot act on another
    # tenant's run.
    if user_id is not None and owned.user_id is not None and owned.user_id != user_id:
        return True
    if tenant_id is not None and owned.tenant_id is not None and owned.tenant_id != tenant_id:
        return True
    return False


class ExecutionService:
    def __init__(self, store: ExecutionStore, compiler: AgentCompiler, *, spec_codec: AgentSpecCodec | None = None, controller: RunController | None = None) -> None:
        self.store = store
        self._compiler = compiler
        self._codec = spec_codec or AgentSpecCodec()
        # In-process registry of the currently executing runs' driving
        # asyncio.Task + CancellationToken, so cancel() can actually interrupt
        # a suspended model/tool await (task.cancel()), not just flip the
        # token that's only checked between execution points.
        self._controller = controller or RunController()

    async def run(self, spec: AgentSpec, prompt: str, *, session_id: str, run_id: str | None = None, user_id: str | None = None, tenant_id: str | None = None) -> object:
        session = await self.store.get_session(session_id)
        if session is None:
            await self.store.create_session(session_id=session_id, user_id=user_id, tenant_id=tenant_id)
        elif _ownership_mismatch(session, user_id, tenant_id):
            raise PrincipalAccessDeniedError("session is not owned by this principal")
        record = await self.store.start_run(StartExecution(run_id or uuid4().hex, session_id, RunKind.USER_TURN, _definition(spec, self._codec), prompt))
        # Claiming the just-started run and loading the session's latest
        # completed snapshot (so the next turn sees the prior turn's context;
        # resume uses the target run's own snapshot, not this path) don't
        # depend on each other -- run concurrently.
        claimed, messages = await asyncio.gather(
            self.store.claim_run(ClaimExecution(record.id, "runtime", datetime.now(timezone.utc), _LEASE_DURATION)),
            self.store.load_session_context(session_id),
        )
        return await self._execute(spec, prompt, claimed, resuming=False, message_history=messages)

    async def resume(self, run_id: str, *, user_id: str | None = None, tenant_id: str | None = None) -> object:
        record = await self._required(run_id)
        if _ownership_mismatch(record, user_id, tenant_id):
            raise PrincipalAccessDeniedError("run is not owned by this principal")
        spec = self._codec.decode(record.definition.spec)
        # A paused run must transition PAUSED -> PENDING (its approval already
        # decided) before it can be claimed; claiming PAUSED directly is not
        # claimable. Resume then restores the target run's OWN snapshot, not the
        # session's latest completed snapshot.
        await self.store.resume_run(ResumeExecution(run_id))
        # Claiming the now-PENDING run and loading its own snapshot don't
        # depend on each other -- run concurrently.
        claimed, snapshot = await asyncio.gather(
            self.store.claim_run(ClaimExecution(run_id, "runtime", datetime.now(timezone.utc), _LEASE_DURATION)),
            self.store.get_snapshot(run_id),
        )
        messages = snapshot.resume_messages if snapshot is not None else ()
        return await self._execute(spec, "", claimed, resuming=True, message_history=messages)

    async def cancel(self, run_id: str, *, user_id: str | None = None, tenant_id: str | None = None) -> None:
        record = await self._required(run_id)
        if _ownership_mismatch(record, user_id, tenant_id):
            raise PrincipalAccessDeniedError("run is not owned by this principal")
        if record.status in {RunStatus.PENDING, RunStatus.PAUSED}:
            # PENDING/PAUSED cancel is terminal: no live execution to signal,
            # no lease to release (PENDING was never claimed; PAUSED lease was
            # already released by _finish). The store transitions directly
            # to CANCELLED.
            await self.store.request_cancel(RequestCancellation(run_id, record.lease.owner or "runtime", record.lease.fence, datetime.now(timezone.utc)))
            return
        if record.lease.owner is None:
            raise StorageError("run has no active owner")
        await self.store.request_cancel(RequestCancellation(run_id, record.lease.owner, record.lease.fence, datetime.now(timezone.utc)))
        # Persisted CANCELLING; also signal the live execution -- both the
        # token (checked between execution points) and task.cancel() (unblocks
        # a currently-suspended await, e.g. a hanging model call).
        await self._controller.cancel(run_id)

    async def decide_approval(self, run_id: str, *, approval_id: str, decision: ApprovalDecision, decided_by: str, user_id: str | None = None, tenant_id: str | None = None) -> RunRecord:
        record = await self._required(run_id)
        if _ownership_mismatch(record, user_id, tenant_id):
            raise PrincipalAccessDeniedError("run is not owned by this principal")
        # ALLOW leaves the run PAUSED (resumable via resume()); DENY is
        # terminal -- the store transitions it straight to CANCELLED and no
        # further resume is possible.
        return await self.store.decide_approval(DecideApproval(run_id, approval_id, decision, decided_by))

    async def _heartbeat(self, run_id: str, owner: str, fence: int, token: CancellationToken) -> None:
        # Renews the lease periodically so a still-RUNNING execution that
        # outlives one lease period isn't mistaken by another worker for an
        # abandoned one. Also checks the cancel flag on each tick: if another
        # process set CANCELLING, the local token is cancelled immediately.
        # If the renewal fails (lease lost), same: cancel and stop.
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL.total_seconds())
            try:
                updated = await self.store.heartbeat_run(HeartbeatExecution(run_id, owner, fence, datetime.now(timezone.utc), _LEASE_DURATION))
                if updated.status is RunStatus.CANCELLING:
                    token.cancel()
                    return
            except StorageError:
                token.cancel()
                return

    async def _execute(self, spec: AgentSpec, prompt: str, record: RunRecord, *, resuming: bool, message_history: tuple[object, ...] = ()) -> object:
        owner = record.lease.owner or "runtime"
        compiled = await self._compiler.compile(spec)
        context = RunContext(record.id, record.root_run_id, record.parent_run_id, record.session_id, record.runnable_id, record.definition.runnable_type, record.user_id, record.tenant_id, None)
        collector = SemanticTraceCollector(record.id, self.store, record.trace_sequence)
        engine = AgentEngine(trace_collector=collector, trace_codec=trace_codec)
        decoded_history = decode_model_messages(message_history) if message_history else ()
        token = CancellationToken()
        task = asyncio.ensure_future(engine.execute_pure(compiled, AgentInput(prompt=prompt, message_history=decoded_history, resuming=resuming), context, cancellation=token, live_events=_Events(), security_events=_Events(), trace_sequence=record.trace_sequence))
        await self._controller.register(record.id, task, token)
        heartbeat = asyncio.ensure_future(self._heartbeat(record.id, owner, record.lease.fence, token))
        try:
            outcome = await task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A programming/config/protocol error (as opposed to a modeled
            # AgentFailed) must not strand the run in RUNNING: persist a
            # minimal FAILED state -- no snapshot, since the engine never
            # produced a coherent outcome -- then re-raise the original
            # exception. A failure here is secondary -- the original error is
            # what the caller must see.
            try:
                await collector.flush()
                await self.store.abort_run(AbortExecution(record.id, owner, record.lease.fence, RunError(type(exc).__name__, str(exc)), collector.next_sequence))
            except Exception:
                pass
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            await self._controller.unregister(record.id)
        snapshot = _snapshot(outcome)
        if isinstance(outcome, AgentCompleted):
            await self.store.complete_run(CompleteExecution(record.id, owner, record.lease.fence, snapshot))
            return ExecutionResultView(record.id, outcome.result.output)
        if isinstance(outcome, AgentPaused):
            approval = outcome.request
            await self.store.pause_run(PauseExecution(record.id, owner, record.lease.fence, snapshot, RunApproval(approval.approval_id, approval.tool_call_id or "", approval.tool_name or "", dict(approval.arguments))))
            return None
        if isinstance(outcome, AgentCancelled):
            await self.store.acknowledge_cancel(AcknowledgeCancellation(record.id, owner, record.lease.fence, snapshot))
            return None
        if isinstance(outcome, AgentFailed):
            await self.store.fail_run(FailExecution(record.id, owner, record.lease.fence, snapshot, outcome.error))
            raise RuntimeError(outcome.error.message)
        raise AssertionError(f"unsupported agent outcome: {type(outcome).__name__}")

    async def _required(self, run_id: str) -> RunRecord:
        record = await self.store.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record


def spec_type(record: RunRecord) -> RunnableType:
    return record.definition.runnable_type


__all__ = ["ExecutionService", "spec_type"]
