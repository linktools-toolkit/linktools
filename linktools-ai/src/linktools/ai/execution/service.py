#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Application orchestration for Run execution.

ExecutionService wires a compiled Agent through the AgentEngine and persists the
lifecycle transitions. Agent construction goes exclusively through
``AgentCompiler.compile`` (never a direct ``PydanticAgent``), and the
RunDefinition is encoded/decoded by ``AgentSpecCodec`` so every spec field --
tools, middleware, output type -- round-trips across pause/resume.
"""


import asyncio
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Mapping
from uuid import uuid4

from linktools.core import environ

from ..agent.assembly.provider import AgentFeatureContext
from ..agent.models import AgentCancelled, AgentCompleted, AgentFailed, AgentInput, AgentPaused
from ..agent.sandbox.protocols import Sandbox
from ..errors import RunDefinitionError, RunDefinitionIntegrityError, RuntimeInitializationError, StorageError
from ..governance.authorization import ExecutionAction
from ..governance.identity import PrincipalContext
from ..json import canonical_json_bytes
from ..observability.events.payloads import SecurityDegraded
from .commands import AbortExecution, AcknowledgeCancellation, ClaimExecution, CompleteExecution, DecideApproval, FailExecution, HeartbeatExecution, PauseExecution, RequestCancellation, ResumeExecution, StartExecution
from .domain import MessageCaptureState, RunApproval, RunDefinition, RunError, RunKind, RunStatus, RunnableType, RunUsage, sanitize_run_error
from .context import RunContext
from .cancellation import CancellationToken
from .controller import ExecutionControllerRegistry
from .query import ExecutionResultView
from .snapshots import AgentSnapshotData, RunUsageCapture
from . import trace_codec
from .trace_collector import SemanticTraceCollector

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.codec import AgentSpecCodec
    from ..agent.compiler import AgentCompiler
    from ..agent.engine import AgentEngine
    from ..agent.assembly.assembler import AgentAssembler
    from ..agent.assembly.models import AgentAssembly
    from ..agent.spec import AgentSpec
    from ..governance.authorization import AuthorizationPolicy
    from ..tasks.models import TaskUsage
    from .domain import ApprovalDecision, RunRecord
    from .live_events import RunLiveEventSink, SecurityEventSink
    from .store import ExecutionStore

logger = environ.get_logger("ai.execution.service")


def _definition(spec: "AgentSpec", codec: "AgentSpecCodec") -> RunDefinition:
    value = codec.encode(spec)
    return RunDefinition(spec.id, RunnableType.AGENT, "agent-spec.v1", value, sha256(canonical_json_bytes(value)).hexdigest())


def _decode_definition(
    definition: RunDefinition,
    codec: "AgentSpecCodec",
) -> "AgentSpec":
    if definition.schema != "agent-spec.v1":
        raise RunDefinitionError("unsupported definition schema")
    actual = sha256(canonical_json_bytes(definition.spec)).hexdigest()
    if not hmac.compare_digest(actual, definition.spec_hash):
        raise RunDefinitionIntegrityError("definition hash mismatch")
    return codec.decode(definition.spec)


def _snapshot(outcome: object) -> AgentSnapshotData:
    if isinstance(outcome, (AgentCompleted, AgentPaused, AgentCancelled, AgentFailed)):
        snapshot = outcome.snapshot
    else:
        snapshot = None
    if not isinstance(snapshot, AgentSnapshotData):
        raise StorageError("agent outcome did not provide a canonical snapshot")
    return snapshot


def decode_model_messages(messages: "tuple[object, ...]") -> "tuple[object, ...]":
    return trace_codec.decode_model_messages(messages)


_LEASE_DURATION = timedelta(minutes=5)
_HEARTBEAT_INTERVAL = min(max(_LEASE_DURATION / 3, timedelta(seconds=1)), timedelta(seconds=10))


@dataclass(frozen=True, slots=True)
class ChildRunResult:
    """Terminal outcome of a swarm child agent run: the persisted status, the
    structured output (when completed), the redacted error (when failed), and
    the real usage the child consumed. A swarm maps this to a NodeRunResult."""

    run_id: str
    status: "RunStatus"
    output: "object | None"
    error: "RunError | None"
    usage: "TaskUsage"


@dataclass(frozen=True, slots=True)
class PreparedAgentExecution:
    agent_spec: "AgentSpec"
    assembled_agent: object
    tool_descriptors: "tuple[object, ...]"
    fingerprint: str


def _outcome_usage(outcome: "object | None", record: "RunRecord | None") -> "TaskUsage":
    """Pull the real token/cost usage out of an AgentExecutionOutcome (or fall
    back to the persisted record's RunUsage). Cost stays None unless every agent
    reports it; unknown cost is NEVER coerced to zero."""
    from ..tasks.models import TaskUsage
    from decimal import Decimal

    if isinstance(outcome, (AgentCompleted, AgentFailed, AgentCancelled)):
        usage = outcome.usage
        total_cost: "Decimal | None" = None
        if getattr(usage, "total_cost", None) is not None:
            total_cost = Decimal(str(usage.total_cost))
        return TaskUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_cost=total_cost,
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0),
        )
    if record is not None:
        return TaskUsage(input_tokens=0, output_tokens=0, total_cost=None)
    from ..tasks.models import TaskUsage as _TU

    return _TU()


def _task_usage_from_run_usage(usage: RunUsage) -> "TaskUsage":
    from ..tasks.models import TaskUsage

    return TaskUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_cost=usage.total_cost,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
    )


def _child_result_from_outcome(
    run_id: str, outcome: "object | None", record: "RunRecord"
) -> "ChildRunResult":
    output: "object | None" = None
    error: "RunError | None" = record.error
    if isinstance(outcome, AgentCompleted):
        output = outcome.result.output
    usage = _outcome_usage(outcome, record)
    return ChildRunResult(run_id=run_id, status=record.status, output=output, error=error, usage=usage)


async def _child_result_from_persistence(
    store: "ExecutionStore",
    run_id: str,
    outcome: "object | None",
    record: "RunRecord",
) -> "ChildRunResult":
    snapshot = await store.get_snapshot(run_id)
    if snapshot is None:
        return _child_result_from_outcome(run_id, outcome, record)
    return ChildRunResult(
        run_id=run_id,
        status=record.status,
        output=snapshot.final_output if record.status is RunStatus.COMPLETED else None,
        error=record.error,
        usage=_task_usage_from_run_usage(snapshot.usage),
    )


class ExecutionService:
    def __init__(
        self,
        store: "ExecutionStore",
        compiler: "AgentCompiler",
        *,
        engine: "AgentEngine",
        assembler: "AgentAssembler",
        tool_execution_ready: bool,
        sandbox: "Sandbox | None",
        spec_codec: "AgentSpecCodec",
        authorization: "AuthorizationPolicy",
        live_events: "RunLiveEventSink",
        security_events: "SecurityEventSink",
        controller: "ExecutionControllerRegistry | None" = None,
    ) -> None:
        self._store = store
        self._compiler = compiler
        self._engine = engine
        self._assembler = assembler
        self._tool_execution_ready = tool_execution_ready
        self._sandbox = sandbox
        self._codec = spec_codec
        self._authorization = authorization
        self._live_events = live_events
        self._security_events = security_events
        # In-process registry of the currently executing runs' driving
        # asyncio.Task + CancellationToken, so cancel() can actually interrupt
        # a suspended model/tool await (task.cancel()), not just flip the
        # token that's only checked between execution points.
        self._controller = controller or ExecutionControllerRegistry()

    async def run(
        self,
        spec: "AgentSpec",
        prompt: str,
        *,
        principal: PrincipalContext,
        session_id: "str | None" = None,
        execution_id: "str | None" = None,
    ) -> object:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        session_id = session_id or uuid4().hex
        execution_id = execution_id or uuid4().hex
        assembly = await self._preflight(
            spec,
            execution_id=execution_id,
            session_id=session_id,
            root_execution_id=execution_id,
            parent_execution_id=None,
            principal=principal,
        )
        session = await self._store.create_session(
            session_id=session_id,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
        )
        self._authorization.assert_session_access(
            principal=principal,
            session=session,
        )
        record = await self._store.start_run(StartExecution(execution_id, session_id, RunKind.USER_TURN, _definition(spec, self._codec), prompt))
        # Claiming the just-started run and loading the session's latest
        # completed snapshot (so the next turn sees the prior turn's context;
        # resume uses the target run's own snapshot, not this path) don't
        # depend on each other -- run concurrently.
        claimed, messages = await asyncio.gather(
            self._store.claim_run(ClaimExecution(record.id, "runtime", datetime.now(timezone.utc), _LEASE_DURATION)),
            self._store.load_session_context(session_id),
        )
        if environ.debug:
            logger.debug(
                "run %s claimed (session=%s owner=%s fence=%s history=%s)",
                claimed.id, session_id, claimed.lease.owner, claimed.lease.fence, len(messages),
            )
        return await self._execute(
            spec,
            prompt,
            claimed,
            resuming=False,
            message_history=messages,
            assembly=assembly,
        )

    async def resume(self, run_id: str, *, principal: PrincipalContext) -> object:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        record = await self._required(run_id)
        self._authorize(principal, record, ExecutionAction.RESUME)
        spec = _decode_definition(record.definition, self._codec)
        assembly = await self._preflight(
            spec,
            execution_id=record.id,
            session_id=record.session_id,
            root_execution_id=record.root_execution_id,
            parent_execution_id=record.parent_execution_id,
            principal=principal,
        )
        # A paused run must transition PAUSED -> PENDING (its approval already
        # decided) before it can be claimed; claiming PAUSED directly is not
        # claimable. Resume then restores the target run's OWN snapshot, not the
        # session's latest completed snapshot.
        await self._store.resume_run(ResumeExecution(run_id))
        # Claiming the now-PENDING run and loading its own snapshot don't
        # depend on each other -- run concurrently.
        claimed, messages = await asyncio.gather(
            self._store.claim_run(ClaimExecution(run_id, "runtime", datetime.now(timezone.utc), _LEASE_DURATION)),
            self._store.load_resume_messages(run_id),
        )
        return await self._execute(
            spec,
            "",
            claimed,
            resuming=True,
            message_history=messages,
            assembly=assembly,
        )

    async def run_child(
        self,
        spec: "AgentSpec",
        prompt: str,
        *,
        principal: PrincipalContext,
        session_id: str,
        execution_id: str,
        root_execution_id: str,
        parent_execution_id: str,
        message_history: "tuple[object, ...]" = (),
        metadata: "Mapping[str, Any] | None" = None,
        prepared_execution: "PreparedAgentExecution | None" = None,
    ) -> "ChildRunResult":
        """Run one agent as a swarm child: a TASK run that propagates the parent
        and root execution ids, reads the parent's immutable message snapshot,
        and never claims a user turn or writes the shared session. The child's
        own ``AgentInput.metadata`` carries the task_graph dependency view so
        the agent sees upstream results without session cross-talk."""
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        if prepared_execution is not None and prepared_execution.agent_spec.id != spec.id:
            raise RuntimeInitializationError("agent_preflight_failed")
        assembly = (
            prepared_execution.assembled_agent
            if prepared_execution is not None
            else await self._preflight(
                spec,
                execution_id=execution_id,
                session_id=session_id,
                root_execution_id=root_execution_id,
                parent_execution_id=parent_execution_id,
                principal=principal,
            )
        )
        record = await self._store.start_run(
            StartExecution(
                execution_id,
                session_id,
                RunKind.TASK,
                _definition(spec, self._codec),
                {"prompt": prompt, "metadata": dict(metadata) if metadata else {}},
                root_execution_id=root_execution_id,
                parent_execution_id=parent_execution_id,
            )
        )
        claimed = await self._store.claim_run(
            ClaimExecution(record.id, "swarm", datetime.now(timezone.utc), _LEASE_DURATION)
        )
        compiled = await self._compiler.compile(spec)
        context = RunContext(
            claimed.id,
            claimed.root_execution_id,
            claimed.parent_execution_id,
            claimed.session_id,
            claimed.runnable_id,
            claimed.definition.runnable_type,
            claimed.user_id,
            claimed.tenant_id,
            None,
            metadata=metadata or {},
        )
        collector = SemanticTraceCollector(claimed.id, self._store, claimed.trace_sequence)
        usage_capture = RunUsageCapture()
        decoded_history = decode_model_messages(message_history) if message_history else ()
        token = CancellationToken()
        agent_input = AgentInput(prompt=prompt, message_history=decoded_history, metadata=metadata or {})
        owner = claimed.lease.owner or "swarm"
        task = await self._controller.start(
            claimed.id,
            self._engine.execute_pure(
                compiled,
                agent_input,
                context,
                cancellation=token,
                live_events=self._live_events,
                security_events=self._security_events,
                assembly=assembly,
                trace_sequence=claimed.trace_sequence,
                trace_collector=collector,
                usage_sink=usage_capture,
            ),
            token,
        )
        heartbeat = asyncio.ensure_future(
            self._heartbeat(claimed.id, owner, claimed.lease.fence, token)
        )
        outcome: "object | None" = None
        run_error: "RunError | None" = None
        try:
            done, _ = await asyncio.wait({task, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat_error = heartbeat.exception()
                if heartbeat_error is not None:
                    await self._controller.cancel(claimed.id)
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise heartbeat_error
            outcome = await task
        except asyncio.CancelledError:
            token.cancel()
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            latest = await self._required(execution_id)
            await self._converge_child_cancel(
                claimed, owner, _task_usage_from_run_usage(usage_capture.snapshot())
            )
            return ChildRunResult(
                run_id=execution_id,
                status=RunStatus.CANCELLED,
                output=None,
                error=latest.error,
                usage=_task_usage_from_run_usage(usage_capture.snapshot()),
            )
        except Exception as exc:
            trace_end = await collector.flush()
            await self._store.abort_run(
                AbortExecution(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    AgentSnapshotData(
                        delta_messages=(),
                        final_output=None,
                        usage=usage_capture.snapshot(),
                        trace_end_sequence=trace_end,
                        capture_state=MessageCaptureState.PARTIAL,
                    ),
                    RunError(type(exc).__name__, "child execution failed"),
                )
            )
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            await self._controller.unregister(claimed.id, task=task)
        snapshot = _snapshot(outcome) if outcome is not None else None
        latest = await self._required(execution_id)
        if latest.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return await _child_result_from_persistence(
                self._store, execution_id, outcome, latest
            )
        if isinstance(outcome, AgentCompleted):
            await self._store.complete_run(
                CompleteExecution(claimed.id, owner, claimed.lease.fence, snapshot)
            )
        elif isinstance(outcome, AgentPaused):
            # task_graph does not support approval: a paused child is a failure.
            await self._store.fail_run(
                FailExecution(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    snapshot,
                    RunError("approval_not_supported", "task_graph child paused for approval"),
                )
            )
            latest = await self._required(execution_id)
            return await _child_result_from_persistence(
                self._store, execution_id, outcome, latest
            )
        elif isinstance(outcome, AgentCancelled):
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(claimed.id, owner, claimed.lease.fence, snapshot)
            )
        elif isinstance(outcome, AgentFailed):
            await self._store.fail_run(
                FailExecution(claimed.id, owner, claimed.lease.fence, snapshot, outcome.error)
            )
        latest = await self._required(execution_id)
        return await _child_result_from_persistence(
            self._store, execution_id, outcome, latest
        )

    async def _converge_child_cancel(
        self, record: "RunRecord", owner: str, usage: "TaskUsage"
    ) -> None:
        latest = await self._required(record.id)
        if latest.status is RunStatus.RUNNING:
            await self._store.request_cancel(
                RequestCancellation(record.id, owner, record.lease.fence, datetime.now(timezone.utc))
            )
            latest = await self._required(record.id)
        if latest.status is RunStatus.CANCELLING:
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    record.id, owner, record.lease.fence,
                    AgentSnapshotData(
                        (), None, RunUsage(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            total_tokens=usage.total_tokens,
                            total_cost=usage.total_cost,
                            cache_write_tokens=usage.cache_write_tokens,
                            cache_read_tokens=usage.cache_read_tokens,
                        ), record.trace_sequence, MessageCaptureState.PARTIAL
                    ),
                )
            )

    async def _preflight(
        self,
        spec: "AgentSpec",
        *,
        execution_id: str,
        session_id: str,
        root_execution_id: str,
        parent_execution_id: "str | None",
        principal: PrincipalContext,
    ) -> "AgentAssembly":
        self._assembler.validate_features(spec)
        assembly = await self._assembler.assemble(
            spec,
            AgentFeatureContext(
                agent_id=spec.id,
                execution_id=execution_id,
                root_execution_id=root_execution_id,
                parent_execution_id=parent_execution_id,
                session_id=session_id,
                tenant_id=principal.tenant_id,
                user_id=principal.user_id,
                workspace=None,
                sandbox=self._sandbox,
            ),
        )
        if assembly.tools and not self._tool_execution_ready:
            raise RuntimeInitializationError(
                "agent tools require ToolStateStore and ToolPolicyResolver"
            )
        return assembly

    async def prepare_agent_execution(
        self,
        agent_spec: "AgentSpec",
        *,
        principal: PrincipalContext,
        session_id: str,
        execution_id: str,
        root_execution_id: str,
        parent_execution_id: "str | None",
    ) -> PreparedAgentExecution:
        """Assemble and freeze the exact tool surface used by a child run."""
        if not isinstance(principal, PrincipalContext):
            raise RuntimeInitializationError("agent_preflight_failed")
        try:
            assembly = await self._preflight(
                agent_spec,
                execution_id=execution_id,
                session_id=session_id,
                root_execution_id=root_execution_id,
                parent_execution_id=parent_execution_id,
                principal=principal,
            )
        except Exception as exc:
            if isinstance(exc, RuntimeInitializationError):
                raise
            raise RuntimeInitializationError("agent_preflight_failed") from exc
        try:
            descriptors = []
            for tool in getattr(assembly, "tools", ()):
                descriptor = getattr(tool, "descriptor", None)
                if descriptor is None:
                    raise RuntimeInitializationError("agent_preflight_failed")
                descriptors.append(descriptor)
                if descriptor.mutating:
                    raise RuntimeInitializationError("mutating_tool_not_allowed")
            encoded_spec = self._codec.encode(agent_spec)
            tool_fingerprints = [descriptor.fingerprint() for descriptor in descriptors]
        except RuntimeInitializationError:
            raise
        except Exception as exc:
            raise RuntimeInitializationError("agent_preflight_failed") from exc
        fingerprint = sha256(
            canonical_json_bytes(
                {
                    "agent": encoded_spec,
                    "tools": tool_fingerprints,
                }
            )
        ).hexdigest()
        return PreparedAgentExecution(
            agent_spec=agent_spec,
            assembled_agent=assembly,
            tool_descriptors=tuple(descriptors),
            fingerprint=fingerprint,
        )

    async def cancel(self, run_id: str, *, principal: PrincipalContext) -> None:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        record = await self._required(run_id)
        self._authorize(principal, record, ExecutionAction.CANCEL)
        if record.status in {RunStatus.PENDING, RunStatus.PAUSED}:
            # PENDING/PAUSED cancel is terminal: no live execution to signal,
            # no lease to release (PENDING was never claimed; PAUSED lease was
            # already released by _finish). The store transitions directly
            # to CANCELLED.
            await self._store.request_cancel(RequestCancellation(run_id, record.lease.owner or "runtime", record.lease.fence, datetime.now(timezone.utc)))
            return
        if record.lease.owner is None:
            raise StorageError("run has no active owner")
        await self._store.request_cancel(RequestCancellation(run_id, record.lease.owner, record.lease.fence, datetime.now(timezone.utc)))
        # Persisted CANCELLING; also signal the live execution -- both the
        # token (checked between execution points) and task.cancel() (unblocks
        # a currently-suspended await, e.g. a hanging model call).
        await self._controller.cancel(run_id)

    async def decide_approval(self, run_id: str, *, approval_id: str, decision: "ApprovalDecision", principal: PrincipalContext) -> "RunRecord":
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        record = await self._required(run_id)
        self._authorize(principal, record, ExecutionAction.DECIDE_APPROVAL)
        # ALLOW leaves the run PAUSED (resumable via resume()); DENY is
        # terminal -- the store transitions it straight to CANCELLED and no
        # further resume is possible.
        return await self._store.decide_approval(DecideApproval(run_id, approval_id, decision, principal.resolved_by))

    async def _heartbeat(self, run_id: str, owner: str, fence: int, token: CancellationToken) -> None:
        # Renews the lease periodically so a still-RUNNING execution that
        # outlives one lease period isn't mistaken by another worker for an
        # abandoned one. Also checks the cancel flag on each tick: if another
        # process set CANCELLING, the local token is cancelled immediately.
        # If the renewal fails (lease lost), same: cancel and stop.
        while True:
            await asyncio.sleep(_HEARTBEAT_INTERVAL.total_seconds())
            updated = await self._store.heartbeat_run(HeartbeatExecution(run_id, owner, fence, datetime.now(timezone.utc), _LEASE_DURATION))
            if updated.status is RunStatus.CANCELLING:
                if environ.debug:
                    logger.debug("run %s heartbeat detected CANCELLING", run_id)
                await self._controller.cancel(run_id)
                return

    async def _execute(self, spec: "AgentSpec", prompt: str, record: "RunRecord", *, resuming: bool, message_history: "tuple[object, ...]" = (), assembly: "AgentAssembly | None" = None) -> object:
        owner = record.lease.owner or "runtime"
        compiled = await self._compiler.compile(spec)
        context = RunContext(record.id, record.root_execution_id, record.parent_execution_id, record.session_id, record.runnable_id, record.definition.runnable_type, record.user_id, record.tenant_id, None)
        collector = SemanticTraceCollector(record.id, self._store, record.trace_sequence)
        decoded_history = decode_model_messages(message_history) if message_history else ()
        token = CancellationToken()
        approved = record.approval if resuming else None
        task = await self._controller.start(
            record.id,
            self._engine.execute_pure(compiled, AgentInput(prompt=prompt, message_history=decoded_history, resuming=resuming, approved_tool_call_id=approved.tool_call_id if approved is not None else None, approved_binding_fingerprint=approved.binding_fingerprint if approved is not None else None), context, cancellation=token, live_events=self._live_events, security_events=self._security_events, assembly=assembly, trace_sequence=record.trace_sequence, trace_collector=collector),
            token,
        )
        heartbeat = asyncio.ensure_future(self._heartbeat(record.id, owner, record.lease.fence, token))
        try:
            done, _ = await asyncio.wait(
                {task, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat in done:
                heartbeat_error = heartbeat.exception()
                if heartbeat_error is not None:
                    await self._controller.cancel(record.id)
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise heartbeat_error
            outcome = await task
        except asyncio.CancelledError:
            token.cancel()
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception) as task_exc:
                    if environ.debug:
                        logger.debug(
                            "run %s cancelled-task cleanup swallowed: %s: %s",
                            record.id, type(task_exc).__name__, task_exc,
                        )
            try:
                latest = await self._required(record.id)
                if latest.status is RunStatus.RUNNING:
                    await self._store.request_cancel(
                        RequestCancellation(
                            record.id,
                            owner,
                            record.lease.fence,
                            datetime.now(timezone.utc),
                        )
                    )
                    latest = await self._required(record.id)
                if latest.status is RunStatus.CANCELLING:
                    trace_end = await collector.flush()
                    await self._store.acknowledge_cancel(
                        AcknowledgeCancellation(
                            record.id,
                            owner,
                            record.lease.fence,
                            AgentSnapshotData(
                                delta_messages=(),
                                final_output=None,
                                usage=RunUsage(),
                                trace_end_sequence=trace_end,
                                capture_state=MessageCaptureState.PARTIAL,
                            ),
                        )
                    )
            except Exception as cleanup_error:
                try:
                    await asyncio.shield(
                        self._security_events.emit(
                            SecurityDegraded(
                                run_id=record.id,
                                component="execution_cancel_cleanup",
                                reason="execution cancel cleanup failed",
                                error_code=type(cleanup_error).__name__,
                            )
                        )
                    )
                except Exception as emit_exc:
                    if environ.debug:
                        logger.debug(
                            "run %s SecurityDegraded emit failed: %s: %s",
                            record.id, type(emit_exc).__name__, emit_exc,
                        )
            raise
        except Exception as exc:
            # A programming/config/protocol error (as opposed to a modeled
            # AgentFailed) must not strand the run in RUNNING: persist a
            # minimal FAILED state -- no snapshot, since the engine never
            # produced a coherent outcome -- then re-raise the original
            # exception. A failure here is secondary -- the original error is
            # what the caller must see.
            try:
                trace_end = await collector.flush()
                await self._store.abort_run(
                    AbortExecution(
                        record.id,
                        owner,
                        record.lease.fence,
                        AgentSnapshotData(
                            delta_messages=(),
                            final_output=None,
                            usage=RunUsage(),
                            trace_end_sequence=trace_end,
                            capture_state=MessageCaptureState.PARTIAL,
                        ),
                        sanitize_run_error(exc),
                    )
                )
            except Exception as abort_exc:
                logger.warning(
                    "run %s abort/flush failed (run may strand in RUNNING): %s: %s",
                    record.id, type(abort_exc).__name__, abort_exc,
                    exc_info=environ.debug,
                )
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            await self._controller.unregister(record.id, task=task)
        snapshot = _snapshot(outcome)
        latest = await self._required(record.id)
        if latest.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            return None
        if (
            latest.status is RunStatus.CANCELLING
            and not isinstance(outcome, AgentFailed)
        ):
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    record.id,
                    owner,
                    record.lease.fence,
                    snapshot,
                )
            )
            return None
        if isinstance(outcome, AgentCompleted):
            await self._store.complete_run(CompleteExecution(record.id, owner, record.lease.fence, snapshot))
            return ExecutionResultView(record.id, outcome.result.output)
        if isinstance(outcome, AgentPaused):
            approval = outcome.request
            binding_fingerprint = str(
                approval.binding.get("fingerprint", "")
                if approval.binding
                else ""
            )
            await self._store.pause_run(
                PauseExecution(
                    record.id,
                    owner,
                    record.lease.fence,
                    snapshot,
                    RunApproval(
                        approval.approval_id,
                        approval.tool_call_id or "",
                        approval.tool_name or "",
                        binding_fingerprint,
                    ),
                )
            )
            return None
        if isinstance(outcome, AgentCancelled):
            await self._store.acknowledge_cancel(AcknowledgeCancellation(record.id, owner, record.lease.fence, snapshot))
            return None
        if isinstance(outcome, AgentFailed):
            await self._store.fail_run(FailExecution(record.id, owner, record.lease.fence, snapshot, outcome.error))
            raise RuntimeError(outcome.error.message)
        raise AssertionError(f"unsupported agent outcome: {type(outcome).__name__}")

    async def _required(self, run_id: str) -> "RunRecord":
        record = await self._store.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def _authorize(
        self,
        principal: PrincipalContext,
        record: "RunRecord",
        action: ExecutionAction,
    ) -> None:
        self._authorization.assert_execution_access(
            principal=principal,
            tenant_id=record.tenant_id,
            user_id=record.user_id,
            action=action,
        )


def spec_type(record: "RunRecord") -> RunnableType:
    return record.definition.runnable_type


__all__ = ["ExecutionService", "PreparedAgentExecution", "spec_type"]
