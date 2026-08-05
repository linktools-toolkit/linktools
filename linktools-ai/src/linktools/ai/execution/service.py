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
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4

from linktools.core import environ

from ..agent.assembly.provider import AgentFeatureContext
from ..agent.models import (
    AgentCancelled,
    AgentCompleted,
    AgentFailed,
    AgentInput,
    AgentPaused,
)
from ..agent.sandbox.protocols import Sandbox
from ..errors import (
    ChildSnapshotError,
    ExecutionLifecycleDeliveryError,
    ExecutionLifecyclePersistenceError,
    ExecutionTerminalMismatchError,
    PrincipalAccessDeniedError,
    RunDefinitionError,
    RuntimeInitializationError,
    StorageError,
)
from ..governance.authorization import ExecutionAction
from ..governance.identity import PrincipalContext
from ..json import canonical_json_bytes
from .cancellation import CancellationToken
from .commands import (
    AbortExecution,
    AcknowledgeCancellation,
    ClaimExecution,
    CheckpointExecutionUsage,
    CompleteExecution,
    DecideApproval,
    FailExecution,
    HeartbeatExecution,
    ParentLeaseGuard,
    PauseExecution,
    RequestCancellation,
    ResumeExecution,
    StartClaimedChildExecution,
    StartExecution,
)
from .controller import ExecutionControllerRegistry
from .context import RunContext
from .domain import (
    MessageCaptureState,
    RunApproval,
    RunDefinition,
    RunError,
    RunKind,
    RunStatus,
    RunnableType,
    RunUsage,
    compute_run_definition_hash,
    sanitize_run_error,
)
from .query import ExecutionResultView
from .session import SessionContextSeed, SessionRecord, SeedTurn
from .snapshots import (
    AgentSnapshotData,
    ModelRequestUsageObservation,
    RunUsageCapture,
)
from . import trace_codec
from .trace_collector import SemanticTraceCollector
from ..prompt import UserPrompt
from .live_events import (
    ExecutionCancelled,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionPaused,
    UsageUpdated,
    publish_execution_event,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent.codec import AgentSpecCodec
    from ..agent.compiler import AgentCompiler
    from ..agent.engine import AgentEngine
    from ..agent.assembly.assembler import AgentAssembler
    from ..agent.assembly.models import AgentAssembly
    from ..agent.spec import AgentSpec
    from ..governance.authorization import AuthorizationPolicy
    from ..model.pricing import ModelPricing
    from ..tasks.models import TaskUsage
    from .domain import ApprovalDecision, RunRecord
    from .live_events import RunLiveEventSink, SecurityEventSink
    from .store import ExecutionStore

logger = environ.get_logger("ai.execution.service")


def _definition(spec: "AgentSpec", codec: "AgentSpecCodec") -> RunDefinition:
    value = codec.encode(spec)
    schema = "agent-spec.v1"
    return RunDefinition(
        spec.id,
        RunnableType.AGENT,
        schema,
        value,
        compute_run_definition_hash(schema=schema, spec=value),
    )


def _decode_definition(
    definition: RunDefinition,
    codec: "AgentSpecCodec",
) -> "AgentSpec":
    if definition.schema != "agent-spec.v1":
        raise RunDefinitionError("unsupported definition schema")
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
    snapshot_revision: int


@dataclass(slots=True)
class _InvocationEnvelope:
    execution_id: str
    record: "RunRecord | None" = None
    boundary: "object | None" = None
    boundary_published: bool = False
    persistence_error: "ExecutionLifecyclePersistenceError | None" = None
    original_error: "BaseException | None" = None
    finalizer_task: "asyncio.Task[None] | None" = None


@dataclass(frozen=True, slots=True)
class PreparedAgentExecution:
    agent_spec: "AgentSpec"
    assembled_agent: object
    tool_descriptors: "tuple[object, ...]"
    fingerprint: str


@dataclass(slots=True)
class PersistedRunUsageSink:
    """Persist each new model-request usage before downstream work continues."""

    capture: RunUsageCapture
    store: "ExecutionStore"
    run_id: str
    owner: str
    fence: int
    snapshot_revision: int
    trace_sequence: "Callable[[], int]"

    async def observe_request(
        self,
        observation: ModelRequestUsageObservation,
        *,
        pricing: "ModelPricing | None",
    ) -> RunUsage:
        if self.capture.has_observation(observation.request_key):
            record = await self.store.get_run(self.run_id)
            if record is None:
                raise StorageError(f"unknown run: {self.run_id}")
            self.snapshot_revision = record.snapshot_revision
            return self.capture.observe_request(observation, pricing=pricing)
        usage = self.capture.observe_request(observation, pricing=pricing)
        snapshot = await self.store.checkpoint_run_usage(
            CheckpointExecutionUsage(
                run_id=self.run_id,
                owner=self.owner,
                fence=self.fence,
                expected_snapshot_revision=self.snapshot_revision,
                usage=usage,
                trace_end_sequence=self.trace_sequence(),
            )
        )
        self.snapshot_revision = snapshot.revision
        logger.debug(
            "run %s persisted usage revision=%s tokens=%s cost=%s",
            self.run_id,
            self.snapshot_revision,
            usage.total_tokens,
            usage.total_cost,
        )
        return usage

    def snapshot(self) -> RunUsage:
        return self.capture.snapshot()

    @property
    def last_snapshot_revision(self) -> int:
        return self.snapshot_revision


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
    return ChildRunResult(
        run_id=run_id,
        status=record.status,
        output=output,
        error=error,
        usage=usage,
        snapshot_revision=record.snapshot_revision,
    )


async def _child_result_from_persistence(
    store: "ExecutionStore",
    run_id: str,
    outcome: "object | None",
    record: "RunRecord",
) -> "ChildRunResult":
    snapshot = await store.get_snapshot(run_id)
    if snapshot is None:
        raise ChildSnapshotError(run_id)
    return ChildRunResult(
        run_id=run_id,
        status=record.status,
        output=snapshot.final_output if record.status is RunStatus.COMPLETED else None,
        error=record.error,
        usage=_task_usage_from_run_usage(snapshot.usage),
        snapshot_revision=snapshot.revision,
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

    async def create_session(
        self,
        session_id: str,
        *,
        principal: PrincipalContext,
        context_seed: "SessionContextSeed | None" = None,
    ) -> SessionRecord:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        session = await self._store.create_session(
            session_id=session_id,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            context_seed=context_seed,
        )
        self._authorization.assert_session_access(principal=principal, session=session)
        return session

    async def get_session(self, session_id: str, *, principal: PrincipalContext) -> "SessionRecord | None":
        session = await self._store.get_session(session_id)
        if session is not None:
            self._authorization.assert_session_access(principal=principal, session=session)
        return session

    async def get_execution_record(
        self, execution_id: str, *, principal: PrincipalContext
    ) -> "RunRecord | None":
        record = await self._store.get_run(execution_id)
        if record is not None:
            self._authorize(principal, record, ExecutionAction.INSPECT)
        return record

    async def list_sessions(self, *, principal: PrincipalContext) -> "tuple[SessionRecord, ...]":
        values = await self._store.list_all_sessions()
        visible = []
        for session in values:
            try:
                self._authorization.assert_session_access(principal=principal, session=session)
            except Exception:
                continue
            visible.append(session)
        return tuple(visible)

    async def run(
        self,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        *,
        principal: PrincipalContext,
        session_id: "str | None" = None,
        execution_id: "str | None" = None,
        extra_toolsets: "tuple[Any, ...]" = (),
    ) -> object:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        if self._store is None:
            raise RuntimeInitializationError("execution store is unavailable")
        session_id = session_id or uuid4().hex
        execution_id = execution_id or uuid4().hex
        envelope = _InvocationEnvelope(execution_id)

        async def body(scope: _InvocationEnvelope) -> object:
            session = await self._store.create_session(
                session_id=session_id,
                user_id=principal.user_id,
                tenant_id=principal.tenant_id,
            )
            self._authorization.assert_session_access(
                principal=principal,
                session=session,
            )
            persisted_prompt = prompt.to_json() if isinstance(prompt, UserPrompt) else prompt
            scope.record = (
                await self._store.start_run(
                    StartExecution(
                        execution_id,
                        session_id,
                        RunKind.USER_TURN,
                        _definition(spec, self._codec),
                        persisted_prompt,
                    )
                )
            ).record
            assembly = await self._preflight(
                spec,
                execution_id=execution_id,
                session_id=session_id,
                root_execution_id=execution_id,
                parent_execution_id=None,
                principal=principal,
            )
            claim_task = asyncio.create_task(
                self._store.claim_run(
                    ClaimExecution(
                        scope.record.id,
                        "runtime",
                        datetime.now(timezone.utc),
                        _LEASE_DURATION,
                    )
                )
            )
            context_task = asyncio.create_task(self._store.load_session_context(session_id))
            try:
                claimed, messages = await asyncio.gather(claim_task, context_task)
            except BaseException:
                for task in (claim_task, context_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(claim_task, context_task, return_exceptions=True)
                raise
            scope.record = claimed
            logger.debug(
                "event=execution.invocation_claimed execution_id=%s session_id=%s history=%s",
                claimed.id,
                session_id,
                len(messages),
            )
            return await self._execute(
                spec,
                prompt,
                claimed,
                resuming=False,
                message_history=messages,
                assembly=assembly,
                extra_toolsets=extra_toolsets,
                envelope=scope,
            )

        return await self._drive_invocation(envelope, body)

    async def fork_session(
        self,
        source_session_id: str,
        target_session_id: str,
        principal: PrincipalContext,
    ) -> SessionRecord:
        """Create an immutable context seed from a complete idle session."""
        source = await self._store.get_session(source_session_id)
        if source is None:
            raise StorageError("unknown session")
        self._authorization.assert_session_access(principal=principal, session=source)
        active = {
            RunStatus.PENDING,
            RunStatus.RUNNING,
            RunStatus.PAUSED,
            RunStatus.CANCELLING,
        }
        if any(
            record.session_id == source_session_id and record.status in active
            for record in await self._store.list_all_runs()
        ):
            raise StorageError("session is busy")
        turns = await self._store.get_session_messages(source_session_id)
        if any(
            turn.status is not RunStatus.COMPLETED
            or turn.capture_state is not MessageCaptureState.COMPLETE
            for turn in turns
        ):
            raise StorageError("session history is incomplete")
        seed = SessionContextSeed(
            schema="session-context-seed.v1",
            source_session_id=source_session_id,
            source_updated_at=source.updated_at,
            turns=tuple(
                SeedTurn(
                    session_id=turn.session_id,
                    sequence=turn.sequence,
                    run_id=turn.run_id,
                    input=turn.input,
                    delta_messages=turn.delta_messages,
                    status=turn.status,
                    capture_state=turn.capture_state,
                )
                for turn in turns
            ),
        )
        target = await self._store.create_session(
            session_id=target_session_id,
            user_id=principal.user_id,
            tenant_id=principal.tenant_id,
            context_seed=seed,
        )
        logger.info(
            "session forked source=%s target=%s turns=%s",
            source_session_id,
            target_session_id,
            len(turns),
        )
        return target

    async def resume(
        self,
        run_id: str,
        *,
        principal: PrincipalContext,
        extra_toolsets: "tuple[Any, ...]" = (),
    ) -> object:
        if not isinstance(principal, PrincipalContext):
            raise TypeError("principal must be a PrincipalContext")
        envelope = _InvocationEnvelope(run_id)

        async def body(scope: _InvocationEnvelope) -> object:
            record = await self._required(run_id)
            scope.record = record
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
            await self._store.resume_run(ResumeExecution(run_id))
            claim_task = asyncio.create_task(
                self._store.claim_run(
                    ClaimExecution(
                        run_id,
                        "runtime",
                        datetime.now(timezone.utc),
                        _LEASE_DURATION,
                    )
                )
            )
            context_task = asyncio.create_task(self._store.load_resume_messages(run_id))
            try:
                claimed, messages = await asyncio.gather(claim_task, context_task)
            except BaseException:
                for task in (claim_task, context_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(claim_task, context_task, return_exceptions=True)
                raise
            scope.record = claimed
            return await self._execute(
                spec,
                "",
                claimed,
                resuming=True,
                message_history=messages,
                assembly=assembly,
                extra_toolsets=extra_toolsets,
                envelope=scope,
            )

        return await self._drive_invocation(envelope, body)

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
        parent_guard: "ParentLeaseGuard",
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
        start_command = StartExecution(
            execution_id,
            session_id,
            RunKind.TASK,
            _definition(spec, self._codec),
            {"prompt": prompt, "metadata": dict(metadata) if metadata else {}},
            root_execution_id=root_execution_id,
            parent_execution_id=parent_execution_id,
            parent_guard=parent_guard,
        )
        started = await self._store.start_claimed_child(
            StartClaimedChildExecution(
                start=start_command,
                child_owner="swarm",
                lease_duration=_LEASE_DURATION,
            )
        )
        if started.terminal:
            return await _child_result_from_persistence(
                self._store, started.record.id, None, started.record
            )
        claimed = started.record
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
        usage_capture = await self._usage_capture(claimed)
        owner = claimed.lease.owner or "swarm"
        usage_sink = PersistedRunUsageSink(
            capture=usage_capture,
            store=self._store,
            run_id=claimed.id,
            owner=owner,
            fence=claimed.lease.fence,
            snapshot_revision=claimed.snapshot_revision,
            trace_sequence=lambda: collector.next_sequence,
        )
        decoded_history = decode_model_messages(message_history) if message_history else ()
        token = CancellationToken()
        agent_input = AgentInput(prompt=prompt, message_history=decoded_history, metadata=metadata or {})
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
                usage_sink=usage_sink,
            ),
            token,
        )
        # task-owner: execution.child_heartbeat
        heartbeat = asyncio.ensure_future(
            self._heartbeat(claimed.id, owner, claimed.lease.fence, token)
        )
        outcome: "object | None" = None
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
                claimed,
                owner,
                _task_usage_from_run_usage(usage_sink.snapshot()),
            )
            latest = await self._required(execution_id)
            return ChildRunResult(
                run_id=execution_id,
                status=RunStatus.CANCELLED,
                output=None,
                error=latest.error,
                usage=_task_usage_from_run_usage(usage_sink.snapshot()),
                snapshot_revision=latest.snapshot_revision,
            )
        except BaseException as exc:
            trace_end = await collector.flush()
            await self._store.abort_run(
                AbortExecution(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    AgentSnapshotData(
                        delta_messages=(),
                        final_output=None,
                        usage=usage_sink.snapshot(),
                        trace_end_sequence=trace_end,
                        capture_state=MessageCaptureState.PARTIAL,
                    ),
                    sanitize_run_error(exc),
                    usage_sink.last_snapshot_revision,
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
        if snapshot is not None:
            snapshot = replace(snapshot, usage=usage_sink.snapshot())
        latest = await self._required(execution_id)
        if latest.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return await _child_result_from_persistence(
                self._store, execution_id, outcome, latest
            )
        if isinstance(outcome, AgentCompleted):
            await self._store.complete_run(
                CompleteExecution(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    snapshot,
                    usage_sink.last_snapshot_revision,
                )
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
                    usage_sink.last_snapshot_revision,
                )
            )
            latest = await self._required(execution_id)
            return await _child_result_from_persistence(
                self._store, execution_id, outcome, latest
            )
        elif isinstance(outcome, AgentCancelled):
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    snapshot,
                    usage_sink.last_snapshot_revision,
                )
            )
        elif isinstance(outcome, AgentFailed):
            await self._store.fail_run(
                FailExecution(
                    claimed.id,
                    owner,
                    claimed.lease.fence,
                    snapshot,
                    outcome.error,
                    usage_sink.last_snapshot_revision,
                )
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
                    latest.snapshot_revision,
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
                logger.debug("run %s heartbeat detected CANCELLING", run_id)
                await self._controller.cancel(run_id)
                return

    async def _execute(
        self,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        record: "RunRecord",
        *,
        resuming: bool,
        message_history: "tuple[object, ...]" = (),
        assembly: "AgentAssembly | None" = None,
        extra_toolsets: "tuple[Any, ...]" = (),
        envelope: _InvocationEnvelope,
    ) -> object:
        return await self._execute_once(
            spec,
            prompt,
            record,
            resuming=resuming,
            message_history=message_history,
            assembly=assembly,
            extra_toolsets=extra_toolsets,
            envelope=envelope,
        )

    async def _drive_invocation(
        self,
        envelope: _InvocationEnvelope,
        body: "Callable[[_InvocationEnvelope], Awaitable[object]]",
    ) -> object:
        try:
            return await body(envelope)
        except asyncio.CancelledError as error:
            envelope.original_error = error
            await self._prepare_cancel_boundary(envelope)
            raise
        except BaseException as error:
            envelope.original_error = error
            await self._prepare_failure_boundary(envelope, error)
            raise
        finally:
            await self._finalize_envelope(envelope)

    async def _prepare_failure_boundary(
        self,
        envelope: _InvocationEnvelope,
        error: BaseException,
    ) -> None:
        if envelope.boundary is not None:
            return
        if envelope.record is None:
            sanitized = sanitize_run_error(error)
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id=sanitized.error_type,
                error_type=type(error).__name__,
            )
            logger.info(
                "event=execution.invocation_boundary_prepared execution_id=%s invocation_stage=transport boundary_type=ExecutionFailed boundary_published=%s error_id=%s",
                envelope.execution_id,
                envelope.boundary_published,
                sanitized.error_type,
            )
            return
        if isinstance(error, PrincipalAccessDeniedError):
            sanitized = sanitize_run_error(error)
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id=sanitized.error_type,
                error_type=type(error).__name__,
            )
            return
        try:
            record = await self._persist_invocation_failure(envelope.record, error)
        except ExecutionLifecyclePersistenceError as persistence_error:
            envelope.persistence_error = persistence_error
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id="lifecycle_persistence_failed",
                error_type=type(persistence_error).__name__,
            )
            logger.error(
                "event=execution.invocation_persistence_failed execution_id=%s invocation_stage=terminal_persistence persistence_target_status=%s persistence_error_id=%s",
                envelope.execution_id,
                persistence_error.target_status,
                persistence_error.error_id,
                exc_info=environ.debug,
            )
            return
        error_id = record.error.error_type if record.error is not None else type(error).__name__
        envelope.boundary = ExecutionFailed(
            execution_id=envelope.execution_id,
            error_id=error_id,
            error_type=type(error).__name__,
        )

    async def _prepare_cancel_boundary(self, envelope: _InvocationEnvelope) -> None:
        if envelope.boundary is not None:
            return
        if envelope.record is None:
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id="cancelled_before_run_record",
                error_type="CancelledError",
            )
            return
        try:
            record = await self._persist_invocation_cancel(envelope.record)
        except ExecutionLifecyclePersistenceError as persistence_error:
            envelope.persistence_error = persistence_error
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id="lifecycle_persistence_failed",
                error_type=type(persistence_error).__name__,
            )
            logger.error(
                "event=execution.invocation_cancel_persistence_failed execution_id=%s invocation_stage=terminal_persistence persistence_target_status=%s persistence_error_id=%s",
                envelope.execution_id,
                persistence_error.target_status,
                persistence_error.error_id,
                exc_info=environ.debug,
            )
            return
        if record.status is RunStatus.FAILED:
            error = record.error
            envelope.boundary = ExecutionFailed(
                execution_id=envelope.execution_id,
                error_id=error.error_type if error is not None else "",
                error_type=error.error_type if error is not None else "",
            )
        elif record.status is RunStatus.COMPLETED:
            envelope.boundary = ExecutionCompleted(execution_id=envelope.execution_id)
        else:
            envelope.boundary = ExecutionCancelled(execution_id=envelope.execution_id)

    async def _persist_invocation_failure(
        self, record: "RunRecord", error: BaseException
    ) -> "RunRecord":
        sanitized = sanitize_run_error(error)
        try:
            latest = await self._required(record.id)
            if latest.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                return latest
            if latest.status is RunStatus.PAUSED:
                latest = await self._store.resume_run(ResumeExecution(record.id))
            return await self._store.abort_run(
                AbortExecution(
                    record.id,
                    latest.lease.owner or record.lease.owner or "runtime",
                    latest.lease.fence,
                    AgentSnapshotData(
                        delta_messages=(),
                        final_output=None,
                        usage=RunUsage(),
                        trace_end_sequence=latest.trace_sequence,
                        capture_state=MessageCaptureState.PARTIAL,
                    ),
                    sanitized,
                    latest.snapshot_revision,
                )
            )
        except BaseException as persist_error:
            persist_error_id = sanitize_run_error(persist_error).error_type
            logger.error(
                "event=execution.invocation_failure_persist_failed execution_id=%s persistence_target_status=FAILED persistence_error_id=%s",
                record.id,
                persist_error_id,
                exc_info=environ.debug,
            )
            raise ExecutionLifecyclePersistenceError(
                record.id,
                RunStatus.FAILED.value,
                persist_error_id,
            ) from persist_error

    async def _persist_invocation_cancel(self, record: "RunRecord") -> "RunRecord":
        try:
            latest = await self._required(record.id)
            owner = latest.lease.owner or record.lease.owner or "runtime"
            if latest.status in {RunStatus.PENDING, RunStatus.PAUSED, RunStatus.RUNNING}:
                latest = await self._store.request_cancel(
                    RequestCancellation(
                        record.id,
                        owner,
                        latest.lease.fence,
                        datetime.now(timezone.utc),
                    )
                )
            if latest.status is RunStatus.CANCELLING:
                return await self._store.acknowledge_cancel(
                    AcknowledgeCancellation(
                        record.id,
                        owner,
                        latest.lease.fence,
                        AgentSnapshotData(
                            delta_messages=(),
                            final_output=None,
                            usage=RunUsage(),
                            trace_end_sequence=latest.trace_sequence,
                            capture_state=MessageCaptureState.PARTIAL,
                        ),
                        latest.snapshot_revision,
                    )
                )
            return latest
        except BaseException as persist_error:
            persist_error_id = sanitize_run_error(persist_error).error_type
            logger.error(
                "event=execution.invocation_cancel_persist_failed execution_id=%s persistence_target_status=CANCELLED persistence_error_id=%s",
                record.id,
                persist_error_id,
                exc_info=environ.debug,
            )
            raise ExecutionLifecyclePersistenceError(
                record.id,
                RunStatus.CANCELLED.value,
                persist_error_id,
            ) from persist_error

    async def _execute_once(
        self,
        spec: "AgentSpec",
        prompt: "str | UserPrompt",
        record: "RunRecord",
        *,
        resuming: bool,
        message_history: "tuple[object, ...]" = (),
        assembly: "AgentAssembly | None" = None,
        extra_toolsets: "tuple[Any, ...]" = (),
        envelope: _InvocationEnvelope,
    ) -> object:
        owner = record.lease.owner or "runtime"
        compiled = await self._compiler.compile(spec)
        context = RunContext(record.id, record.root_execution_id, record.parent_execution_id, record.session_id, record.runnable_id, record.definition.runnable_type, record.user_id, record.tenant_id, None)
        collector = SemanticTraceCollector(record.id, self._store, record.trace_sequence)
        usage_sink = PersistedRunUsageSink(
            capture=await self._usage_capture(record),
            store=self._store,
            run_id=record.id,
            owner=owner,
            fence=record.lease.fence,
            snapshot_revision=record.snapshot_revision,
            trace_sequence=lambda: collector.next_sequence,
        )
        async def publish_boundary(event: object) -> None:
            if envelope.boundary is not None:
                raise ExecutionTerminalMismatchError(record.id)
            envelope.boundary = event
            logger.debug(
                "event=execution.invocation_boundary_prepared execution_id=%s invocation_stage=execution boundary_type=%s boundary_published=%s",
                record.id,
                type(event).__name__,
                envelope.boundary_published,
            )

        decoded_history = decode_model_messages(message_history) if message_history else ()
        token = CancellationToken()
        approved = record.approval if resuming else None
        task = await self._controller.start(
            record.id,
            self._engine.execute_pure(compiled, AgentInput(prompt=prompt, message_history=decoded_history, resuming=resuming, approved_tool_call_id=approved.tool_call_id if approved is not None else None, approved_binding_fingerprint=approved.binding_fingerprint if approved is not None else None), context, cancellation=token, live_events=self._live_events, security_events=self._security_events, assembly=assembly, trace_sequence=record.trace_sequence, trace_collector=collector, usage_sink=usage_sink, extra_toolsets=extra_toolsets),
            token,
        )
        try:
            # task-owner: execution.heartbeat
            heartbeat = asyncio.ensure_future(
                self._heartbeat(record.id, owner, record.lease.fence, token)
            )
        except BaseException:
            await self._controller.cancel(record.id)
            await self._controller.unregister(record.id, task=task)
            raise
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
                    logger.debug(
                        "run %s cancelled-task cleanup swallowed: %s: %s",
                        record.id, type(task_exc).__name__, task_exc,
                    )
            raise
        except Exception:
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
            await self._controller.unregister(record.id, task=task)
        snapshot = _snapshot(outcome)
        snapshot = replace(snapshot, usage=usage_sink.snapshot())
        latest = await self._required(record.id)
        if latest.status in {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }:
            if latest.status is RunStatus.COMPLETED:
                terminal_event: object = ExecutionCompleted(execution_id=record.id)
            elif latest.status is RunStatus.FAILED:
                error = latest.error
                terminal_event = ExecutionFailed(
                    execution_id=record.id,
                    error_id=error.error_type if error is not None else "",
                    error_type=error.error_type if error is not None else "",
                )
            else:
                terminal_event = ExecutionCancelled(execution_id=record.id)
            await publish_boundary(terminal_event)
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
                    usage_sink.last_snapshot_revision,
                )
            )
            usage = usage_sink.snapshot()
            if usage.total_tokens > 0 or usage.total_cost is not None:
                await self._publish_lifecycle_event(
                    UsageUpdated(
                        execution_id=record.id,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        total_tokens=usage.total_tokens,
                        total_cost=(
                            str(usage.total_cost)
                            if usage.total_cost is not None
                            else None
                        ),
                    )
                )
            await publish_boundary(
                ExecutionCancelled(execution_id=record.id)
            )
            return None
        if isinstance(outcome, AgentCompleted):
            await self._store.complete_run(
                CompleteExecution(
                    record.id,
                    owner,
                    record.lease.fence,
                    snapshot,
                    usage_sink.last_snapshot_revision,
                )
            )
            await publish_boundary(
                ExecutionCompleted(execution_id=record.id)
            )
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
                    usage_sink.last_snapshot_revision,
                )
            )
            await publish_boundary(
                ExecutionPaused(
                    execution_id=record.id,
                    approval_id=approval.approval_id,
                    tool_call_id=approval.tool_call_id or "",
                    tool_name=approval.tool_name or "",
                )
            )
            return None
        if isinstance(outcome, AgentCancelled):
            await self._store.acknowledge_cancel(
                AcknowledgeCancellation(
                    record.id,
                    owner,
                    record.lease.fence,
                    snapshot,
                    usage_sink.last_snapshot_revision,
                )
            )
            await publish_boundary(
                ExecutionCancelled(execution_id=record.id)
            )
            return None
        if isinstance(outcome, AgentFailed):
            error = outcome.error or RunError("AgentFailed", "execution failed")
            await self._store.fail_run(
                FailExecution(
                    record.id,
                    owner,
                    record.lease.fence,
                    snapshot,
                    error,
                    usage_sink.last_snapshot_revision,
                )
            )
            await publish_boundary(
                ExecutionFailed(
                    execution_id=record.id,
                    error_id=error.error_type,
                    error_type=error.error_type,
                )
            )
            raise RuntimeError(error.message)
        error = AssertionError(f"unsupported agent outcome: {type(outcome).__name__}")
        raise error

    async def _finalize_envelope(self, envelope: _InvocationEnvelope) -> None:
        task = envelope.finalizer_task
        if task is None:
            task = asyncio.create_task(self._finalize_envelope_once(envelope))
            envelope.finalizer_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _finalize_envelope_once(self, envelope: _InvocationEnvelope) -> None:
        if envelope.boundary is None:
            raise ExecutionTerminalMismatchError(envelope.execution_id)
        if envelope.boundary_published:
            return
        publish_error: "BaseException | None" = None
        try:
            await self._publish_lifecycle_event(envelope.boundary)
            envelope.boundary_published = True
        except BaseException as exc:
            publish_error = exc
        finally:
            await self._close_live_cycle(envelope.execution_id)
        if publish_error is not None:
            delivery_error = ExecutionLifecycleDeliveryError(
                f"execution boundary delivery failed: {envelope.execution_id}"
            )
            cause = envelope.original_error or publish_error
            raise delivery_error from cause
        logger.info(
            "event=execution.invocation_boundary_finalized execution_id=%s boundary_type=%s boundary_published=%s persistence_error_id=%s",
            envelope.execution_id,
            type(envelope.boundary).__name__,
            envelope.boundary_published,
            envelope.persistence_error.error_id if envelope.persistence_error is not None else None,
        )
        if envelope.persistence_error is not None:
            raise envelope.persistence_error

    async def _publish_lifecycle_event(self, event: object) -> None:
        await publish_execution_event(self._live_events, event.execution_id, event)

    async def _close_live_cycle(self, execution_id: str) -> None:
        closer = getattr(self._live_events, "close_execution_cycle", None)
        if closer is None:
            return
        try:
            await closer(execution_id)
        except Exception as exc:
            logger.error(
                "event=execution.extra_cycle_closed execution_id=%s error_id=%s",
                execution_id,
                type(exc).__name__,
            )

    async def _required(self, run_id: str) -> "RunRecord":
        record = await self._store.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    async def _usage_capture(self, record: "RunRecord") -> RunUsageCapture:
        if record.snapshot_revision == 0:
            return RunUsageCapture()
        snapshot = await self._store.get_snapshot(record.id)
        if snapshot is None:
            raise StorageError("run snapshot is missing")
        return RunUsageCapture.from_usage(snapshot.usage)

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
