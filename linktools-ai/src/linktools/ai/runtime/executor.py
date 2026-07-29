"""Application orchestration for Run execution."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import Any
from uuid import uuid4

from ..agent.dependencies import AgentDependencies
from ..agent.engine import AgentEngine
from ..agent.models import AgentCancelled, AgentCompleted, AgentFailed, AgentPaused, AgentInput, CompiledAgent
from ..agent.spec import AgentSpec, MiddlewareRef, PromptSpec
from ..errors import StorageError
from ..model.resolver import ModelResolver
from ..storage.json import JsonValue, canonical_json_bytes
from ..execution.commands import AcknowledgeRunCancel, ClaimRun, CompleteRun, FailRun, PauseRun, RequestRunCancel, StartRun
from ..execution.run import RunDefinition, RunKind, RunRecord, RunStatus, RunnableType
from ..execution.context import RunContext
from ..execution.cancellation import CancellationToken
from ..execution.models import RunSnapshot
from ..execution.store import ExecutionStore
from ..execution import trace_codec
from ..execution.trace_collector import SemanticTraceCollector


class _Events:
    async def emit(self, event: object) -> None:
        return None

    async def publish(self, event: object) -> None:
        return None


def _spec_value(spec: AgentSpec) -> JsonValue:
    output_schema = None if spec.output_schema is None else spec.output_schema.model_json_schema()
    return {
        "id": spec.id,
        "name": spec.name,
        "model": asdict(spec.model),
        "instructions": {"instructions": spec.instructions.instructions, "sections": dict(spec.instructions.sections)},
        "tools": None if spec.tools is None else [asdict(item) for item in spec.tools],
        "middleware": [{"name": item.name, "config": dict(item.config)} for item in spec.middleware],
        "output_schema": output_schema,
        "metadata": dict(spec.metadata),
    }


def _definition(spec: AgentSpec) -> RunDefinition:
    value = _spec_value(spec)
    return RunDefinition(spec.id, RunnableType.AGENT, "agent-spec.v1", value, sha256(canonical_json_bytes(value)).hexdigest())


def _snapshot(outcome: object) -> RunSnapshot:
    if isinstance(outcome, (AgentCompleted, AgentPaused, AgentCancelled, AgentFailed)):
        snapshot = outcome.snapshot
    else:
        snapshot = None
    if not isinstance(snapshot, RunSnapshot):
        raise StorageError("agent outcome did not provide a canonical snapshot")
    return snapshot


class RuntimeExecutor:
    def __init__(self, store: ExecutionStore, model_resolver: ModelResolver) -> None:
        self.store = store
        self.model_resolver = model_resolver

    async def run(self, spec: AgentSpec, prompt: str, *, session_id: str, run_id: str | None = None, user_id: str | None = None, tenant_id: str | None = None) -> object:
        if await self.store.get_session(session_id) is None:
            await self.store.create_session(session_id=session_id, user_id=user_id, tenant_id=tenant_id)
        record = await self.store.start_run(StartRun(run_id or uuid4().hex, session_id, RunKind.USER_TURN, _definition(spec), prompt))
        claimed = await self.store.claim_run(ClaimRun(record.id, "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
        return await self._execute(spec, prompt, claimed, resuming=False)

    async def resume(self, run_id: str) -> object:
        record = await self._required(run_id)
        value = record.definition.spec
        spec = _agent_spec(value)
        claimed = await self.store.claim_run(ClaimRun(run_id, "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
        messages = await self.store.load_session_context(record.session_id)
        return await self._execute(spec, "", claimed, resuming=True, message_history=messages)

    async def cancel(self, run_id: str) -> None:
        record = await self._required(run_id)
        if record.lease.owner is None:
            raise StorageError("run has no active owner")
        from ..execution.commands import RequestRunCancel

        await self.store.request_cancel(RequestRunCancel(run_id, record.lease.owner, record.lease.fence, datetime.now(timezone.utc)))

    async def _execute(self, spec: AgentSpec, prompt: str, record: RunRecord, *, resuming: bool, message_history: tuple[object, ...] = ()) -> object:
        from pydantic_ai import Agent as PydanticAgent

        resolved = self.model_resolver.resolve(spec.model)
        agent = PydanticAgent(resolved.model, output_type=spec.output_schema or str, deps_type=AgentDependencies, instructions=spec.instructions.instructions)
        compiled = CompiledAgent(spec=spec, pydantic_agent=agent, model_bundle=resolved, policy_capability=None)
        context = RunContext(record.id, record.root_run_id, record.parent_run_id, record.session_id, record.runnable_id, spec_type(record), record.user_id, record.tenant_id, None)
        collector = SemanticTraceCollector(record.id, self.store, record.trace_sequence)
        engine = AgentEngine(trace_collector=collector, trace_codec=trace_codec)
        decoded_history = decode_model_messages(message_history) if message_history else ()
        outcome = await engine.execute_pure(compiled, AgentInput(prompt=prompt, message_history=decoded_history, resuming=resuming), context, cancellation=CancellationToken(), live_events=_Events(), security_events=_Events(), trace_sequence=record.trace_sequence)
        snapshot = _snapshot(outcome)
        if isinstance(outcome, AgentCompleted):
            await self.store.complete_run(CompleteRun(record.id, record.lease.owner or "runtime", record.lease.fence, snapshot))
            return outcome.result.output
        if isinstance(outcome, AgentPaused):
            approval = outcome.request
            from ..execution.run import RunApproval

            await self.store.pause_run(PauseRun(record.id, record.lease.owner or "runtime", record.lease.fence, snapshot, RunApproval(approval.approval_id, approval.tool_call_id or "", approval.tool_name or "", dict(approval.arguments))))
            return None
        if isinstance(outcome, AgentCancelled):
            await self.store.acknowledge_cancel(AcknowledgeRunCancel(record.id, record.lease.owner or "runtime", record.lease.fence, snapshot))
            return None
        if isinstance(outcome, AgentFailed):
            await self.store.fail_run(FailRun(record.id, record.lease.owner or "runtime", record.lease.fence, snapshot))
            raise RuntimeError(outcome.error.message)
        raise AssertionError(f"unsupported agent outcome: {type(outcome).__name__}")

    async def _required(self, run_id: str) -> RunRecord:
        record = await self.store.get_run(run_id)
        if record is None:
            raise KeyError(run_id)
        return record


def spec_type(record: RunRecord):
    return record.definition.runnable_type


def _agent_spec(value: JsonValue) -> AgentSpec:
    from ..model.policy import ModelPolicy

    data = dict(value)
    output_schema = None
    if data.get("output_schema"):
        from pydantic import create_model
        output_schema = create_model(f"{data['id'].replace('-', '_')}Output")
    return AgentSpec(
        data["id"],
        data["name"],
        ModelPolicy(**data["model"]),
        PromptSpec(**data["instructions"]),
        metadata=data.get("metadata", {}),
        middleware=tuple(MiddlewareRef(**item) for item in data.get("middleware", [])),
        output_schema=output_schema,
    )


__all__ = ["RuntimeExecutor"]
