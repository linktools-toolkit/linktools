#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeTaskHandler: drive the existing ``linktools.ai.Runtime`` as a task.

The handler resolves a :class:`RunnableRef` to an ``AgentSpec`` / ``SwarmSpec``
via a :class:`RunnableResolver`, mints a ``run_id``, binds it to the attempt,
and calls ``Runtime.run`` using the context's principal (tenant / user). The
existing Runtime's Security Pipeline, Capability Assembler, Tool Executor,
Policy, MCP and SubAgent paths all run unchanged because the call goes through
the public ``Runtime.run`` API -- not the private ``_runtime``.

Task->Run correlation is recorded two ways: ``bind_run`` sets
``attempt.run_id`` (traceability from the task side), and ``context_metadata``
threads the job / task / attempt / fencing lineage into RunContext.metadata so
RunRecord and RunDefinitionSnapshot carry it for cross-domain queries.
"""

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..models import TaskFailureKind
from ..protocols import TaskContext, TaskFailure, TaskRequest, TaskSuccess
from ..store import TaskClaimLostError
from ..validation import MAX_OUTPUT_PAYLOAD_BYTES


class _OutputTooLarge(Exception):
    """Raised when a sealed RunResult exceeds the per-task payload cap, so the
    handler fails closed BEFORE the oversized blob is written to the
    content-addressed store."""


@dataclass(frozen=True, slots=True)
class RunnableRef:
    id: str
    revision: "str | None" = None


class RunnableResolver(Protocol):
    async def resolve(self, ref: RunnableRef) -> object: ...


class MappingRunnableResolver:
    """Resolve by ``ref.id`` from a caller-supplied mapping. No scanning."""

    def __init__(self, mapping: "Mapping[str, object]") -> None:
        self._mapping = dict(mapping)

    async def resolve(self, ref: RunnableRef) -> object:
        if ref.id not in self._mapping:
            raise KeyError(f"runnable not found: {ref.id!r}")
        return self._mapping[ref.id]


@dataclass(frozen=True, slots=True)
class RuntimeTaskInput:
    """Carried in ``request.metadata`` until artifact-resolved inputs land."""

    runnable_id: str
    prompt: str
    session_id: "str | None" = None


class RuntimeTaskHandler:
    def __init__(
        self,
        runtime,
        resolver: RunnableResolver,
        *,
        task_store=None,
        artifact_store=None,
    ) -> None:
        self._runtime = runtime
        self._resolver = resolver
        self._task_store = task_store
        self._artifact_store = artifact_store

    async def execute(
        self, request: TaskRequest, context: TaskContext
    ) -> "TaskSuccess | TaskFailure":
        if context.delegated_scopes == ():
            return TaskFailure(
                kind=TaskFailureKind.POLICY_DENIED,
                error_type="ScopeDenied",
                message="task delegated_scopes were revoked; execution refused",
            )
        try:
            inp = RuntimeTaskInput(
                runnable_id=request.metadata["runnable_id"],
                prompt=request.metadata["prompt"],
                session_id=request.metadata.get("session_id"),
            )
        except KeyError as exc:
            return TaskFailure(
                kind=TaskFailureKind.INVALID_INPUT,
                error_type=type(exc).__name__,
                message=f"missing input key: {exc}",
            )

        try:
            spec = await self._resolver.resolve(RunnableRef(id=inp.runnable_id))
        except Exception as exc:  # noqa: BLE001
            return TaskFailure(
                kind=TaskFailureKind.PERMANENT,
                error_type=type(exc).__name__,
                message=f"runnable resolution failed: {exc}",
            )

        run_id = f"run-{uuid.uuid4().hex[:12]}"

        # Bind the run_id to the attempt so task → run traceability works.
        if self._task_store is not None:
            try:
                await self._task_store.bind_run(
                    task_id=context.task_id,
                    attempt_id=context.attempt_id,
                    fencing_token=context.fencing_token,
                    worker_id=context.worker_id,
                    run_id=run_id,
                )
            except TaskClaimLostError:
                return TaskFailure(
                    kind=TaskFailureKind.SUPERSEDED,
                    error_type="ClaimLost",
                    message="lease was lost before bind_run; task reclaimed",
                )

        # Validate pinned resource snapshots before running: a stale or missing
        # snapshot makes the run non-deterministic, so fail fast rather than
        # execute against resources that may have changed.
        if self._artifact_store is not None:
            snapshot_failure = await self._validate_snapshots(context)
            if snapshot_failure is not None:
                return snapshot_failure

        try:
            result = await self._runtime.run(
                spec,
                inp.prompt,
                session_id=inp.session_id,
                run_id=run_id,
                user_id=context.principal.user_id,
                tenant_id=context.principal.tenant_id,
                context_metadata=_task_correlation(context),
            )
        except Exception as exc:  # noqa: BLE001
            # An agent run drives tools / MCP / writes; once it has started, a
            # raised exception means the side-effect state is unknowable. Treat
            # it as SIDE_EFFECT_UNKNOWN (non-retryable) so a transient failure
            # never re-runs a side-effectful run. The message is
            # redacted by the worker before persistence.
            return TaskFailure(
                kind=TaskFailureKind.SIDE_EFFECT_UNKNOWN,
                error_type=type(exc).__name__,
                message=str(exc),
            )

        output_artifact = None
        if self._artifact_store is not None and result is not None:
            try:
                output_artifact = await self._seal_run_result(result, context)
            except _OutputTooLarge:
                return TaskFailure(
                    kind=TaskFailureKind.PERMANENT,
                    error_type="OutputTooLarge",
                    message=(
                        "run output exceeds the per-task artifact size cap "
                        f"({MAX_OUTPUT_PAYLOAD_BYTES} bytes)"
                    ),
                )

        return TaskSuccess(
            output_artifact=output_artifact,
            metadata={"run_id": run_id},
        )

    async def _validate_snapshots(self, context: TaskContext) -> "TaskFailure | None":
        """Fail fast if any pinned resource snapshot is missing or its content
        changed since it was pinned -- executing against a stale snapshot would
        make the run non-deterministic."""
        tenant = context.principal.tenant_id
        for snap in context.resource_snapshots:
            record = await self._artifact_store.stat(snap.artifact_id, tenant_id=tenant)
            if record is None or record.ref.sha256 != snap.sha256:
                return TaskFailure(
                    kind=TaskFailureKind.INVALID_INPUT,
                    error_type="StaleResourceSnapshot",
                    message=(
                        f"resource snapshot {snap.path} is missing or its "
                        "content changed since it was pinned"
                    ),
                )
        return None

    async def _seal_run_result(self, result, context: TaskContext):
        """Seal the RunResult into a content-addressed Artifact so downstream
        tasks can consume the run's output through the artifact chain. The
        RunResult is serialized as JSON via the domain serde; an output the
        serde cannot handle is coerced to its str form (and flagged) so sealing
        never crashes on an exotic output."""
        import json

        from ..models import to_jsonable

        output = getattr(result, "output", result)
        token_usage = dict(getattr(result, "token_usage", {}) or {})
        metadata = dict(getattr(result, "metadata", {}) or {})
        envelope: "dict[str, object]" = {
            "output": output,
            "token_usage": token_usage,
            "metadata": metadata,
        }
        try:
            text = json.dumps(to_jsonable(envelope))
        except TypeError:
            envelope = {
                **envelope,
                "output": str(output),
                "_output_coerced": True,
            }
            text = json.dumps(to_jsonable(envelope))
        payload = text.encode("utf-8")
        if len(payload) > MAX_OUTPUT_PAYLOAD_BYTES:
            raise _OutputTooLarge()
        record = await self._artifact_store.put(
            payload,
            media_type="application/json",
            tenant_id=context.principal.tenant_id,
            created_by_job_id=context.job_id,
            created_by_task_id=context.task_id,
            created_by_attempt_id=context.attempt_id,
        )
        return record.ref


def _task_correlation(context: TaskContext) -> "dict[str, object]":
    """Build the context_metadata dict threaded into Runtime.run so the Run
    carries its Task lineage (job / task / attempt / fencing) and the logical
    workspace scope. ``workspace_key`` is a logical tag only -- the Runtime
    still mints a fresh physical workspace per run, so no raw path is exposed."""
    meta: "dict[str, object]" = {
        "job_id": context.job_id,
        "task_id": context.task_id,
        "attempt_id": context.attempt_id,
        "fencing_token": context.fencing_token,
    }
    if context.principal.workspace_key is not None:
        meta["workspace_key"] = context.principal.workspace_key
    # Propagate the narrowed scopes + actor chain so the Run record carries the
    # task's effective permission lineage. None means unrestricted.
    if context.delegated_scopes is not None:
        meta["delegated_scopes"] = tuple(context.delegated_scopes)
    if context.actor_chain is not None:
        meta["actor_chain"] = [
            {"kind": a.kind, "id": a.id} for a in context.actor_chain.actors
        ]
    return meta


__all__: "list[str]" = [
    "RunnableRef",
    "RunnableResolver",
    "MappingRunnableResolver",
    "RuntimeTaskInput",
    "RuntimeTaskHandler",
]
