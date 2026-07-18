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

import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from ..models import TaskFailureKind, to_jsonable
from ..protocols import TaskContext, TaskFailure, TaskRequest, TaskSuccess
from ..store import RunnableBindingError, TaskClaimLostError
from ..validation import MAX_OUTPUT_PAYLOAD_BYTES
from ...security.principal import ActorRef, PrincipalContext, ScopeSet


class _OutputTooLarge(Exception):
    """Raised when a sealed RunResult exceeds the per-task payload cap, so the
    handler fails closed BEFORE the oversized blob is written to the
    content-addressed store."""


def _fingerprint(spec: object) -> str:
    """A stable content fingerprint of a resolved runnable spec, so bind_runnable
    can detect that a mapping change returned a different spec for the same id
    (even with no revision). Only canonical JSON is accepted; nondeterministic
    repr-based fingerprints are rejected.

    Assumes the spec serializes deterministically (dict key order is normalized
    via sort_keys; unordered collections like sets are not -- a spec carrying a
    set would be non-deterministic and should be normalized by its owner)."""
    try:
        payload = json.dumps(to_jsonable(spec), sort_keys=True).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - no nondeterministic repr fallback
        raise ValueError(
            "runnable resolver must provide a deterministically serializable fingerprint"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RunnableRef:
    id: str
    revision: "str | None" = None


class RunnableResolver(Protocol):
    async def resolve(self, ref: RunnableRef) -> object: ...


class MappingRunnableResolver:
    """Resolve by ``ref.id`` from a caller-supplied id-only mapping. No scanning.

    An id-only mapping CANNOT honor a revision: if ``ref.revision`` is set,
    ``resolve`` raises rather than silently returning whatever the id currently
    maps to (which could be a different agent after a mapping change, silently
    re-running a retry against new code). Wire a revision-aware resolver (keyed
    by ``(id, revision)``) when revisions matter."""

    def __init__(self, mapping: "Mapping[str, object]") -> None:
        self._mapping = dict(mapping)

    async def resolve(self, ref: RunnableRef) -> object:
        if ref.revision is not None:
            raise ValueError(
                f"MappingRunnableResolver cannot honor revision {ref.revision!r} "
                f"for runnable {ref.id!r}; provide a revision-aware resolver"
            )
        if ref.id not in self._mapping:
            raise KeyError(f"runnable not found: {ref.id!r}")
        return self._mapping[ref.id]


@dataclass(frozen=True, slots=True)
class RuntimeTaskInput:
    """Carried in ``request.metadata`` until artifact-resolved inputs land.

    Carries a full :class:`RunnableRef` (id + revision), not just an id, so the
    resolver can pin the exact runnable and a retry re-resolves the SAME
    revision instead of silently picking up a changed mapping."""

    runnable: RunnableRef
    prompt: str
    session_id: "str | None" = None


class RuntimeTaskHandler:
    def __init__(
        self,
        runtime,
        resolver: RunnableResolver,
        *,
        task_store,
        artifact_store,
        cancel_grace_seconds: float = 30.0,
    ) -> None:
        if task_store is None or artifact_store is None:
            raise TypeError("RuntimeTaskHandler requires task_store and artifact_store")
        self._runtime = runtime
        self._resolver = resolver
        self._task_store = task_store
        self._artifact_store = artifact_store
        # Grace window given to a cancelled Run to stop cleanly before the
        # handler force-cancels its coroutine.
        self._cancel_grace_seconds = cancel_grace_seconds

    async def execute(
        self, request: TaskRequest, context: TaskContext
    ) -> "TaskSuccess | TaskFailure":
        if context.delegated_scopes.is_empty:
            return TaskFailure(
                kind=TaskFailureKind.POLICY_DENIED,
                error_type="ScopeDenied",
                message="task delegated_scopes were revoked; execution refused",
            )
        try:
            runnable = _runnable_from_metadata(request.metadata)
            prompt = request.metadata["prompt"]
        except KeyError as exc:
            return TaskFailure(
                kind=TaskFailureKind.INVALID_INPUT,
                error_type=type(exc).__name__,
                message=f"missing input key: {exc}",
            )
        inp = RuntimeTaskInput(
            runnable=runnable,
            prompt=prompt,
            session_id=request.metadata.get("session_id"),
        )

        try:
            spec = await self._resolver.resolve(inp.runnable)
        except Exception as exc:  # noqa: BLE001
            return TaskFailure(
                kind=TaskFailureKind.PERMANENT,
                error_type=type(exc).__name__,
                message=f"runnable resolution failed: {exc}",
            )

        # Pin the resolved runnable on the task: the first attempt binds it, a
        # retry re-resolves and bind_runnable rejects a drift. This is what stops
        # a mapping change between attempts from silently re-running a different
        # agent. Best-effort for stores that don't implement bind_runnable (both
        # production backends do); such a store simply skips drift protection.
        if self._task_store is not None and hasattr(self._task_store, "bind_runnable"):
            try:
                await self._task_store.bind_runnable(
                    task_id=context.task_id,
                    attempt_id=context.attempt_id,
                    fencing_token=context.fencing_token,
                    worker_id=context.worker_id,
                    runnable_id=inp.runnable.id,
                    revision=inp.runnable.revision,
                    fingerprint=_fingerprint(spec),
                )
            except RunnableBindingError as exc:
                return TaskFailure(
                    kind=TaskFailureKind.PERMANENT,
                    error_type="RunnableDrift",
                    message=str(exc),
                )
            except TaskClaimLostError:
                return TaskFailure(
                    kind=TaskFailureKind.SUPERSEDED,
                    error_type="ClaimLost",
                    message="lease was lost before runnable bind; task reclaimed",
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

        result = await self._run_with_cancellation(spec, inp, run_id, context)
        if isinstance(result, TaskFailure):
            return result

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

    async def _run_with_cancellation(
        self, spec, inp, run_id: str, context: TaskContext
    ):
        """Drive ``runtime.run`` while racing the task's cancellation token.

        Returns the RunResult on success, or a :class:`TaskFailure` (CANCELLED
        or SIDE_EFFECT_UNKNOWN) on cancellation/failure. When cancellation fires
        first, the run is stopped via ``runtime.cancel`` and given a grace
        window before its coroutine is force-cancelled; the task then lands
        CANCELLED -- never SIDE_EFFECT_UNKNOWN -- because the run was stopped by
        us, not by an unknown side-effect failure."""
        run_task = asyncio.ensure_future(
            self._runtime.run(
                spec,
                inp.prompt,
                session_id=inp.session_id,
                run_id=run_id,
                user_id=context.principal.user_id,
                tenant_id=context.principal.tenant_id,
                context_metadata=_task_correlation(context),
            )
        )
        cancel_task = asyncio.ensure_future(context.cancellation.wait())
        try:
            await asyncio.wait(
                {run_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            # Worker shutdown: stop both and re-raise so the worker leaves the
            # task CLAIMED for recovery (its own CancelledError handling).
            cancel_task.cancel()
            run_task.cancel()
            await asyncio.gather(run_task, cancel_task, return_exceptions=True)
            raise

        if run_task.done() and not run_task.cancelled():
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            try:
                return await run_task
            except Exception as exc:  # noqa: BLE001
                # The run itself raised. If it was cancelled mid-flight, land
                # CANCELLED; only a spontaneous failure with no cancellation is
                # an unknown side-effect.
                if context.cancellation.is_set:
                    return TaskFailure(
                        kind=TaskFailureKind.CANCELLED,
                        error_type="TaskCancelled",
                        message="task execution was cancelled",
                        retryable=False,
                    )
                return TaskFailure(
                    kind=TaskFailureKind.SIDE_EFFECT_UNKNOWN,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )

        # Cancellation won the race (or the run is still pending): stop the run.
        cancel_task.cancel()
        await asyncio.gather(cancel_task, return_exceptions=True)
        task_principal = PrincipalContext(
            tenant_id=context.principal.tenant_id,
            user_id=context.principal.user_id,
            actor=ActorRef("task-attempt", context.attempt_id),
            scopes=ScopeSet.of("run.cancel:self"),
        )
        try:
            await self._runtime.cancel(run_id, principal=task_principal)
        except Exception as exc:  # authorization/cancellation failure is not success
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            return TaskFailure(kind=TaskFailureKind.SIDE_EFFECT_UNKNOWN,
                error_type=type(exc).__name__, message=str(exc), retryable=False)
        try:
            await asyncio.wait_for(run_task, timeout=self._cancel_grace_seconds)
        except asyncio.TimeoutError:
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
            return TaskFailure(kind=TaskFailureKind.SIDE_EFFECT_UNKNOWN,
                error_type="RuntimeCancelTimeout",
                message="runtime did not reach terminal state within cancel grace",
                retryable=False)
        except Exception as exc:  # the run failed while cancellation converged
            return TaskFailure(kind=TaskFailureKind.SIDE_EFFECT_UNKNOWN,
                error_type=type(exc).__name__, message=str(exc), retryable=False)
        return TaskFailure(
            kind=TaskFailureKind.CANCELLED,
            error_type="TaskCancelled",
            message="task execution was cancelled",
            retryable=False,
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


def _runnable_from_metadata(metadata: "Mapping[str, object]") -> RunnableRef:
    """Build a RunnableRef from task metadata. Prefers the structured
    ``runnable`` dict ``{"id", "revision"}``; falls back to the flat
    ``runnable_id`` (+ optional ``runnable_revision``) keys for older callers.
    Raises KeyError if neither form names a runnable."""
    if "runnable" in metadata:
        raw = metadata["runnable"]
        return RunnableRef(id=raw["id"], revision=raw.get("revision"))
    return RunnableRef(
        id=metadata["runnable_id"],
        revision=metadata.get("runnable_revision"),
    )


def _task_correlation(context: TaskContext) -> "dict[str, object]":
    """Build the context_metadata dict threaded into Runtime.run so the Run
    carries its Task lineage (job / task / attempt / fencing) and the logical
    workspace scope. ``workspace_key`` is a logical tag only -- the Runtime
    still mints a fresh physical workspace per run, so no raw path is exposed."""
    meta: "dict[str, object]" = {
        "job_id": context.job_id,
        "task_id": context.task_id,
        "attempt_id": context.attempt_id,
        "task_attempt_id": context.attempt_id,
        "fencing_token": context.fencing_token,
    }
    if context.principal.workspace_key is not None:
        meta["workspace_key"] = context.principal.workspace_key
    # Propagate the narrowed scopes + actor chain so the Run record carries the
    # task's effective permission lineage. An unrestricted ScopeSet is omitted
    # (the downstream Runtime reads a missing key as unrestricted).
    if not context.delegated_scopes.unrestricted:
        meta["delegated_scopes"] = tuple(context.delegated_scopes.values)
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
