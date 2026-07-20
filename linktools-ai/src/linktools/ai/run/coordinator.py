#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunCoordinator: the single application service for a Run's lifecycle --
create/start, pause (via the runner's own state machine), approve/reject,
resume, cancel, commit (via commit_coordinator), terminal convergence.

Runtime is a thin facade delegating every run-lifecycle method here; it keeps
only build/inspect/close + MCP-lifecycle ownership. Moved verbatim out of
Runtime -- no behavior change, only the container."""

import asyncio
from typing import TYPE_CHECKING, Any, Mapping

from .lifecycle import prepare_run
from .models import RunInput
from .preparation import RunPreparationCoordinator
from ..errors import SwarmError
from ..swarm.spec import SwarmSpec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .._runtime.build import RuntimeComponents
    from ..agent.spec import AgentSpec
    from ..identity.principal import PrincipalContext


class RunCoordinator:
    def __init__(self, components: "RuntimeComponents") -> None:
        self._components = components
        # Single owner of RunDefinitionSnapshot creation across every entry point.
        self._prepare = RunPreparationCoordinator(components.storage.run_definitions)
        # One-time crash-recovery guard: the File coordinator's journal is
        # replayed before the first run/resume so an interrupted pause/complete
        # is made consistent. No-op for coordinators without recovery (SQL) and
        # idempotent (recovery discards each journal it resolves). The lock +
        # double-check serialize concurrent first-callers, and the flag is set
        # only after recovery succeeds so a failed recovery can be retried.
        self._recovery_done = False
        self._recovery_lock = asyncio.Lock()

    async def _ensure_recovered(self) -> None:
        if self._recovery_done:
            return
        async with self._recovery_lock:
            if self._recovery_done:
                return
            coordinator = self._components.commit_coordinator
            recover = getattr(coordinator, "recover_incomplete_commits", None)
            if recover is not None:
                await recover()
            # Only flag done after recovery succeeds; a raise leaves the flag
            # False so the next entry point retries.
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
            storage=self._components.storage,
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
            # SwarmRunner owns the swarm snapshot creation (via the same
            # RunPreparationCoordinator) so both Runtime- and test-driven swarm
            # runs persist a snapshot. No double-create: Runtime does not
            # pre-create for swarm.
            return await self._components.swarm_runner.run(
                spec, RunInput(prompt=prompt), prepared.context, agents=agents
            )

        compiled = await self._components.compiler.compile(spec)
        # Persist the immutable run-definition snapshot AFTER compile (single
        # owner) so the resolved model bundle's revision is captured in the
        # manifest -- resume refuses if the provider config has since drifted.
        await self._prepare.prepare_agent_run(
            spec=spec,
            context=prepared.context,
            model_bundle=compiled.model_bundle,
        )
        return await self._components.runner.run(
            compiled, RunInput(prompt=prompt), prepared.context
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
            storage=self._components.storage,
            local_trusted_mode=self._components.settings.local_trusted_mode,
            run_id=run_id,
            principal=principal,
            action=action,
            authorization=self._components.authorization,
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
          AgentEngine.execute(). Transition the store to CANCELLING
          (distinguishes "cancel requested" from "actually cancelled"), then
          call ``run_controller.cancel(run_id)`` which (a) sets the
          CancellationToken so the runner's next execution-point check raises
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

        storage = self._components.storage
        controller = self._components.run_controller
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
            # metadata retain the old seeded/local behavior for migration.
            target = RunStatus.CANCELLING if record.worker_id else RunStatus.CANCELLED
            await storage.runs.transition(
                run_id, target, expected_version=record.version, **audit
            )
        self._components.metrics.counter("run_cancellation_requested_total")

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
            storage=self._components.storage,
            spec=spec,
            session_id=session_id,
            run_id=run_id,
            user_id=user_id,
            tenant_id=tenant_id,
            context_metadata=context_metadata,
        )

        compiled = await self._components.compiler.compile(spec)
        # Persist the immutable run-definition snapshot AFTER compile (single
        # owner) so the resolved model bundle's revision is captured in the
        # manifest for drift detection on resume.
        await self._prepare.prepare_agent_run(
            spec=spec,
            context=prepared.context,
            model_bundle=compiled.model_bundle,
        )
        async for event in self._components.runner.run_stream(
            compiled, RunInput(prompt=prompt), prepared.context
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

        return await ApprovalService(self._components.storage.approvals, self._components.authorization).approve(
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

        return await ApprovalService(self._components.storage.approvals, self._components.authorization).reject(
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
        storage = self._components.storage
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
            schema_registry=self._components.schema_registry,
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
                    bundle = await self._components.model_router.resolve(
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
        compiled = await self._components.compiler.compile(spec)
        # 12. Construct the full context, restoring the ORIGINAL identity from
        # the snapshot (user/tenant/workspace) + lineage from the record.
        from .._runtime.lifecycle import create_run_context

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
        await storage.runs.transition(
            run_id,
            RunStatus.RUNNING,
            expected_version=record.version,
        )
        # 14. Resume execution. The ORIGINAL user prompt is carried through so
        # the complete commit persists a real USER message (not an empty one).
        yield {"type": "resumed", "run_id": run_id}
        async for event in self._components.runner.run_stream(
            compiled,
            RunInput(prompt=record.input.prompt or ""),
            context,
            message_history=messages,
        ):
            yield event
