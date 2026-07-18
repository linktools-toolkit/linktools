#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolExecutor: consults PolicyEngine before a tool executes, translating
its decision into the corresponding domain error.

Approval flow: the executor mints an ``approval_id`` and raises ``RunPaused``
carrying every field needed to build the ApprovalRequest. AgentRunner's pause
handler then persists the ApprovalRequest (deduping on
``(run_id, tool_call_id)``) alongside the checkpoint save + WAITING_APPROVAL
transition + pause events -- on a cross-store-transactional Storage that
happens in one UnitOfWork, so a crash between "approval persisted" and "run
paused" is impossible. ``Runtime.resume`` completes the flow. The executor only
emits the domain signal (RunPaused); it never persists approval state itself,
so there is a single approval path."""

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..agent.approval import ApprovalStatus
from ..errors import (
    IdempotencyConflictError,
    IdempotencyInProgressError,
    RunPaused,
    ToolDeniedError,
)
from ..policy.engine import PolicyDecisionKind, PolicyEngine, ToolContext, ToolRequest
from .idempotency import IdempotencyStore, compute_request_hash

if TYPE_CHECKING:
    from ..agent.approval import ApprovalStore
    from .idempotency import ToolIdempotencyOptions
    from .models import ToolDescriptor
    from .policy import EffectiveToolPolicy

_LOGGER = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        *,
        policy: PolicyEngine,
        approval_store: "ApprovalStore | None" = None,
        run_id_resolver: "Callable[[ToolContext], str] | None" = None,
        idempotency_store: "IdempotencyStore | None" = None,
        retry_policy: Any = None,
        idempotency_options: "ToolIdempotencyOptions | None" = None,
        receipt_store: Any = None,
        tenant_id_resolver: "Callable[[ToolContext], str | None] | None" = None,
        metrics: Any = None,
    ) -> None:
        self._policy = policy
        self._approval_store = approval_store
        self._run_id_resolver = run_id_resolver
        self._idempotency_store = idempotency_store
        # When set, a long-running idempotent Handler is kept alive by a
        # background renew loop (the lease heartbeat); None disables the
        # heartbeat (the claim's initial lease from claim() governs).
        self._idempotency_options = idempotency_options
        self._receipt_store = receipt_store
        self._tenant_id_resolver = tenant_id_resolver
        self._metrics = metrics
        # Retry is policy-driven: only clearly-transient errors, and never a
        # mutating non-idempotent tool. DefaultRetryPolicy when none supplied.
        from .retry import DefaultRetryPolicy

        self._retry_policy = retry_policy or DefaultRetryPolicy()

    async def _invoke_handler(self, handler, arguments, timeout, claim):
        """Invoke the handler once with timeout and -- when idempotency options
        are configured and a claim is held -- a background lease-renew
        heartbeat. Returns the result or raises."""
        if self._idempotency_options is not None and claim is not None:
            return await self._run_with_heartbeat(
                handler(**arguments), claim, timeout
            )
        if timeout is not None:
            return await asyncio.wait_for(handler(**arguments), timeout=timeout)
        return await handler(**arguments)

    async def _run_with_heartbeat(self, coro, claim, timeout):
        """Run ``coro`` as a task while a background loop renews the claim's
        lease. If a renew fails the claim was stolen -- the handler task is
        cancelled (it must not keep producing side effects under a lost claim)
        and LostIdempotencyClaimError is raised."""
        from datetime import datetime, timezone

        from ..errors import LostIdempotencyClaimError

        options = self._idempotency_options
        handler_task = asyncio.ensure_future(coro)
        lease_lost = []
        # The claim's persisted expiry is the last point at which this worker
        # is allowed to produce side effects. Transient renew failures cannot
        # extend that deadline locally.
        confirmed_deadline = getattr(claim, "lease_expires_at", None)

        async def _heartbeat():
            nonlocal confirmed_deadline
            while True:
                await asyncio.sleep(options.heartbeat_seconds)
                try:
                    renewed = await self._idempotency_store.renew(
                        claim,
                        now=datetime.now(timezone.utc),
                        lease_seconds=options.lease_seconds,
                    )
                    confirmed_deadline = renewed.lease_expires_at
                except LostIdempotencyClaimError as exc:
                    if self._metrics is not None:
                        self._metrics.counter("tool_idempotency_lease_lost_total")
                    # Claim stolen / superseded: stop the handler so it does not
                    # keep producing side effects under a lost claim.
                    lease_lost.append(exc)
                    handler_task.cancel()
                    return
                except Exception:
                    # Transient store error: retry only while the last
                    # confirmed lease is still valid. Once it expires, stop
                    # the handler fail-closed; another worker may reclaim it.
                    if confirmed_deadline is not None and datetime.now(timezone.utc) >= confirmed_deadline:
                        exc = LostIdempotencyClaimError(
                            f"idempotent claim ({claim.scope}, {claim.key}) lease expired"
                        )
                        lease_lost.append(exc)
                        handler_task.cancel()
                        return
                    continue

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            if timeout is not None:
                return await asyncio.wait_for(handler_task, timeout=timeout)
            return await handler_task
        except asyncio.CancelledError:
            # Distinguish a lease-loss cancellation (heartbeat cancelled the
            # handler) from an external cancellation.
            if lease_lost:
                raise LostIdempotencyClaimError(
                    f"idempotent claim ({claim.scope}, {claim.key}) "
                    f"was stolen mid-execution"
                ) from lease_lost[0]
            raise
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass

    async def check(self, request: ToolRequest, context: ToolContext) -> None:
        decision = await self._policy.evaluate(request, context)
        if decision.kind == PolicyDecisionKind.DENY:
            raise ToolDeniedError(
                decision.reason or f"tool denied: {request.tool_name}"
            )
        if decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL:
            # Resume gate: if the approval_store already holds an APPROVED
            # request matching (run_id, tool_call_id), the call was approved
            # externally (e.g. via the pause UI) and is now being re-driven by
            # the model with the same tool_call_id -- let it through instead
            # of re-raising RunPaused.
            if await self._approved_binding_matches(request, context):
                return
            run_id = (
                self._run_id_resolver(context)
                if self._run_id_resolver is not None
                else context.run_id
            )
            # The executor only emits the domain signal (RunPaused); it does
            # NOT persist approval state. AgentRunner's pause handler persists
            # the ApprovalRequest atomically with the checkpoint/transition/
            # event writes, so there is a single approval path.
            tool_call_id = context.tool_call_id or str(uuid.uuid4())
            from ..agent.approval import compute_arguments_hash
            raise RunPaused(
                run_id=run_id,
                approval_id=str(uuid.uuid4()),
                tool_call_id=tool_call_id,
                tool_name=request.tool_name,
                reason=decision.reason,
                arguments=dict(request.arguments),
                binding={
                    key: context.metadata[key]
                    for key in (
                        "descriptor_fingerprint", "handler_revision",
                        "provider_revision", "policy_revision",
                        "capability_revision", "result_processor_revision",
                    )
                    if key in context.metadata
                } | {"arguments_hash": compute_arguments_hash(
                    request.tool_name, request.arguments
                )},
            )

    async def retry_tool_commit(self, claim, *, result_processor,
                                result_processor_revision, binding_fingerprint) -> Any:
        """Retry only the fenced commit for an EXECUTED receipt.

        This recovery path deliberately has no handler argument and therefore
        cannot re-drive an external side effect.
        """
        if self._idempotency_store is None:
            raise ToolDeniedError("tool commit recovery requires an idempotency store")
        record = await self._idempotency_store.get(claim.scope, claim.key)
        if record is None or record.status.value != "executed":
            raise ToolDeniedError("no EXECUTED tool receipt is available")
        if record.binding_fingerprint != binding_fingerprint:
            raise ToolDeniedError("tool binding fingerprint mismatch")
        if record.result_processor_revision != result_processor_revision:
            raise ToolDeniedError("result processor revision mismatch")
        if record.receipt_artifact_id is None:
            raise ToolDeniedError("EXECUTED receipt has no raw receipt")
        raw = record.result
        if self._receipt_store is not None and record.receipt_artifact_id:
            tenant_id = claim.scope.split(":", 1)[0]
            blob = await self._receipt_store.get(
                record.receipt_artifact_id, tenant_id=tenant_id)
            if blob is None:
                raise ToolDeniedError("raw execution receipt is unavailable")
            from ..json import canonical_json
            import json
            raw = json.loads(blob.decode("utf-8"))
        payload = result_processor(raw)
        if asyncio.iscoroutine(payload):
            payload = await payload
        await self._idempotency_store.complete(claim, payload)
        return payload

    async def _already_approved(
        self, request: ToolRequest, context: ToolContext
    ) -> bool:
        """True iff the approval_store has an APPROVED request matching
        ``(run_id, tool_call_id)`` -- the resume case. False when no store is
        wired, when the context carries no ``tool_call_id`` (no stable key to
        match on -- the uuid fallback path can't recognize a re-drive), or
        when no matching APPROVED request exists.

        Consults ``list_for_run`` (status-agnostic) rather than
        ``list_pending`` because the matching request is APPROVED, not PENDING
        -- ``list_pending`` would filter it out and the gate would never fire.
        """
        if self._approval_store is None or context.tool_call_id is None:
            return False
        run_id = (
            self._run_id_resolver(context)
            if self._run_id_resolver is not None
            else context.run_id
        )
        from ..agent.approval import compute_arguments_hash
        binding = {"tool_name": request.tool_name,
                   "arguments_hash": compute_arguments_hash(request.tool_name, request.arguments),
                   "schema_version": 1}
        binding.update({key: context.metadata.get(key) for key in (
            "descriptor_fingerprint", "handler_revision", "provider_revision",
            "policy_revision", "capability_revision", "result_processor_revision")})
        return await self._is_approved_binding(run_id, context.tool_call_id, binding=binding)

    async def _is_approved_binding(
        self, run_id: "str | None", tool_call_id: "str | None",
        *, binding: "dict[str, Any] | None" = None,
    ) -> bool:
        """Public resume gate: True iff the approval_store holds an APPROVED
        request matching ``(run_id, tool_call_id)``. Used by managed-path
        approval (pipeline REQUIRE_APPROVAL / policy.require_approval) to
        recognize a re-driven call after external approval -- without it, a
        stateless pipeline would re-raise RunPaused on every resume drive and
        the run could never complete. False when no store/tool_call_id wired."""
        if self._approval_store is None or run_id is None or tool_call_id is None or not binding:
            return False
        requests = await self._approval_store.list_for_run(run_id)
        from .binding import ToolExecutionBinding
        try:
            expected_binding = ToolExecutionBinding(**binding)
        except (TypeError, ValueError):
            return False
        return any(
            r.tool_call_id == tool_call_id and r.status is ApprovalStatus.APPROVED
            and r.binding_fingerprint == expected_binding.fingerprint()
            for r in requests
        )

    async def _approved_binding_matches(self, request: ToolRequest, context: ToolContext) -> bool:
        """Approval replay must match the current tool and argument binding.

        Legacy records without a binding are intentionally not reusable.
        """
        if self._approval_store is None or context.tool_call_id is None:
            return False
        run_id = self._run_id_resolver(context) if self._run_id_resolver else context.run_id
        requests = await self._approval_store.list_for_run(run_id)
        from ..agent.approval import compute_arguments_hash
        expected = compute_arguments_hash(request.tool_name, request.arguments)
        required = ("descriptor_fingerprint", "handler_revision", "provider_revision",
                    "policy_revision", "capability_revision", "result_processor_revision")
        if any(not isinstance(context.metadata.get(key), str) or not context.metadata.get(key)
               for key in required):
            return False
        from .binding import ToolExecutionBinding
        expected_binding = ToolExecutionBinding(schema_version=1,
            tool_name=request.tool_name, arguments_hash=expected,
            descriptor_fingerprint=context.metadata["descriptor_fingerprint"],
            handler_revision=context.metadata["handler_revision"],
            provider_revision=context.metadata["provider_revision"],
            policy_revision=context.metadata["policy_revision"],
            capability_revision=context.metadata["capability_revision"],
            result_processor_revision=context.metadata["result_processor_revision"])
        return any(
            r.tool_call_id == context.tool_call_id
            and r.status is ApprovalStatus.APPROVED
            and (
                r.schema_version >= 1
                and r.tool_name == request.tool_name
                and r.arguments_hash == expected
                and r.binding_fingerprint == expected_binding.fingerprint()
            )
            for r in requests
        )

    async def execute(
        self,
        request: ToolRequest,
        context: ToolContext,
        handler: "Callable[..., Awaitable[Any]]",
        *,
        descriptor: "ToolDescriptor",
        effective_policy: "EffectiveToolPolicy",
        timeout: "float | None" = None,
        max_retries: int = 0,
        idempotency_key: "str | None" = None,
        schema_version: str = "1",
        idempotency_scope: "str | None" = None,
        result_processor: "Callable[[Any], Any] | None" = None,
        result_processor_revision: str = "identity-v1",
        execution_binding: "Any | None" = None,
    ) -> Any:
        """Policy-check then run ``handler``, optionally with timeout/retry.

        ``descriptor`` and ``effective_policy`` are required: the retry decision
        reads ``descriptor.mutating`` and ``effective_policy.idempotent`` (a
        mutating non-idempotent tool is never retried), so a caller that omits
        them would let the executor invent a non-mutating default policy and
        retry a write it must not. Every real call site
        (ManagedToolAdapter) passes the finalized descriptor and policy it just
        resolved; a direct executor call with no descriptor/policy is a
        programming error, surfaced as ``TypeError`` by the signature itself.

        ``timeout`` wraps each handler call in :func:`asyncio.wait_for`; an
        :class:`asyncio.TimeoutError` is caught like any other exception
        (retried if attempts remain, raised otherwise). ``max_retries`` is the
        number of additional attempts after the first -- on any exception the
        call is retried up to ``max_retries`` times, after which the last error
        is re-raised. The defaults (``timeout=None``, ``max_retries=0``) mean a
        single handler call with no timeout wrap.

        ``idempotency_key`` enables persistent tool-call idempotency. When the
        executor was constructed with an
        ``idempotency_store`` AND ``idempotency_key`` is provided, the
        ``(scope, key)`` is ``(context.run_id, idempotency_key)``. The store
        survives process restart (dict-only idempotency is forbidden).
        Behavior per ``IdempotencyStatus``:

        - COMPLETED + same request hash -> cached ``result`` returned, the
          handler is NOT invoked.
        - RESERVED  + same request hash -> ``IdempotencyInProgressError``
          (another in-flight call owns the reservation).
        - FAILED    + same request hash -> retry: the handler runs again and
          the terminal transition overwrites the FAILED record.
        - Same (scope, key) with a different request hash -> reserve raises
          ``IdempotencyConflictError`` (same key reused for different args).
        - Fresh reservation -> handler runs; ``complete`` on success,
          ``fail`` on exception.

        ``schema_version`` (default ``"1"``) is folded into the request hash
        so a ToolSpec whose input contract changed shape bumps its
        schema_version and gets a fresh hash -- a stale idempotency record
        from before the change is never mistaken for a match. Callers that
        have a ``ToolSpec`` in hand should pass ``spec.schema_version`` here.

        With no ``idempotency_store`` (or no ``idempotency_key``), idempotency
        is off: the handler runs every call and nothing is persisted."""
        await self.check(request, context)
        if execution_binding is not None:
            if execution_binding.tool_name != request.tool_name:
                raise ToolDeniedError("tool execution binding tool name mismatch")
            if execution_binding.result_processor_revision != result_processor_revision:
                raise ToolDeniedError("tool execution binding processor revision mismatch")
        # Idempotency is scoped to context.run_id so the same idempotency_key
        # can be reused across different runs without colliding. ``scope`` is
        # part of the request_hash too, so a hash mismatch always reflects a
        # genuine difference in (tool, args, scope) -- not just (tool, args).
        # Fail closed: a tool declared idempotent (so a key was provided) but
        # run against a Storage with no IdempotencyStore must not silently run
        # non-idempotently -- that would let a replayed call execute twice.
        if idempotency_key is not None and self._idempotency_store is None:
            from ..errors import StorageCapabilityError

            raise StorageCapabilityError(
                f"tool {request.tool_name!r} is idempotent but no IdempotencyStore "
                f"is wired; refusing to run non-idempotently"
            )
        use_idempotency = (
            self._idempotency_store is not None and idempotency_key is not None
        )
        claim = None
        if use_idempotency:
            import uuid as _uuid

            from .idempotency import ClaimDisposition

        scope = idempotency_scope or context.run_id
        principal = getattr(context, "principal", None)
        if principal is not None:
            scope = ":".join(
                (
                    str(getattr(principal, "tenant_id", "")),
                    str(getattr(principal, "user_id", "")),
                    str(getattr(getattr(principal, "actor", None), "id", "")),
                    scope,
                )
            )
        if use_idempotency:
            request_hash = compute_request_hash(
                request.tool_name, request.arguments, scope,
                schema_version=schema_version,
            )
            owner_id = _uuid.uuid4().hex
            claim_kwargs = {"scope": scope, "key": idempotency_key,
                            "request_hash": request_hash, "owner_id": owner_id}
            if self._idempotency_options is not None:
                claim_kwargs["lease_seconds"] = self._idempotency_options.lease_seconds
            claim_result = await self._idempotency_store.claim(**claim_kwargs)
            disposition = claim_result.disposition
            if disposition is ClaimDisposition.REPLAY:
                # Replay returns the committed safe result. The receipt is
                # raw/audit-only and must never become the user-visible result.
                return claim_result.record.result
            if disposition is ClaimDisposition.IN_PROGRESS:
                raise IdempotencyInProgressError(
                    f"idempotent request in progress: key={idempotency_key!r}"
                )
            if disposition is ClaimDisposition.CONFLICT:
                raise IdempotencyConflictError(
                    f"idempotency key {idempotency_key!r} reused with a different request"
                )
            claim = claim_result.claim
        arguments = dict(request.arguments)
        from .retry import backoff_delay
        from ..errors import LostIdempotencyClaimError  # noqa: F401 (isinstance below)

        # --- Handler phase: retry transient HANDLER errors only. A failure
        # caught here means the side effect may NOT have happened yet, so
        # retrying the Handler is safe. The fenced idempotency commit is
        # intentionally OUT of this loop so a commit failure can never resolve
        # by re-invoking the Handler (the forbidden commit-failure-retries-handler pattern).
        last_error: "Exception | None" = None
        result: "Any" = None
        succeeded = False
        attempt = 0
        while True:
            try:
                result = await self._invoke_handler(handler, arguments, timeout, claim)
                succeeded = True
                break
            except Exception as exc:  # noqa: BLE001 - decide via retry policy
                if isinstance(exc, LostIdempotencyClaimError):
                    # Claim was stolen mid-execution: do not retry and do not
                    # fail the (already-lost) claim -- propagate to the caller.
                    raise
                last_error = exc
                if attempt >= max_retries:
                    break
                if not self._retry_policy.should_retry(
                    error=exc,
                    attempt=attempt,
                    policy=effective_policy,
                    descriptor=descriptor,
                ):
                    break
                # Transient: back off (cancellable) and retry.
                await asyncio.sleep(backoff_delay(attempt + 1))
                attempt += 1

        if not succeeded:
            # Handler never returned: no confirmed side effect, so failing the
            # claim is safe.
            if use_idempotency:
                from ..security.redact import redact_exception

                safe_error = (
                    redact_exception(last_error)
                    if last_error is not None
                    else "unknown error"
                )
                await self._idempotency_store.fail(claim, safe_error)
            raise last_error  # type: ignore[misc]

        # --- Commit phase: the Handler succeeded (its side effect happened).
        # Record the execution receipt, then land the fenced commit. A failure
        # here is NEVER resolved by re-running the Handler.
        receipt_artifact_id = None
        if use_idempotency:
            from ..errors import ToolCommitError

            if self._receipt_store is not None:
                from ..json import canonical_json
                tenant_id = self._tenant_id_resolver(context) if self._tenant_id_resolver else None
                if tenant_id:
                    try:
                        receipt = await self._receipt_store.put(
                            canonical_json(result).encode("utf-8"),
                            media_type="application/json", tenant_id=tenant_id,
                            metadata={"tool_name": request.tool_name,
                                      "request_hash": request_hash})
                        receipt_artifact_id = receipt.ref.id
                    except Exception as exc:
                        await self._idempotency_store.mark_unknown(claim)
                        raise ToolCommitError(
                            "raw execution receipt could not be persisted") from exc

        if use_idempotency:

            try:
                if receipt_artifact_id is None:
                    try:
                        await self._idempotency_store.mark_executed(claim, result,
                            binding_fingerprint=(execution_binding.fingerprint() if execution_binding else None),
                            result_processor_revision=result_processor_revision)
                    except TypeError:
                        await self._idempotency_store.mark_executed(claim, result)
                else:
                    try:
                        await self._idempotency_store.mark_executed(
                            claim, result, receipt_artifact_id=receipt_artifact_id,
                            binding_fingerprint=(execution_binding.fingerprint() if execution_binding else None),
                            result_processor_revision=result_processor_revision)
                    except TypeError:
                        await self._idempotency_store.mark_executed(
                            claim, result, receipt_artifact_id=receipt_artifact_id)
            except Exception as receipt_exc:  # noqa: BLE001 - never re-run handler
                if self._metrics is not None:
                    self._metrics.counter("tool_side_effect_unknown_total")
                # No receipt could be stored, so the outcome is unknowable.
                # Best-effort UNKNOWN (the side effect happened); never re-drive.
                try:
                    await self._idempotency_store.mark_unknown(claim)
                except Exception:  # noqa: BLE001 - best-effort
                    pass
                raise ToolCommitError(
                    f"tool {request.tool_name!r} Handler succeeded but its "
                    f"execution receipt could not be stored; marked UNKNOWN"
                ) from receipt_exc
        try:
            safe_result = result if result_processor is None else result_processor(result)
            if asyncio.iscoroutine(safe_result):
                safe_result = await safe_result
        except Exception:
            # EXECUTED is intentionally retained for processor retry.
            raise
        if use_idempotency:
            try:
                # The execution receipt is already persisted before this
                # commit. Tenant-scoped ArtifactStore receipts are used for
                # production runs; the raw value remains for local legacy
                # stores that do not carry a Tenant.
                await self._idempotency_store.complete(claim, safe_result)
            except Exception as commit_exc:  # noqa: BLE001 - never re-run handler
                if self._metrics is not None:
                    self._metrics.counter("tool_commit_retry_total")
                # The receipt (EXECUTED) landed, so the outcome is recoverable:
                # leave the record EXECUTED for retry_tool_commit / startup
                # recovery to re-attempt the commit. Do NOT re-run the Handler.
                raise ToolCommitError(
                    f"tool {request.tool_name!r} Handler succeeded but its "
                    f"result commit could not be confirmed; record left EXECUTED "
                    f"for recovery"
                ) from commit_exc
        return safe_result
