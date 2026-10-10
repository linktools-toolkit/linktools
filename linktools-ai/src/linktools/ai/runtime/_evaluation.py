#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset publication and persisted evaluation plans over native Task graphs."""

import asyncio
import json
import sys
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, TypeVar

from linktools.core import environ
from pydantic_ai.messages import BinaryContent

from ..agent import AgentInputCaptureRef

from ..asset import AssetStoreReader, AssetVersionRef
from ..core import (
    AuthorizationAction, AuthorizationPolicy, CursorPayload, CursorSigner,
    ExecutionStatus, IdempotencyStatus, JsonValue, Page, Principal,
    ResourceKind, ResourceRef, TaskStatus, UsageMetrics, canonical_json_bytes, canonical_sha256,
    idempotency_key_digest, normalize_json_value, principal_identity_payload, validate_idempotency_key,
    validate_page_limit,
)
from ..errors import AIError, ErrorCode, ObservationError
from ..evaluation import (
    EvaluationPurgeResult,
    CaseContract, CaseRef, CaseSpec, ComparisonReadCutoff,
    ComparisonReport, ComparisonSpec, DatasetContract, DatasetRef, DatasetSpec,
    EvaluationIssue, EvaluationManifest, EvaluationPolicy, EvaluationProgress,
    EvaluationReadCutoff, EvaluationReport, EvaluationView, EvidenceAttachmentRef, EvidenceBundle,
    EvidenceRef, ExecutionSubjectRef, ExecutionTargetEvidence, GraphOutputItem,
    GraphSubjectRef, GraphTargetEvidence, HumanScoreRequest, InlineValue,
    RescoreRequest, ScoreAttemptView, ScoreBundle, ScoreFilter, ScoreNotApplicable, ScoreSelection,
    ModelUsage, PriceTable, estimate_model_budget,
    ScorerContract, ScoringInput, SlotDispositionView, StartEvaluationRequest,
    TargetTrialRef, TrialFilter, TrialPlan, TrialView, build_comparison_report,
    build_evaluation_report, capture_mapping, evaluation_completion,
)
from ..task import (
    Task, TaskGraph, TaskGraphService, TaskGraphState, TaskInputSupplyRequest,
    TaskNode, TaskNodeContext, TaskNodeResultRef, TaskRef, TaskGraphLimits, TaskSubmissionCancellation,
)
from ._agent_task_input import AgentTaskInput
from ._evaluation_compile import EvaluationCompiler, task_contract
from ._evaluation_retention import EvaluationRetention, require_evaluation_content
from ._evaluation_scope import EvaluationTrialScope, EvaluationTrialScopeCallback, _EnteredTrialScope
from ._input import input_intent
from ._input_capture import CaptureInputRequest
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .service_api import ExecutionService, UsageSummary, TaskGraphRunEvent
from ._wait import WaitResult
from ._observation import (
    _ObservationSession, _wait, _validate_wait, _await_stream_cleanup,
    _drain_stream_tasks, _is_observation_cleanup, _report_observation_error,
)
from ._watch_cursor import (
    decode_evaluation_watch_cursor, encode_evaluation_watch_cursor, decode_graph_watch_cursor,
)
from .state import RuntimeDomain, RuntimeRetentionMode, RuntimeStorage, SnapshotExclusiveGuard
from .state._contracts import IdempotencyRecord
from .state._evaluation_records import (
    EvaluationLaunchIntent, EvaluationRecord, EvaluationSlotDisposition,
    EvaluationTrialEvidence, EvaluationHumanDecision,
)

if TYPE_CHECKING:
    from ._input_capture import RuntimeInputCaptures
    from ._runtime_history import RuntimeHistory
    from ._tasks import TaskEngine

AppT = TypeVar("AppT")
ScopeAppT = TypeVar("ScopeAppT")
_logger = environ.get_logger("ai.runtime.evaluation")
_TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED})
_RECORDED = frozenset({"valid", "error", "not_applicable", "not_attempted"})


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _target_slot(trial: TargetTrialRef) -> str:
    return f"target:{trial.trial_id}"


def _score_slot(trial: TargetTrialRef, scorer: str) -> str:
    return f"score:{trial.trial_id}:{scorer}"


def _completion(
    record: EvaluationRecord, trials: tuple[TrialView, ...], scores: tuple[ScoreAttemptView, ...],
    *, blocked: bool = False,
) -> str:
    budget_stopped = record.gate == "closed_budget" and any(
        item.disposition.reason_code == "closed_budget" for item in record.dispositions)
    return evaluation_completion(
        trials if record.manifest.kind == "experiment" else (), scores,
        pending_launches=any(not item.released for item in record.intents),
        blocked=blocked or any(not item.disposition.terminal for item in record.dispositions),
        cancellation_requested=record.gate == "closed_cancel", budget_stopped=budget_stopped,
    )


class RuntimeEvaluations:
    """Own immutable evaluation plans; delegate all execution to TaskEngine."""

    def __init__(
        self, namespace: str, storage: RuntimeStorage, authorization: AuthorizationPolicy,
        execution: ExecutionService, graph: TaskGraphService,
        captures: "RuntimeInputCaptures", history: "RuntimeHistory", *,
        cursor_signer: CursorSigner, asset_readers: tuple[AssetStoreReader, ...] = (),
    ) -> None:
        self._namespace = namespace
        self._storage = storage
        self._state = storage.evaluation.records
        self._authorization = authorization
        self._execution = execution
        self._graph = graph
        self._history = history
        self._captures = captures
        self._compiler = EvaluationCompiler(captures, namespace)
        self._objects = storage.object_store(RuntimeDomain.EVALUATION)
        self._object_keys = RuntimeObjectKeyFactory(namespace)
        self._cursor_signer = cursor_signer
        self._asset_readers = asset_readers
        self._retention = EvaluationRetention(storage, authorization, self._cancel_submission, captures,
                                             release_scopes=self._close_trial_scopes)
        self._watchers: dict[str, asyncio.Task[None]] = {}
        self._trial_scopes: dict[tuple[str, str], _EnteredTrialScope] = {}
        self._closed = False
        self._recorder = Task("evaluation.record.v1", self._record_score, effect_policy="replay_safe")

    def _bind_observation(
        self,
        watch_graph: Callable[[str, Principal, str | None, bool, asyncio.Event | None], AsyncIterator[TaskGraphRunEvent]],
        replay_graph: Callable[[str, Principal, str | None, bool], Awaitable[AsyncIterator[TaskGraphRunEvent]]],
        register: Callable[[_ObservationSession], None],
        release: Callable[[_ObservationSession], None],
    ) -> None:
        self._watch_graph = watch_graph
        self._replay_graph = replay_graph
        self._register_observation = register
        self._release_observation = release

    async def purge_expired(
        self, *, principal: Principal, now: datetime,
        exclusive: SnapshotExclusiveGuard, limit: int = 100,
    ) -> EvaluationPurgeResult:
        return await self._retention.purge_expired(
            principal=principal, now=now, exclusive=exclusive, limit=limit)

    async def close(self) -> None:
        """Stop coordination and reject new execution and control requests."""
        self._closed = True
        tasks = tuple(self._watchers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._watchers.clear()
        await self._close_trial_scopes()

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, retryable=False)

    async def _authorize(
        self, principal: Principal, action: AuthorizationAction, identity: str,
        *, owner: str | None = None,
    ) -> None:
        await self._authorization.authorize(principal, action, ResourceRef(
            ResourceKind.EVALUATION, identity, self._storage.tenant_id, owner))

    async def _record(
        self, experiment_id: str, principal: Principal,
        action: AuthorizationAction = AuthorizationAction.EVALUATION_READ,
        *, allow_expired: bool = False,
    ) -> EvaluationRecord:
        await self._authorize(principal, action, experiment_id)
        record = await self._state.get(experiment_id, tenant_id=principal.tenant_id)
        if record is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        await self._authorize(principal, action, experiment_id,
                              owner=record.manifest.principal.principal_id)
        if not allow_expired and record.metadata_expires_at is not None and record.metadata_expires_at <= _now():
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE, safe_details={"reason": "metadata_expired"})
        return record

    async def _require_content(self, record: EvaluationRecord, principal: Principal) -> None:
        seen: set[str] = set()
        now = _now()
        while True:
            if record.experiment_id in seen:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            seen.add(record.experiment_id)
            require_evaluation_content(record, now=now)
            source = record.manifest.source_experiment_id
            if source is None:
                return
            record = await self._record(source, principal, allow_expired=True)

    async def publish_dataset(
        self, spec: DatasetSpec, *, principal: Principal, idempotency_key: str,
        content_expires_at: datetime | None = None,
    ) -> DatasetRef:
        if content_expires_at is not None and (content_expires_at.tzinfo is None or content_expires_at <= _now()):
            raise AIError(ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE)
        validate_idempotency_key(idempotency_key)
        await self._authorize(principal, AuthorizationAction.EVALUATION_DATASET_PUBLISH, spec.ref.id,
                              owner=await self._state.dataset_owner(spec.ref))
        await self._state.get_dataset(spec.ref)
        cases = []
        for item in spec.cases:
            ref = item.ref if isinstance(item, CaseSpec) else item
            await self._authorize(principal, AuthorizationAction.EVALUATION_DATASET_PUBLISH, ref.dataset_id,
                                  owner=await self._state.case_owner(ref))
            await self._state.get_case(ref)
            case = (await self._compiler.case(item, principal=principal, content_expires_at=content_expires_at) if isinstance(item, CaseSpec)
                    else await self._case(item, principal=principal))
            if case.expected is not None:
                await self._value(case.expected)
            cases.append(case)
        kinds = {case.input_kind for case in cases}
        if len(kinds) != 1:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "dataset input kinds differ")
        dataset = DatasetContract(spec.ref, tuple(case.ref for case in cases), kinds.pop(), spec.selection,
            tuple(case.expected for case in cases if isinstance(case.expected, AssetVersionRef)))
        digest = canonical_sha256({"principal": principal_identity_payload(principal),
            "dataset": dataset.to_mapping(), "cases": [case.to_mapping() for case in cases],
            "content_expires_at": None if content_expires_at is None else content_expires_at.isoformat()})
        now = _now()
        return await self._state.publish_dataset(dataset, tuple(cases), owner_principal_id=principal.principal_id,
            content_expires_at=content_expires_at, idempotency=IdempotencyRecord(
            "evaluation.dataset.publish", idempotency_key_digest(idempotency_key), digest,
            ResourceKind.EVALUATION, spec.ref.id, IdempotencyStatus.COMPLETED, dataset.digest,
            None, now, now))

    async def get_dataset(self, ref: DatasetRef, *, principal: Principal) -> DatasetContract:
        await self._authorize(principal, AuthorizationAction.EVALUATION_DATASET_READ, ref.id)
        owner = await self._state.dataset_owner(ref)
        await self._authorize(principal, AuthorizationAction.EVALUATION_DATASET_READ, ref.id, owner=owner)
        value = await self._state.get_dataset(ref)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return value

    async def _case(self, ref: CaseRef, *, principal: Principal | None = None) -> CaseContract:
        if principal is not None:
            await self._authorize(principal, AuthorizationAction.EVALUATION_DATASET_READ, ref.dataset_id,
                                  owner=await self._state.case_owner(ref))
        value = await self._state.get_case(ref)
        if value is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return value

    async def list_cases(
        self, ref: DatasetRef, *, principal: Principal,
        cursor: str | None = None, limit: int = 100,
    ) -> Page[CaseContract]:
        dataset = await self.get_dataset(ref, principal=principal)
        digest = canonical_sha256({"dataset": dataset.to_mapping(), "principal": principal_identity_payload(principal)})
        position = self._cursor_position(cursor, "evaluation.cases", digest, principal) if cursor else "0"
        index = int(position)
        validate_page_limit(limit)
        cases = tuple([await self._case(item) for item in dataset.ordered_case_refs[index:index + limit]])
        next_cursor = self._cursor("evaluation.cases", digest, str(index + limit), principal) if index + limit < len(dataset.ordered_case_refs) else None
        return Page(cases, next_cursor)

    def _engine(self, engine: "TaskEngine[AppT]") -> "TaskEngine[AppT]":
        if engine.runtime.evaluations is not self:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "engine belongs to another Runtime")
        return engine

    def _validate_storage(self, policy: EvaluationPolicy) -> None:
        domains = (RuntimeDomain.EVALUATION, RuntimeDomain.TASK, RuntimeDomain.EXECUTION,
                   RuntimeDomain.RECOVERY, RuntimeDomain.ARTIFACT)
        for domain in domains:
            retention = self._storage.plan.route(domain).retention
            if retention is RuntimeRetentionMode.TRANSIENT or (
                    retention is RuntimeRetentionMode.VOLATILE and not policy.allow_volatile):
                raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "evaluation evidence requires retained storage")

    async def start(
        self, request: StartEvaluationRequest, *, engine: "TaskEngine[AppT]",
        trial_scope: EvaluationTrialScopeCallback[ScopeAppT] | None = None,
    ) -> "EvaluationRun":
        record = await self._reserve_start(request, engine=engine, scope_required=trial_scope is not None)
        self._watch(record.experiment_id, engine, request.principal, trial_scope=trial_scope)
        return EvaluationRun(self, record.experiment_id, request.principal)

    async def cancel_admission(
        self, request: StartEvaluationRequest, *, engine: "TaskEngine[AppT]",
        trial_scope: EvaluationTrialScopeCallback[ScopeAppT] | None = None,
    ) -> "EvaluationRun":
        """Cancel an admitted run or reserve its start request with a closed gate."""
        self._ensure_open()
        await self._authorize(request.principal, AuthorizationAction.EVALUATION_CANCEL,
                              idempotency_key_digest(request.idempotency_key))
        record = await self._reserve_start(request, engine=engine, cancelled=True,
                                           scope_required=trial_scope is not None)
        await self._cancel(record.experiment_id, request.principal, request.idempotency_key)
        return EvaluationRun(self, record.experiment_id, request.principal)

    async def _reserve_start(
        self, request: StartEvaluationRequest, *, engine: "TaskEngine[AppT]",
        cancelled: bool = False, scope_required: bool = False,
    ) -> EvaluationRecord:
        self._ensure_open()
        bound = self._engine(engine)
        spec, principal = request.spec, request.principal
        identity = idempotency_key_digest(request.idempotency_key)
        await self._authorize(principal, AuthorizationAction.EVALUATION_RUN, identity)
        previous = await self._storage.evaluation.idempotency.get(
            "evaluation.run", identity, tenant_id=principal.tenant_id)
        existing = None
        if previous is not None:
            existing = await self._record(previous.resource_id, principal)
            require_evaluation_content(existing, now=_now())
        self._validate_storage(spec.policy)
        if spec.policy.price_table is not None:
            PriceTable.from_mapping(await self._value(spec.policy.price_table), currency=spec.policy.currency)
        dataset = await self.get_dataset(spec.dataset, principal=principal)
        source_expires_at = await self._state.dataset_expiry(spec.dataset)
        cases = tuple([await self._case(ref) for ref in dataset.ordered_case_refs])
        total = len(cases) * len(spec.candidates) * spec.repetitions
        if total > spec.policy.max_trials:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "trial limit exceeded")
        candidates = tuple([await self._compiler.candidate(candidate, engine=bound, principal=principal,
            policy=spec.policy) for candidate in spec.candidates])
        scorers = tuple(self._compiler.scorer(scorer, engine=bound, policy=spec.policy) for scorer in spec.scorers)
        for scorer in scorers:
            if scorer.rubric is not None:
                await self._value(scorer.rubric)
        plans = tuple(TrialPlan(canonical_sha256({"case": case.ref.to_mapping(), "candidate": candidate.slot_id,
            "repetition": repetition}), case.ref, candidate.slot_id, repetition)
            for case in cases for candidate in candidates for repetition in range(1, spec.repetitions + 1))
        experiment_id = uuid.uuid4().hex if previous is None else previous.resource_id
        if cancelled:
            await self._authorize(principal, AuthorizationAction.EVALUATION_CANCEL, experiment_id,
                owner=principal.principal_id if existing is None else existing.manifest.principal.principal_id)
        owned_captures = []
        for candidate in candidates:
            for case in cases:
                await self._compiler.graph(candidate, case, graph_id="evaluation-preflight", principal=principal,
                    input_mode=spec.input_mode, owner_id=experiment_id, materialize=False, owned_captures=owned_captures)
        manifest = EvaluationManifest(experiment_id, "experiment", None, spec.dataset,
            candidates, scorers, plans, (), spec.policy, spec.input_mode, principal,
            trial_scope_required=scope_required)
        payload = manifest.to_mapping()
        payload.pop("experiment_id")
        digest = canonical_sha256(payload)
        now = _now()
        deadlines = ([] if source_expires_at is None else [source_expires_at])
        if spec.policy.content_retention_seconds is not None:
            deadlines.append(now + timedelta(seconds=spec.policy.content_retention_seconds))
        record = await self._state.reserve_experiment(EvaluationRecord(
            manifest, manifest.digest, digest, identity,
            "closed_cancel" if cancelled else "open", 0, now, now,
            content_expires_at=min(deadlines) if deadlines else None,
            metadata_expires_at=None if spec.policy.metadata_retention_seconds is None else now + timedelta(seconds=spec.policy.metadata_retention_seconds),
            owned_input_captures=tuple(dict.fromkeys(owned_captures))))
        return record

    async def get(self, experiment_id: str, *, principal: Principal) -> "EvaluationRun":
        await self._record(experiment_id, principal)
        return EvaluationRun(self, experiment_id, principal)

    async def reconcile(
        self, experiment_id: str, *, engine: "TaskEngine[AppT]", principal: Principal,
        idempotency_key: str,
        trial_scope: EvaluationTrialScopeCallback[ScopeAppT] | None = None,
    ) -> "EvaluationRun":
        self._ensure_open()
        validate_idempotency_key(idempotency_key)
        record = await self._record(experiment_id, principal, AuthorizationAction.EVALUATION_RECONCILE)
        require_evaluation_content(record, now=_now())
        bound = self._engine(engine)
        self._require_trial_scope(record, trial_scope)
        await self._validate_definitions(record.manifest, bound)
        previous_scopes = set(self._trial_scopes)
        try:
            for intent in record.intents:
                if intent.released:
                    continue
                if record.gate == "closed_cancel":
                    await self._cancel_intent(record, intent, principal)
                    continue
                async with self._trial_engine(record, intent, bound, trial_scope) as selected:
                    result = await selected.start_prepared(intent.submission)
                    if result.admitted:
                        await self._state.settle_intent(experiment_id, intent.slot_id,
                                                       confirmed=True, released=False)
                    if result.admitted and result.result.status not in _TERMINAL:
                        run = await selected.get(intent.submission.graph.graph_id, principal=principal)
                        try:
                            await run.recover(idempotency_key=f"{idempotency_key}:{intent.slot_id}")
                        except AIError as error:
                            if error.code is not ErrorCode.TASK_NOT_READY:
                                raise
                            if (await run.state()).status not in _TERMINAL:
                                raise
                if not result.admitted:
                    await self._close_trial_scope((experiment_id, intent.slot_id))
                    await self._state.settle_intent(experiment_id, intent.slot_id,
                                                   confirmed=False, released=True)
                if result.admitted and result.result.status not in _TERMINAL:
                    if intent.scorer_slot_id is not None:
                        state = await self._graph.state(intent.submission.graph.graph_id, principal=principal)
                        await self._resume_decision(record, intent, state)
            await self._state.update(experiment_id, lambda value: replace(value,
                dispositions=tuple(item for item in value.dispositions if item.disposition.terminal)))
            self._watch(experiment_id, bound, principal, trial_scope=trial_scope)
        except BaseException:
            await self._close_trial_scopes(experiment_id, keys=tuple(
                key for key in self._trial_scopes if key not in previous_scopes))
            raise
        return EvaluationRun(self, experiment_id, principal)

    async def _validate_definitions(self, manifest: EvaluationManifest, engine: "TaskEngine") -> None:
        declarations = ([definition for candidate in manifest.candidates for definition in candidate.definition_contracts]
                        if manifest.kind == "experiment" else [])
        declarations += [scorer.task_contract for scorer in manifest.scorers if scorer.task != TaskRef.deferred_input()]
        for definition in declarations:
            ref = TaskRef(definition["id"], definition["revision"])
            if task_contract(engine, ref) != dict(definition):
                raise AIError(ErrorCode.BINDING_CONFLICT)
        for candidate in manifest.candidates if manifest.kind == "experiment" else ():
            if candidate.graph_template is None:
                continue
            template = candidate.graph_template.template
            if template is None:
                raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
            if any(node.expander is not None for node in template.nodes):
                definitions = tuple(task_contract(engine, task.ref) for task in engine.definitions)
                expanders = tuple({"version": 1, "id": item.id, "revision": item.revision} for item in engine.expanders)
                if definitions != candidate.definition_contracts or expanders != template.expander_contracts:
                    raise AIError(ErrorCode.BINDING_CONFLICT)

    def _require_trial_scope(
        self, record: EvaluationRecord, trial_scope: EvaluationTrialScopeCallback | None,
    ) -> None:
        if record.manifest.trial_scope_required != (trial_scope is not None):
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "trial scope mode differs from the admitted evaluation")

    @asynccontextmanager
    async def _trial_engine(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, engine: "TaskEngine",
        trial_scope: EvaluationTrialScopeCallback | None,
    ) -> AsyncIterator["TaskEngine"]:
        self._require_trial_scope(record, trial_scope)
        if trial_scope is None:
            yield engine if intent.scorer_slot_id is None else engine.with_definitions(self._recorder)
            return
        self._ensure_open()
        key = (record.experiment_id, intent.slot_id)
        entered = self._trial_scopes.get(key)
        if entered is None:
            @asynccontextmanager
            async def open_scope() -> AsyncIterator["TaskEngine"]:
                planning = engine if intent.scorer_slot_id is None else engine.with_definitions(self._recorder)
                created = await planning._prepare_trial_submission(intent.submission)
                async with trial_scope(EvaluationTrialScope(
                    record.experiment_id, intent.trial, intent.slot_id, record.manifest.principal,
                    intent.submission, intent.scorer_slot_id, newly_prepared=created,
                )) as selected:
                    yield selected

            entered = _EnteredTrialScope(open_scope())
            self._trial_scopes[key] = entered
        try:
            await entered.engine()
            with entered.borrow_engine() as selected:
                self._ensure_open()
                if selected is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                runtime = selected.runtime
                other = runtime.evaluations
                if (other is self or runtime.namespace != self._namespace
                        or runtime.tenant_id != self._storage.tenant_id
                        or other._storage is self._storage or other._storage.plan != self._storage.plan):
                    raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "trial scope requires independent Runtime ownership over shared storage")
                for domain in (RuntimeDomain.EVALUATION, RuntimeDomain.TASK, RuntimeDomain.EXECUTION,
                               RuntimeDomain.RECOVERY, RuntimeDomain.ARTIFACT):
                    if (other._storage.plan.route(domain).retention is not RuntimeRetentionMode.DURABLE
                            or other._storage.object_store(domain).store_id != self._storage.object_store(domain).store_id):
                        raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "trial scope requires shared retained evidence stores")
                await self._validate_definitions(record.manifest, selected)
                yield selected if intent.scorer_slot_id is None else selected.with_definitions(self._recorder)
        except BaseException:
            await self._close_trial_scope(key)
            raise

    async def _close_trial_scope(self, key: tuple[str, str]) -> None:
        entered = self._trial_scopes.get(key)
        if entered is not None:
            await entered.close()
            self._trial_scopes.pop(key, None)

    async def _close_trial_scopes(
        self, experiment_id: str | None = None, *, keys: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        selected = tuple(self._trial_scopes) if keys is None else keys
        results = await asyncio.gather(*(self._close_trial_scope(key) for key in selected
            if experiment_id is None or key[0] == experiment_id), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def _watch(
        self, experiment_id: str, engine: "TaskEngine | None", principal: Principal,
        *, trial_scope: EvaluationTrialScopeCallback | None = None,
    ) -> None:
        self._ensure_open()
        current = self._watchers.get(experiment_id)
        if current is None or current.done():
            task = asyncio.create_task(
                self._coordinate(experiment_id, engine, principal, trial_scope=trial_scope))
            task.add_done_callback(self._coordinate_done)
            self._watchers[experiment_id] = task

    def _coordinate_done(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            _logger.warning("evaluation coordinator failed: exception_type=%s", type(error).__name__)

    async def _coordinate(
        self, experiment_id: str, engine: "TaskEngine | None", principal: Principal,
        *, trial_scope: EvaluationTrialScopeCallback | None = None,
    ) -> None:
        try:
            while not self._closed:
                await self._tick(experiment_id, engine, principal, trial_scope=trial_scope)
                record = await self._record(experiment_id, principal, allow_expired=True)
                if record.gate != "open" and all(item.released for item in record.intents):
                    try:
                        await self._require_content(record, principal)
                    except AIError as error:
                        if error.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE:
                            return
                        raise
                view = await self._inspect(experiment_id, principal)
                if view.completion in {"complete", "cancelled", "needs_attention"}:
                    return
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            _logger.warning("evaluation needs reconciliation: experiment=%s", experiment_id, exc_info=environ.debug)
            await self._disposition(experiment_id, "coordinator", str(error.code) if isinstance(error, AIError) else type(error).__name__, retryable=True)
        finally:
            try:
                await self._close_trial_scopes(experiment_id)
            except Exception as error:
                await self._disposition(experiment_id, "coordinator",
                    str(error.code) if isinstance(error, AIError) else type(error).__name__, retryable=True)
                raise
            finally:
                self._watchers.pop(experiment_id, None)

    async def read_evidence(self, evidence_ref: EvidenceRef, *, principal: Principal) -> EvidenceBundle:
        if evidence_ref.namespace != self._namespace or evidence_ref.tenant_id != self._storage.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorize(principal, AuthorizationAction.EVALUATION_EVIDENCE_READ, evidence_ref.evidence_id)
        bundle = await self._state.read_evidence(evidence_ref)
        if bundle is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        source_record = await self._record(bundle.trial.target_experiment_id, principal, AuthorizationAction.EVALUATION_EVIDENCE_READ)
        require_evaluation_content(source_record, now=_now())
        if bundle.ref != evidence_ref or bundle.digest != evidence_ref.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for attachment in bundle.attachments:
            if isinstance(attachment, AssetVersionRef):
                await self._asset(attachment)
            else:
                reference = attachment.object_ref
                if reference.key != self._object_keys.key(RuntimeDomain.EVALUATION, principal.tenant_id, reference.digest):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                stat = await self._objects.stat(reference.key)
                if stat is None:
                    raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, "retained evidence attachment is unavailable")
                if stat.digest != reference.digest or stat.size != reference.size:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return bundle

    async def read_evidence_attachment(
        self, evidence_ref: EvidenceRef, attachment_id: str, *, principal: Principal,
    ) -> bytes:
        bundle = await self.read_evidence(evidence_ref, principal=principal)
        attachment = next((item for item in bundle.attachments if isinstance(item, EvidenceAttachmentRef)
                           and item.attachment_id == attachment_id), None)
        if attachment is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        return await read_runtime_object(self._objects, attachment.object_ref)

    def _evidence_ids(self, evidence: EvidenceBundle) -> tuple[str, ...]:
        return (evidence.ref.evidence_id, *(item.attachment_id if isinstance(item, EvidenceAttachmentRef)
                                           else item.key.id for item in evidence.attachments))

    async def _asset(self, ref: AssetVersionRef) -> bytes:
        for reader in self._asset_readers:
            try:
                return (await reader.read_versions((ref,)))[0]
            except AIError as error:
                if error.code not in {ErrorCode.STORAGE_NOT_FOUND, ErrorCode.ASSET_VERSION_LAYER_UNKNOWN}:
                    raise
        raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, "evaluation Asset version unavailable")

    async def _value(self, value: InlineValue | AssetVersionRef | None) -> JsonValue:
        if value is None:
            return None
        if isinstance(value, InlineValue):
            return value.value
        return normalize_json_value(json.loads((await self._asset(value)).decode("utf-8")))

    async def _trials(
        self, record: EvaluationRecord, principal: Principal,
    ) -> tuple[tuple[TrialView, ...], dict[str, int]]:
        if record.manifest.kind == "score_only":
            source = await self._record(record.manifest.source_experiment_id, principal)
            trials, revisions = await self._trials(source, principal)
            wanted = set(record.manifest.source_trials)
            refs = {item.trial: item.evidence_ref for item in record.evidence}
            return tuple(replace(item, evidence_ref=refs.get(item.trial, item.evidence_ref))
                         for item in trials if item.trial in wanted), revisions
        intents = {item.slot_id: item for item in record.intents}
        dispositions = {item.slot_id: item.disposition for item in record.dispositions}
        evidence = {item.trial: item.evidence_ref for item in record.evidence}
        candidates = {item.slot_id: item for item in record.manifest.candidates}
        result = []
        revisions = {f"evaluation:{record.experiment_id}": record.revision}
        for plan in record.manifest.trials:
            trial = TargetTrialRef(record.experiment_id, plan.trial_id)
            slot = _target_slot(trial)
            item = TrialView(trial, plan.case_ref, plan.candidate_slot_id, plan.repetition,
                             disposition=dispositions.get(slot), evidence_ref=evidence.get(trial))
            intent = intents.get(slot)
            if intent is not None:
                state = await self._graph_state(intent, principal)
                if state is not None:
                    graph_ref = GraphSubjectRef(self._namespace, principal.tenant_id, state.graph_id)
                    revisions[f"graph:{state.graph_id}"] = state.event_seq
                    if candidates[plan.candidate_slot_id].task is not None:
                        node = state.node_states[0]
                        subject = (None if node.execution_id is None else ExecutionSubjectRef(
                            self._namespace, principal.tenant_id, node.execution_id))
                        status = None
                        if subject is not None:
                            execution = await self._execution.inspect(subject.execution_id, principal=principal)
                            status = execution.status
                            revisions[f"execution:{subject.execution_id}"] = execution.event_seq
                        item = replace(item, graph_ref=graph_ref, graph_status=state.status, subject=subject,
                                       execution_status=status or state.status, error_code=node.error_code)
                    else:
                        item = replace(item, graph_ref=graph_ref, graph_status=state.status,
                                       subject=graph_ref, execution_status=state.status)
            result.append(item)
        return tuple(result), revisions

    async def _scores(self, record: EvaluationRecord, trials: tuple[TrialView, ...],
                      revisions: dict[str, int] | None = None) -> tuple[ScoreAttemptView, ...]:
        saved = {(item.trial, item.scorer_slot_id): item for item in record.scores}
        intents = {item.slot_id: item for item in record.intents}
        dispositions = {item.slot_id: item.disposition for item in record.dispositions}
        results = []
        for trial in trials:
            for scorer in record.manifest.scorers:
                key = (trial.trial, scorer.slot_id)
                if key in saved:
                    results.append(saved[key])
                    continue
                slot = _score_slot(trial.trial, scorer.slot_id)
                intent = intents.get(slot)
                disposition = dispositions.get(slot)
                status = "not_attempted" if disposition is not None and disposition.terminal else "pending"
                node = None
                evidence_ref = trial.evidence_ref
                if intent is not None:
                    state = await self._graph_state(intent, record.manifest.principal)
                    if state is not None:
                        if revisions is not None:
                            revisions[f"graph:{state.graph_id}"] = state.event_seq
                        node = next(item for item in state.node_states if item.node_id == "score")
                    reference = intent.submission.graph.nodes[-1].input["evidence_ref"]
                    evidence_ref = EvidenceRef(reference["namespace"], reference["tenant_id"],
                                               reference["evidence_id"], reference["digest"])
                results.append(ScoreAttemptView(record.experiment_id,
                    None if node is None or node.execution_id is None else canonical_sha256({"experiment": record.experiment_id, "slot": slot}),
                    trial.trial, scorer.slot_id, scorer.task, status,
                    scorer_execution=None if node is None or node.execution_id is None else ExecutionSubjectRef(
                        self._namespace, record.manifest.principal.tenant_id, node.execution_id),
                    scorer_graph=None if intent is None else GraphSubjectRef(self._namespace, record.manifest.principal.tenant_id,
                        intent.submission.graph.graph_id), scorer_node_id=None if intent is None else "score",
                    evidence_ref=evidence_ref, reason=None if disposition is None else disposition.reason_code))
        return tuple(results)

    async def _inspect(self, experiment_id: str, principal: Principal) -> EvaluationView:
        record = await self._record(experiment_id, principal)
        trials, _ = await self._trials(record, principal)
        scores = await self._scores(record, trials)
        issues = [EvaluationIssue(item.disposition.reason_code, item.disposition.reason_code,
                                 retryable=item.disposition.retryable)
                  for item in record.dispositions if not item.disposition.terminal]
        for trial in trials if record.manifest.kind == "experiment" else ():
            if (trial.graph_status is TaskStatus.RECOVERY_REQUIRED
                    or trial.execution_status in {TaskStatus.RECOVERY_REQUIRED, ExecutionStatus.RECOVERY_REQUIRED}):
                issues.append(EvaluationIssue("recovery_required", "native graph requires recovery", trial.trial, retryable=True))
        for intent in record.intents:
            if intent.scorer_slot_id is not None and not intent.released:
                state = await self._graph_state(intent, principal)
                if state is not None and state.status is TaskStatus.RECOVERY_REQUIRED:
                    issues.append(EvaluationIssue("recovery_required", "scorer graph requires recovery",
                                                 intent.trial, intent.scorer_slot_id, True))
        terminal_trials = sum(item.terminal for item in trials)
        terminal_scores = sum(item.status in _RECORDED for item in scores)
        completion = _completion(record, trials, scores, blocked=bool(issues))
        return EvaluationView(experiment_id, record.manifest.kind, record.manifest.source_experiment_id,
            record.manifest.dataset, EvaluationProgress(
                0 if record.manifest.kind == "score_only" else len(trials),
                0 if record.manifest.kind == "score_only" else terminal_trials,
                len(scores), terminal_scores, sum(item.status == "valid" for item in scores), len(issues),
                len(trials) if record.manifest.kind == "score_only" else 0),
            completion, tuple(issues), record.created_at, record.updated_at)

    async def _graph_state(self, intent: EvaluationLaunchIntent, principal: Principal) -> TaskGraphState | None:
        if not intent.confirmed:
            return None
        try:
            return await self._graph.state(intent.submission.graph.graph_id, principal=principal)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_NOT_FOUND:
                return None
            raise

    async def _snapshot(
        self, experiment_id: str, principal: Principal,
    ) -> tuple[EvaluationRecord, EvaluationReport]:
        record, report = await self._build_snapshot(experiment_id, principal)
        await self._state.publish_report(report)
        return record, report

    async def _build_snapshot(
        self, experiment_id: str, principal: Principal,
    ) -> tuple[EvaluationRecord, EvaluationReport]:
        record = await self._record(experiment_id, principal)
        await self._require_content(record, principal)
        trials, revisions = await self._trials(record, principal)
        scores = await self._scores(record, trials, revisions)
        revisions[f"evaluation:{experiment_id}"] = record.revision
        blocked = False
        target_usage = {item.trial.trial_id: item.subject is None and item.graph_ref is None for item in trials}
        for item in record.evidence:
            target_usage[item.trial.trial_id] = (await self.read_evidence(item.evidence_ref, principal=principal)).usage_complete
        scorer_usage = {scorer.slot_id: {item.trial.trial_id: True for item in trials} for scorer in record.manifest.scorers}
        for intent in record.intents:
            if intent.scorer_slot_id is not None:
                state = await self._graph_state(intent, principal)
                if state is not None:
                    revisions[f"graph:{state.graph_id}"] = state.event_seq
                    blocked |= state.status is TaskStatus.RECOVERY_REQUIRED
                    summary = await self._history.graph_usage(state.graph_id, principal=principal)
                    _, complete = await self._model_usage(summary, principal)
                    scorer_usage[intent.scorer_slot_id][intent.trial.trial_id] = complete
        cutoff = EvaluationReadCutoff(experiment_id, record.manifest_digest, revisions,
            score_selections=tuple(ScoreSelection(scorer.slot_id, dimension.name, experiment_id)
                for scorer in record.manifest.scorers for dimension in scorer.dimensions),
            source_evidence_refs=tuple(item.evidence_ref for item in record.evidence),
            target_usage_complete=target_usage, scorer_usage_complete=scorer_usage)
        cases = tuple([await self._case(ref) for ref in
                       (await self.get_dataset(record.manifest.dataset, principal=principal)).ordered_case_refs])
        report = build_evaluation_report(record.manifest, cases, trials, scores, cutoff=cutoff,
                                         report_id=uuid.uuid4().hex, created_at=_now())
        report = replace(report, completion=_completion(record, trials, scores, blocked=blocked))
        return record, report

    async def get_report(self, report_id: str, *, principal: Principal) -> EvaluationReport | ComparisonReport:
        await self._authorize(principal, AuthorizationAction.EVALUATION_READ, report_id)
        report = await self._state.get_report(report_id)
        if report is None:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        ids = ((report.experiment_id,) if isinstance(report, EvaluationReport)
               else (report.baseline.experiment_id, report.candidate.experiment_id,
                     *(cutoff.experiment_id for cutoff in report.cutoff.scoring)))
        for experiment_id in ids:
            source = await self._record(experiment_id, principal)
            await self._require_content(source, principal)
        return report

    async def create_comparison_report(self, spec: ComparisonSpec, *, principal: Principal) -> ComparisonReport:
        await self._record(spec.baseline.experiment_id, principal, AuthorizationAction.EVALUATION_COMPARE)
        await self._record(spec.candidate.experiment_id, principal, AuthorizationAction.EVALUATION_COMPARE)
        ids = tuple(dict.fromkeys((spec.baseline.experiment_id, spec.candidate.experiment_id, *(
            item.scoring_experiment_id for selection in spec.scores for item in (selection.baseline, selection.candidate)
            if item.scoring_experiment_id is not None))))
        if spec.cutoff is None:
            snapshots = {identity: await self._snapshot(identity, principal) for identity in ids}
        else:
            cutoffs = {item.experiment_id: item for item in
                       (spec.cutoff.baseline, spec.cutoff.candidate, *spec.cutoff.scoring)}
            if (spec.cutoff.baseline.experiment_id == spec.cutoff.candidate.experiment_id and
                    spec.cutoff.baseline != spec.cutoff.candidate) or set(cutoffs) != set(ids):
                raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "cutoff does not identify selected experiments")
            snapshots = {}
            for identity in ids:
                record = await self._record(identity, principal)
                snapshot = await self._state.get_report_at(cutoffs[identity])
                if snapshot is None:
                    raise AIError(ErrorCode.STORAGE_NOT_FOUND, "selected report snapshot is unavailable")
                snapshots[identity] = record, snapshot
        for record, _report in snapshots.values():
            await self._require_content(record, principal)
        left, left_report = snapshots[spec.baseline.experiment_id]
        right, right_report = snapshots[spec.candidate.experiment_id]
        scoring = tuple(report.cutoff for identity, (_, report) in snapshots.items()
                        if identity not in {left.experiment_id, right.experiment_id})
        cutoff = ComparisonReadCutoff(left_report.cutoff, right_report.cutoff, scoring)
        if spec.cutoff is not None and spec.cutoff != cutoff:
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "requested cutoff is no longer current")
        left_dataset = await self.get_dataset(left.manifest.dataset, principal=principal)
        right_dataset = await self.get_dataset(right.manifest.dataset, principal=principal)
        report = build_comparison_report(spec, left.manifest, right.manifest,
            tuple([await self._case(ref) for ref in left_dataset.ordered_case_refs]),
            tuple([await self._case(ref) for ref in right_dataset.ordered_case_refs]),
            tuple(item for _, snapshot in snapshots.values() for item in snapshot.trials),
            tuple(item for _, snapshot in snapshots.values() for item in snapshot.score_attempts),
            baseline_dataset=left_dataset, candidate_dataset=right_dataset, cutoff=cutoff,
            report_id=uuid.uuid4().hex, created_at=_now(),
            scoring_manifests=tuple(record.manifest for record, _ in snapshots.values()),
            completions={identity: snapshot.completion for identity, (_, snapshot) in snapshots.items()},
            usage_complete=self._comparison_usage_complete(spec, snapshots))
        await self._state.publish_report(report)
        return report

    def _comparison_usage_complete(
        self, spec: ComparisonSpec, snapshots: Mapping[str, tuple[EvaluationRecord, EvaluationReport]],
    ) -> bool:
        for side, selections in ((spec.baseline, (item.baseline for item in spec.scores)),
                                 (spec.candidate, (item.candidate for item in spec.scores))):
            target = snapshots[side.experiment_id][1]
            trials = {item.trial for item in target.trials if item.candidate_slot_id == side.slot_id}
            if any(not target.cutoff.target_usage_complete.get(item.trial_id, False) for item in trials):
                return False
            for selection in selections:
                scoring = snapshots[selection.scoring_experiment_id or side.experiment_id][1]
                planned = {item.trial for item in scoring.trials}
                observations = scoring.cutoff.scorer_usage_complete.get(selection.scorer_slot_id, {})
                if any(not observations.get(item.trial_id, False) for item in trials & planned):
                    return False
        return True

    def _cursor(self, kind: str, digest: str, position: str, principal: Principal) -> str:
        return self._cursor_signer.encode(CursorPayload(1, principal.tenant_id, kind, digest,
                                                       position, 0, int(time.time()) + 86400))

    def _cursor_position(self, token: str, kind: str, digest: str, principal: Principal) -> str:
        value = self._cursor_signer.decode(token)
        if value.tenant_id != principal.tenant_id or value.resource_kind != kind or value.filter_digest != digest:
            raise AIError(ErrorCode.CURSOR_INVALID)
        return value.position

    async def _page(
        self, experiment_id: str, principal: Principal, filters: TrialFilter | ScoreFilter,
        cursor: str | None, limit: int,
    ) -> Page[TrialView] | Page[ScoreAttemptView]:
        validate_page_limit(limit)
        is_trial = isinstance(filters, TrialFilter)
        selected = ({"candidates": list(filters.candidate_slot_ids),
                     "cases": [ref.to_mapping() for ref in filters.case_refs], "terminal": filters.terminal}
                    if is_trial else {"scorers": list(filters.scorer_slot_ids),
                                    "trials": list(filters.trial_ids), "statuses": list(filters.statuses)})
        digest = canonical_sha256({"experiment": experiment_id, "filters": selected,
                                  "principal": principal_identity_payload(principal)})
        kind = "evaluation.trials" if is_trial else "evaluation.scores"
        if cursor:
            report_id, raw_index = self._cursor_position(cursor, kind, digest, principal).rsplit(":", 1)
            report = await self.get_report(report_id, principal=principal)
            if not isinstance(report, EvaluationReport) or report.experiment_id != experiment_id:
                raise AIError(ErrorCode.CURSOR_INVALID)
            index = int(raw_index)
        else:
            _, report = await self._build_snapshot(experiment_id, principal)
            index = 0
        if is_trial:
            values = tuple(item for item in report.trials if
                (not filters.candidate_slot_ids or item.candidate_slot_id in filters.candidate_slot_ids) and
                (not filters.case_refs or item.case_ref in filters.case_refs) and
                (filters.terminal is None or item.terminal == filters.terminal))
        else:
            values = tuple(item for item in report.score_attempts if
                (not filters.scorer_slot_ids or item.scorer_slot_id in filters.scorer_slot_ids) and
                (not filters.trial_ids or item.trial.trial_id in filters.trial_ids) and
                (not filters.statuses or item.status in filters.statuses))
        next_cursor = (self._cursor(kind, digest, f"{report.report_id}:{index + limit}", principal)
                       if index + limit < len(values) else None)
        if not cursor and next_cursor is not None:
            await self._state.publish_report(report)
        return Page(values[index:index + limit], next_cursor)

    async def _disposition(
        self, experiment_id: str, slot: str, reason: str, *, retryable: bool = False,
        cancelled: bool = False,
    ) -> None:
        item = EvaluationSlotDisposition(slot, SlotDispositionView(
            "recoverable_blocked" if retryable else "cancelled" if cancelled else "permanent_unavailable",
            reason, not retryable, retryable, _now()))
        await self._state.append_slot_disposition(experiment_id, item)

    async def _tick(
        self, experiment_id: str, engine: "TaskEngine | None", principal: Principal,
        *, trial_scope: EvaluationTrialScopeCallback | None = None,
    ) -> None:
        record = await self._record(experiment_id, principal, allow_expired=True)
        expired = False
        try:
            await self._require_content(record, principal)
        except AIError as error:
            if error.code is not ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE:
                raise
            expired = True
            record = await self._state.close_reservation_gate(experiment_id)
        for intent in record.intents:
            if not intent.released and (record.gate == "closed_cancel" or intent.deadline_at is not None and intent.deadline_at <= _now()):
                await self._cancel_intent(record, intent, principal)
            elif not intent.confirmed and not intent.released:
                async with self._trial_engine(record, intent, engine, trial_scope) as selected:
                    result = await selected.start_prepared(intent.submission)
                await self._state.settle_intent(experiment_id, intent.slot_id,
                                               confirmed=result.admitted, released=not result.admitted)
            state = await self._graph_state(intent, principal)
            if state is None:
                continue
            if not expired:
                if intent.scorer_slot_id is not None:
                    await self._resume_decision(record, intent, state)
                    await self._collect_score(record, intent, state)
                elif state.status in _TERMINAL:
                    await self._capture_evidence(record, intent, state, principal)
            if state.status in _TERMINAL:
                await self._close_trial_scope((experiment_id, intent.slot_id))
                await self._state.settle_intent(experiment_id, intent.slot_id, confirmed=True, released=True)
        record = await self._record(experiment_id, principal, allow_expired=True)
        if record.gate == "open" and await self._budget_exhausted(record, principal):
            record = await self._state.close_reservation_gate(experiment_id, budget=True)
        if record.gate != "open":
            for intent in record.intents:
                if record.gate == "closed_cancel" and not intent.released:
                    await self._cancel_intent(record, intent, principal)
            record = await self._record(experiment_id, principal, allow_expired=True)
            intents = {item.slot_id: item for item in record.intents}
            scored = {(item.trial, item.scorer_slot_id) for item in record.scores}
            trials = (record.manifest.source_trials if record.manifest.kind == "score_only" else
                      tuple(TargetTrialRef(experiment_id, item.trial_id) for item in record.manifest.trials))
            for trial in trials:
                if record.manifest.kind == "experiment" and _target_slot(trial) not in intents:
                    await self._disposition(experiment_id, _target_slot(trial), record.gate, cancelled=True)
                for scorer in record.manifest.scorers:
                    slot = _score_slot(trial, scorer.slot_id)
                    intent = intents.get(slot)
                    if intent is None or intent.released and (trial, scorer.slot_id) not in scored:
                        await self._disposition(experiment_id, slot,
                            "evidence_expired" if expired else record.gate, cancelled=True)
            return
        assert engine is not None
        trials, _ = await self._trials(record, principal)
        dispositions = {item.slot_id: item.disposition for item in record.dispositions}
        intents = {item.slot_id: item for item in record.intents}
        cases = {trial.case_ref: await self._case(trial.case_ref) for trial in trials}
        candidates = {candidate.slot_id: candidate for candidate in record.manifest.candidates}
        for trial in trials:
            slot = _target_slot(trial.trial)
            if record.manifest.kind == "experiment" and slot not in intents and slot not in dispositions:
                try:
                    graph = await self._compiler.graph(candidates[trial.candidate_slot_id], cases[trial.case_ref],
                        graph_id=canonical_sha256({"experiment": experiment_id, "slot": slot}),
                        principal=record.manifest.principal, input_mode=record.manifest.input_mode, owner_id=record.experiment_id,
                        owned_captures=list(record.owned_input_captures))
                    candidate = candidates[trial.candidate_slot_id]
                    limits = (record.manifest.policy.target_graph_limits if candidate.graph_template is None
                              else candidate.graph_template.limits)
                    await self._launch(record, trial.trial, None, graph, limits, engine, trial_scope=trial_scope)
                except AIError as error:
                    await self._launch_error(experiment_id, slot, error)
            for scorer in record.manifest.scorers:
                score_slot = _score_slot(trial.trial, scorer.slot_id)
                if score_slot in intents or score_slot in dispositions:
                    continue
                if trial.evidence_ref is None:
                    if trial.disposition is not None and trial.disposition.terminal:
                        await self._disposition(experiment_id, score_slot, "evidence_unavailable")
                    continue
                try:
                    source = await self.read_evidence(trial.evidence_ref, principal=principal)
                    target_kind = "execution" if isinstance(source.target, ExecutionTargetEvidence) else "graph"
                    if target_kind not in scorer.accepts_target_kinds:
                        await self._disposition(experiment_id, score_slot, "target_kind_not_supported")
                        continue
                    if source.target.status != "succeeded" and not scorer.accepts_target_failure:
                        await self._disposition(experiment_id, score_slot, "target_failed")
                        continue
                    evidence = await self._project_evidence(source, scorer)
                    sample = ScoringInput(trial.trial, evidence.target,
                        await self._value(cases[trial.case_ref].expected), cases[trial.case_ref].expected is not None,
                        await self._value(scorer.rubric), scorer.config, evidence.ref,
                        canonical_sha256({"candidate": trial.candidate_slot_id})[:16],
                        rubric_present=scorer.rubric is not None,
                        target_input=None if evidence.input is None else evidence.input.value)
                    graph = self._scoring_graph(record, trial.trial, scorer, sample)
                    await self._launch(record, trial.trial, scorer, graph,
                                       record.manifest.policy.scorer_graph_limits, engine, trial_scope=trial_scope)
                except AIError as error:
                    await self._launch_error(experiment_id, score_slot, error)

    async def _launch_error(self, experiment_id: str, slot: str, error: AIError) -> None:
        retryable = error.code not in {ErrorCode.INPUT_CAPTURE_UNAVAILABLE, ErrorCode.EVALUATION_INCOMPATIBLE,
            ErrorCode.BINDING_CONFLICT, ErrorCode.BINDING_NOT_REGISTERED, ErrorCode.REQUEST_FIELD_INVALID}
        await self._disposition(experiment_id, slot, str(error.code), retryable=retryable)

    async def _launch(
        self, record: EvaluationRecord, trial: TargetTrialRef, scorer: ScorerContract | None,
        graph: TaskGraph, limits: "TaskGraphLimits", engine: "TaskEngine",
        *, trial_scope: EvaluationTrialScopeCallback | None = None,
    ) -> None:
        require_evaluation_content(record, now=_now())
        policy = record.manifest.policy
        slot = _target_slot(trial) if scorer is None else _score_slot(trial, scorer.slot_id)
        capacity = policy.target_concurrency if scorer is None else policy.scorer_concurrency
        current = await self._record(record.experiment_id, record.manifest.principal)
        if current.gate != "open" or sum(not item.released and (item.scorer_slot_id is None) == (scorer is None)
                                          for item in current.intents) >= capacity:
            return
        planning = engine if scorer is None else engine.with_definitions(self._recorder)
        submission = await planning.describe_submission(graph, principal=record.manifest.principal,
            idempotency_key=f"evaluation:{record.experiment_id}:{slot}", limits=limits,
            correlation={"evaluation_experiment": record.experiment_id, "evaluation_trial": trial.trial_id,
                         "evaluation_slot": slot})
        timeout = (policy.trial_timeout_seconds if scorer is None else
                   policy.human_timeout_seconds if scorer.task == TaskRef.deferred_input() else policy.scorer_timeout_seconds)
        intent = EvaluationLaunchIntent(slot, trial, None if scorer is None else scorer.slot_id,
                                        submission, None if timeout is None else _now() + timedelta(seconds=timeout))
        registered = await self._state.register_launch_intent(record.experiment_id, intent, capacity=capacity)
        selected = next((item for item in registered.intents if item.slot_id == slot), None)
        if selected is None:
            return
        if registered.gate == "closed_cancel":
            await self._cancel_intent(registered, selected, record.manifest.principal)
            return
        async with self._trial_engine(registered, selected, engine, trial_scope) as scoped_engine:
            result = await scoped_engine.start_prepared(selected.submission)
        if not result.admitted:
            await self._close_trial_scope((record.experiment_id, slot))
        await self._state.settle_intent(record.experiment_id, slot,
                                       confirmed=result.admitted, released=not result.admitted)
        if not result.admitted:
            await self._disposition(record.experiment_id, slot, "submission_cancelled", cancelled=True)

    async def _cancel_intent(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, principal: Principal,
    ) -> None:
        result = await self._cancel_submission(record, intent, principal,
            f"evaluation-cancel:{record.experiment_id}:{intent.slot_id}")
        if result.status in _TERMINAL:
            await self._close_trial_scope((record.experiment_id, intent.slot_id))
        await self._state.settle_intent(record.experiment_id, intent.slot_id,
                                       confirmed=result.admitted, released=result.status in _TERMINAL)
        if not result.admitted:
            await self._disposition(record.experiment_id, intent.slot_id, "submission_cancelled", cancelled=True)

    async def _cancel_submission(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, principal: Principal,
        idempotency_key: str,
    ) -> TaskSubmissionCancellation:
        scope = self._trial_scopes.get((record.experiment_id, intent.slot_id))
        if scope is not None:
            with scope.borrow_engine() as engine:
                if engine is not None:
                    return await engine.cancel_submission(intent.submission.ref, principal=principal,
                                                          idempotency_key=idempotency_key)
            if scope.closing:
                await scope.close()
            else:
                scope.request_close()
        return await self._graph.cancel_submission(intent.submission.ref, principal=principal,
                                                  idempotency_key=idempotency_key)

    async def _capture_evidence(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, state: TaskGraphState,
        principal: Principal,
    ) -> None:
        if any(item.trial == intent.trial for item in record.evidence):
            return
        plan = next(item for item in record.manifest.trials if item.trial_id == intent.trial.trial_id)
        candidate = next(item for item in record.manifest.candidates if item.slot_id == plan.candidate_slot_id)
        nodes = {item.node_id: item for item in state.node_states}
        if candidate.task is not None:
            node = nodes["target"]
            if node.execution_id is None:
                await self._disposition(record.experiment_id, intent.slot_id, node.error_code or "execution_unavailable")
                return
            value = (InlineValue.from_value(await self._history.task_result(state.graph_id, "target", principal=principal))
                     if node.status is TaskStatus.SUCCEEDED else None)
            target = ExecutionTargetEvidence(ExecutionSubjectRef(self._namespace, principal.tenant_id, node.execution_id),
                                              node.status.value.lower(), value, node.error_code)
        else:
            selected = candidate.graph_template.outputs
            if candidate.graph_template.selector == "terminal_sinks":
                dependencies = {name for node in state.nodes for name in node.dependencies}
                selected = {name: name for name in nodes if name not in dependencies}
            outputs = {}
            for alias, node_id in selected.items():
                node = nodes[node_id]
                value = (InlineValue.from_value(await self._history.task_result(state.graph_id, node_id, principal=principal))
                         if node.status is TaskStatus.SUCCEEDED else None)
                outputs[alias] = GraphOutputItem(node.status.value.lower(), value,
                    None if node.status is TaskStatus.SUCCEEDED else node.error_code or node.status.value.lower())
            target = GraphTargetEvidence(GraphSubjectRef(self._namespace, principal.tenant_id, state.graph_id),
                state.status.value.lower(), outputs, {name: node.status.value.lower() for name, node in nodes.items()})
        usage = await self._history.graph_usage(state.graph_id, principal=principal)
        model_usage, usage_complete = await self._model_usage(usage, principal)
        trace = {}
        include_trace = any(item.evidence_policy.include_trace for item in record.manifest.scorers)
        include_input = any(item.evidence_policy.include_input for item in record.manifest.scorers)
        include_output = any(item.evidence_policy.include_output for item in record.manifest.scorers)
        execution_ids = tuple(dict.fromkeys((*(node.execution_id for node in nodes.values() if node.execution_id),
                                             *(item.execution_id for item in usage.cutoffs))))
        if include_trace:
            for execution_id in execution_ids:
                cursor = None
                while True:
                    page = await self._history.trace(execution_id, principal=principal, cursor=cursor,
                                                     limit=100)
                    for item in page.items:
                        trace[(item.execution_id, item.step_event_seq)] = {"execution_id": item.execution_id,
                            "step_event_seq": item.step_event_seq, "payload": item.payload}
                    cursor = page.next_cursor
                    if cursor is None:
                        break
        include_attachments = any(item.evidence_policy.include_attachments for item in record.manifest.scorers)
        attachments, attachment_issues, attachment_sources = await self._capture_attachments(
            intent.trial, execution_ids, principal, selected=include_attachments)
        captured_input, input_issues = (await self._capture_target_input(state, principal,
            graph_target=candidate.graph_template is not None) if include_input else (None, ()))
        if not include_output:
            target = (replace(target, output=None) if isinstance(target, ExecutionTargetEvidence)
                      else replace(target, outputs={}))
        bundle = EvidenceBundle(EvidenceRef(self._namespace, principal.tenant_id, uuid.uuid4().hex, "0" * 64),
            intent.trial, target, captured_input, tuple(trace.values()), attachments,
            UsageMetrics(usage.logical_requests, 0, usage.input_tokens, usage.output_tokens,
                         usage.cache_read_tokens, usage.cache_write_tokens),
            usage_complete,
            {"graph_event_seq": state.event_seq, "include_trace": include_trace,
             "include_input": include_input and not input_issues, "input_issues": list(input_issues),
             "include_output": include_output,
             "include_attachments": include_attachments and not attachment_issues,
             "attachment_issues": list(attachment_issues), "attachment_sources": attachment_sources, "usage": [{"execution_id": item.execution_id,
                "agent_run_seq": item.agent_run_seq, "model_request_seq": item.model_request_seq}
                for item in usage.cutoffs]}, model_usage=model_usage)
        bundle = replace(bundle, ref=replace(bundle.ref, digest=bundle.digest))
        await self._state.publish_evidence(bundle)
        await self._state.publish_trial_evidence(record.experiment_id, EvaluationTrialEvidence(intent.trial, bundle.ref))

    async def _capture_target_input(
        self, state: TaskGraphState, principal: Principal, *, graph_target: bool,
    ) -> tuple[InlineValue | None, tuple[str, ...]]:
        inputs: dict[str, JsonValue] = {}
        declarations = {node.node_id: node for node in state.nodes}
        states = {node.node_id: node for node in state.node_states}
        for node in state.node_states:
            if node.execution_id is None:
                inputs[node.node_id] = {"status": node.status.value.lower(), "input": None}
                continue
            try:
                execution = await self._execution.inspect(node.execution_id, principal=principal)
                if execution.binding_kind == "agent":
                    prompt = await self._captures.read_execution_prompt(node.execution_id, principal=principal)
                    value = {"kind": "agent_input", "prompt": input_intent(prompt, ()).prompt}
                else:
                    parameters = await self._captures.read_execution_input(node.execution_id, principal=principal)
                    dependencies = {}
                    declaration = declarations[node.node_id]
                    capture = declaration.input_capture
                    if capture is not None:
                        task = await self._captures.read_task(capture, principal=principal)
                        for item in task.dependencies:
                            dependencies[item.name] = {"status": item.state.status.value.lower(),
                                "value": (await self._captures.read_dependency(capture, item.name, principal=principal)
                                          if item.state.status is TaskStatus.SUCCEEDED else None),
                                "reason": item.state.error_code}
                    for name, reference in declaration.input_refs.items():
                        if isinstance(reference, TaskNodeResultRef):
                            dependency = states[reference.node_id]
                            status, reason, digest = dependency.status, dependency.error_code, dependency.result_digest
                            body = (await self._history.task_result(state.graph_id, reference.node_id, principal=principal)
                                    if status is TaskStatus.SUCCEEDED else None)
                        else:
                            if reference.namespace != self._namespace or reference.tenant_id != principal.tenant_id:
                                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                            status, reason, digest = TaskStatus.SUCCEEDED, None, reference.result_digest
                            body = await self._history.task_result(reference.graph_id, reference.node_id, principal=principal)
                        if status is TaskStatus.SUCCEEDED and canonical_sha256(body) != digest:
                            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE)
                        dependencies[name] = {"status": status.value.lower(), "value": body, "reason": reason}
                    value = {"kind": "task_input", "input": dict(parameters), "dependencies": dependencies}
                inputs[node.node_id] = value
            except AIError as error:
                if error.code not in {ErrorCode.INPUT_CAPTURE_UNAVAILABLE, ErrorCode.INPUT_CONTEXT_UNAVAILABLE}:
                    raise
                return None, (f"input_unavailable:{node.node_id}",)
        return InlineValue.from_value({"kind": "graph_input", "inputs": inputs} if graph_target else inputs["target"]), ()

    async def _capture_attachments(
        self, trial: TargetTrialRef, execution_ids: tuple[str, ...], principal: Principal, *, selected: bool,
    ) -> tuple[tuple[EvidenceAttachmentRef, ...], tuple[str, ...], dict[str, JsonValue]]:
        if not selected:
            return (), (), {}
        attachments = {}
        issues = []
        sources = {}
        for execution_id in execution_ids:
            facts = []
            cursor = None
            while True:
                page = await self._history.attachment_facts(execution_id, principal=principal, cursor=cursor)
                facts.extend(page.items)
                cursor = page.next_cursor
                if cursor is None:
                    break
            if not facts:
                continue
            try:
                capture = await self._captures.capture_input(execution_id, CaptureInputRequest(principal,
                    f"evaluation-evidence:{trial.target_experiment_id}:{trial.trial_id}:{execution_id}", "clean"))
                sources[execution_id] = capture_mapping(capture)
                content = ((await self._captures.read_agent(capture, principal=principal)).prompt
                           if isinstance(capture, AgentInputCaptureRef) else ())
            except AIError as error:
                if error.code not in {ErrorCode.INPUT_CAPTURE_UNAVAILABLE, ErrorCode.INPUT_CONTEXT_UNAVAILABLE}:
                    raise
                content = ()
            retained = set()
            if not isinstance(content, str):
                for position, item in enumerate(content):
                    if not isinstance(item, BinaryContent):
                        continue
                    reference = await put_runtime_object(self._objects, self._object_keys,
                        RuntimeDomain.EVALUATION, principal.tenant_id, item.data)
                    identity = canonical_sha256({"execution": execution_id, "position": position,
                                                 "digest": reference.digest, "media_type": item.media_type})
                    attachments[identity] = EvidenceAttachmentRef(identity, item.media_type, reference)
                    retained.add(reference.digest)
            issues.extend(f"attachment_bytes_unavailable:{fact.attachment_id}" for fact in facts
                          if fact.digest not in retained)
        return tuple(attachments.values()), tuple(dict.fromkeys(issues)), sources

    async def _project_evidence(self, source: EvidenceBundle, scorer: ScorerContract) -> EvidenceBundle:
        policy = scorer.evidence_policy
        if any(required and not source.cutoff.get(name) for name, required in policy.to_mapping().items()):
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, "required evidence was not captured")
        target = source.target
        if not policy.include_output:
            if isinstance(target, ExecutionTargetEvidence):
                target = replace(target, output=None)
            else:
                target = replace(target, outputs={})
        identity = canonical_sha256({"source": source.ref.to_mapping(), "policy": policy.to_mapping()})
        bundle = replace(source, ref=EvidenceRef(self._namespace, source.ref.tenant_id, identity, "0" * 64),
            target=target, input=source.input if policy.include_input else None,
            trace=source.trace if policy.include_trace else (),
            attachments=source.attachments if policy.include_attachments else (), source_ref=source.ref)
        bundle = replace(bundle, ref=replace(bundle.ref, digest=bundle.digest))
        return await self._state.publish_evidence(bundle)

    def _scoring_graph(
        self, record: EvaluationRecord, trial: TargetTrialRef, scorer: ScorerContract, sample: ScoringInput,
    ) -> TaskGraph:
        slot = _score_slot(trial, scorer.slot_id)
        data = sample.to_mapping()
        if scorer.input_projection["kind"] == "agent_projected":
            data = dict(AgentTaskInput("Evaluate the supplied scoring input.", parameters=data))
        elif scorer.input_projection["kind"] == "agent_literal":
            data = dict(AgentTaskInput(scorer.input_projection["instructions"] + "\nDATA\n" +
                                       canonical_json_bytes(data).decode("utf-8")))
        return TaskGraph(canonical_sha256({"experiment": record.experiment_id, "slot": slot}), (
            TaskNode("score", task=scorer.task, input=data, output_type=ScoreBundle, failure_policy="isolate"),
            TaskNode("record", task=self._recorder.ref, dependencies=("score",),
                input={"experiment_id": record.experiment_id, "slot_id": slot, "evidence_ref": sample.evidence_ref.to_mapping()},
                dependency_policy="all_terminal", failure_policy="isolate", max_attempts=3),
        ))

    async def _record_score(self, context: TaskNodeContext[AppT]) -> JsonValue:
        identity, slot = context.input["experiment_id"], context.input["slot_id"]
        record = await self._record(identity, context.principal, allow_expired=True)
        intent = next(item for item in record.intents if item.slot_id == slot)
        if intent.submission.graph.graph_id != context.graph_id or intent.scorer_slot_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = await self._graph.state(context.graph_id, principal=context.principal)
        await self._collect_score(record, intent, state)
        return {"recorded": True}

    async def _collect_score(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, state: TaskGraphState,
    ) -> None:
        node = next(item for item in state.node_states if item.node_id == "score")
        if node.status not in _TERMINAL:
            return
        record = await self._record(record.experiment_id, record.manifest.principal, allow_expired=True)
        if any(item.trial == intent.trial and item.scorer_slot_id == intent.scorer_slot_id for item in record.scores):
            return
        try:
            await self._require_content(record, record.manifest.principal)
        except AIError as error:
            if error.code is ErrorCode.EVALUATION_EVIDENCE_UNAVAILABLE:
                return
            raise
        scorer = next(item for item in record.manifest.scorers if item.slot_id == intent.scorer_slot_id)
        source = next((item for item in record.evidence if item.trial == intent.trial), None)
        if source is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        evidence = await self._project_evidence(await self.read_evidence(source.evidence_ref,
            principal=record.manifest.principal), scorer)
        score = None
        status, reason = "error", node.error_code or node.status.value.lower()
        if node.status is TaskStatus.CANCELLED and record.gate != "closed_cancel" and intent.deadline_at is not None and intent.deadline_at <= _now():
            reason = "unanswered" if scorer.task == TaskRef.deferred_input() else "scorer_timeout"
        if node.status is TaskStatus.SUCCEEDED:
            value = await self._history.task_result(state.graph_id, "score", principal=record.manifest.principal)
            try:
                score = ScoreBundle.from_mapping(value)
                score.validate_contract(scorer, evidence_ids=self._evidence_ids(evidence))
                status = "not_applicable" if all(isinstance(item, ScoreNotApplicable) for item in score.dimensions.values()) else "valid"
                reason = None
            except (TypeError, ValueError):
                score, status, reason = None, "error", "invalid_score"
        decision = next((item for item in record.human_decisions if item.slot_id == intent.slot_id), None)
        if node.execution_id is None:
            status = "not_attempted"
        result = ScoreAttemptView(record.experiment_id,
            None if node.execution_id is None else canonical_sha256({"experiment": record.experiment_id, "slot": intent.slot_id}),
            intent.trial, scorer.slot_id, scorer.task, status, score,
            None if node.execution_id is None else ExecutionSubjectRef(self._namespace, record.manifest.principal.tenant_id, node.execution_id),
            GraphSubjectRef(self._namespace, record.manifest.principal.tenant_id, state.graph_id),
            "score", evidence.ref, None if decision is None else decision.decision_id, reason)
        def append(value: EvaluationRecord) -> EvaluationRecord:
            existing = next((item for item in value.scores if item.trial == result.trial and item.scorer_slot_id == result.scorer_slot_id), None)
            if existing is not None:
                if existing != result:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return value
            return replace(value, scores=(*value.scores, result))
        await self._state.update(record.experiment_id, append)

    async def _model_usage(
        self, summary: UsageSummary, principal: Principal,
    ) -> tuple[tuple[ModelUsage, ...], bool]:
        observations = []
        for execution_id in dict.fromkeys(item.execution_id for item in summary.cutoffs):
            cursor = None
            while True:
                page = await self._history.model_interactions(execution_id, principal=principal,
                    cursor=cursor, include_content=True, limit=100,
                    cutoffs=tuple(item for item in summary.cutoffs if item.execution_id == execution_id))
                for item in page.items:
                    response = item.response if isinstance(item.response, Mapping) else {}
                    provider, model = response.get("provider_name"), response.get("model_name")
                    observations.append(ModelUsage(provider if isinstance(provider, str) else "",
                        model if isinstance(model, str) else "", item.usage or UsageMetrics(), item.usage is not None))
                cursor = page.next_cursor
                if cursor is None:
                    break
        complete = summary.unknown_usage_requests == 0 and summary.unrecorded_executions == 0
        if len(observations) < summary.logical_requests:
            totals = [summary.input_tokens, summary.output_tokens, summary.cache_read_tokens, summary.cache_write_tokens]
            for item in observations:
                for index, count in enumerate((item.usage.input_tokens, item.usage.output_tokens,
                                                item.usage.cache_read_tokens, item.usage.cache_write_tokens)):
                    totals[index] -= count
            if min(totals) < 0:
                complete = False
            else:
                observations.append(ModelUsage("", "", UsageMetrics(summary.logical_requests - len(observations),
                    0, *totals), complete))
        return tuple(observations), complete and all(item.complete for item in observations)

    async def _budget_exhausted(self, record: EvaluationRecord, principal: Principal) -> bool:
        policy = record.manifest.policy
        if policy.token_limit is None and policy.cost_limit is None:
            return False
        observations = []
        complete = True
        for intent in record.intents:
            if not intent.confirmed:
                continue
            summary = await self._history.graph_usage(intent.submission.graph.graph_id, principal=principal)
            usage, known = await self._model_usage(summary, principal)
            observations.extend(usage)
            complete &= known
        prices = None if policy.price_table is None else PriceTable.from_mapping(
            await self._value(policy.price_table), currency=policy.currency)
        return estimate_model_budget(policy, observations, usage_complete=complete, price_table=prices).stop

    async def _cancel(self, experiment_id: str, principal: Principal, key: str) -> EvaluationView:
        self._ensure_open()
        validate_idempotency_key(key)
        await self._record(experiment_id, principal, AuthorizationAction.EVALUATION_CANCEL)
        view = await self._inspect(experiment_id, principal)
        if view.completion in {"complete", "cancelled"}:
            return view
        record = await self._state.close_reservation_gate(experiment_id)
        for intent in record.intents:
            if not intent.released:
                await self._cancel_intent(record, intent, principal)
        self._watch(experiment_id, None, principal)
        await self._tick(experiment_id, None, principal)
        return await self._inspect(experiment_id, principal)

    async def _rescore(
        self, experiment_id: str, principal: Principal, request: RescoreRequest, engine: "TaskEngine",
        *, trial_scope: EvaluationTrialScopeCallback | None = None,
    ) -> "EvaluationRun":
        self._ensure_open()
        source = await self._record(experiment_id, principal, AuthorizationAction.EVALUATION_RESCORE)
        if source.manifest.kind != "experiment":
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "rescore requires a target experiment")
        await self._require_content(source, principal)
        bound = self._engine(engine)
        self._validate_storage(source.manifest.policy)
        trials, _ = await self._trials(source, principal)
        selected = tuple(item for item in trials if request.trial_ids is None or item.trial.trial_id in request.trial_ids)
        if request.trial_ids is not None and {item.trial.trial_id for item in selected} != set(request.trial_ids):
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)
        if not selected or any(item.evidence_ref is None for item in selected):
            raise AIError(ErrorCode.INPUT_CAPTURE_UNAVAILABLE, "rescore requires fixed terminal evidence")
        scorers = tuple(self._compiler.scorer(spec, engine=bound, policy=source.manifest.policy) for spec in request.scorers)
        for trial in selected:
            evidence = await self.read_evidence(trial.evidence_ref, principal=principal)
            for scorer in scorers:
                await self._project_evidence(evidence, scorer)
        manifest = EvaluationManifest(uuid.uuid4().hex, "score_only", experiment_id,
            source.manifest.dataset, source.manifest.candidates, scorers, (), tuple(item.trial for item in selected),
            source.manifest.policy, source.manifest.input_mode, principal,
            trial_scope_required=trial_scope is not None)
        data = manifest.to_mapping()
        data.pop("experiment_id")
        evidence = tuple(EvaluationTrialEvidence(item.trial, item.evidence_ref) for item in selected)
        data["evidence"] = [item.evidence_ref.to_mapping() for item in evidence]
        now = _now()
        source_deadlines = tuple(value for value in (source.content_expires_at, source.metadata_expires_at)
                                 if value is not None)
        record = await self._state.reserve_experiment(EvaluationRecord(manifest, manifest.digest,
            canonical_sha256(data), idempotency_key_digest(request.idempotency_key), "open", 0, now, now, evidence=evidence,
            content_expires_at=min(source_deadlines) if source_deadlines else None,
            metadata_expires_at=None if source.manifest.policy.metadata_retention_seconds is None else
            now + timedelta(seconds=source.manifest.policy.metadata_retention_seconds)))
        self._watch(record.experiment_id, bound, principal, trial_scope=trial_scope)
        return EvaluationRun(self, record.experiment_id, principal)

    async def _human_score(
        self, experiment_id: str, principal: Principal, request: HumanScoreRequest,
    ) -> ScoreAttemptView:
        self._ensure_open()
        record = await self._record(experiment_id, principal, AuthorizationAction.EVALUATION_HUMAN_SCORE)
        await self._require_content(record, principal)
        intent = next((item for item in record.intents if item.trial.trial_id == request.trial_id and
                       item.scorer_slot_id == request.scorer_slot_id), None)
        if intent is None:
            raise AIError(ErrorCode.TASK_NOT_READY)
        scorer = next(item for item in record.manifest.scorers if item.slot_id == request.scorer_slot_id)
        if scorer.task != TaskRef.deferred_input():
            raise AIError(ErrorCode.EVALUATION_INCOMPATIBLE, "scorer is not a deferred human input")
        source = next(item for item in record.evidence if item.trial == intent.trial)
        evidence = await self._project_evidence(await self.read_evidence(source.evidence_ref, principal=principal), scorer)
        if request.evidence_ref != evidence.ref:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        request.score.validate_contract(scorer, evidence_ids=self._evidence_ids(evidence))
        digest = canonical_sha256({"actor": principal_identity_payload(principal), "slot": intent.slot_id,
            "evidence": evidence.ref.to_mapping(), "score": request.score.to_mapping()})
        decision = EvaluationHumanDecision(intent.slot_id, uuid.uuid4().hex, principal, _now(), digest,
                                           idempotency_key_digest(request.idempotency_key), request.score)
        header = await self._storage.task.tasks.get_header(
            intent.submission.graph.graph_id, tenant_id=principal.tenant_id)
        native = (None if header is None else
                  await self._graph.state(intent.submission.graph.graph_id, principal=principal))
        terminal = native is not None and (native.status in _TERMINAL or any(
            node.node_id == "score" and node.status in _TERMINAL for node in native.node_states))
        def reserve(value: EvaluationRecord) -> EvaluationRecord:
            existing = next((item for item in value.human_decisions if item.slot_id == intent.slot_id or
                             item.idempotency_key_digest == decision.idempotency_key_digest), None)
            if existing is not None:
                if existing.request_digest != digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return value
            current = next(item for item in value.intents if item.slot_id == intent.slot_id)
            if (terminal or value.gate == "closed_cancel" or current.released or
                    current.deadline_at is not None and current.deadline_at <= _now() or
                    any(item.trial == intent.trial and item.scorer_slot_id == intent.scorer_slot_id for item in value.scores)):
                raise AIError(ErrorCode.TASK_NOT_READY)
            return replace(value, human_decisions=(*value.human_decisions, decision))
        record = await self._state.update(experiment_id, reserve)
        state = await self._graph_state(intent, principal)
        if state is not None:
            await self._resume_decision(record, intent, state)
            state = await self._graph.state(intent.submission.graph.graph_id, principal=principal)
            await self._collect_score(record, intent, state)
        record = await self._record(experiment_id, principal)
        trials, _ = await self._trials(record, principal)
        return next(item for item in await self._scores(record, trials) if item.trial == intent.trial and
                    item.scorer_slot_id == request.scorer_slot_id)

    async def _resume_decision(
        self, record: EvaluationRecord, intent: EvaluationLaunchIntent, state: TaskGraphState,
    ) -> None:
        if record.gate == "closed_cancel" or intent.released or intent.deadline_at is not None and intent.deadline_at <= _now():
            return
        decision = next((item for item in record.human_decisions if item.slot_id == intent.slot_id), None)
        node = next(item for item in state.node_states if item.node_id == "score")
        if decision is not None and node.status is TaskStatus.WAITING:
            request = TaskInputSupplyRequest(
                decision.actor, node.execution_id, decision.score.to_mapping(),
                f"evaluation-human:{decision.decision_id}")
            if not record.manifest.trial_scope_required:
                await self._graph.resume(state.graph_id, "score", request)
                return
            scope = self._trial_scopes.get((record.experiment_id, intent.slot_id))
            if scope is not None:
                with scope.borrow_engine() as engine:
                    if engine is not None:
                        run = await engine.with_definitions(self._recorder).get(
                            state.graph_id, principal=decision.actor)
                        await run.resume("score", request)


@dataclass(frozen=True, slots=True)
class EvaluationRun:
    _evaluations: RuntimeEvaluations
    experiment_id: str
    _principal: Principal

    async def inspect(self) -> EvaluationView:
        return await self._evaluations._inspect(self.experiment_id, self._principal)

    async def wait(
        self, *, on_event: Callable[[TaskGraphRunEvent], Awaitable[None]] | None = None,
        cursor: str | None = None, include_event_content: bool = False,
        timeout_seconds: float | None = None, close_timeout_seconds: float = 5.0,
    ) -> WaitResult[EvaluationView]:
        _validate_wait(on_event, cursor, include_event_content, timeout_seconds, close_timeout_seconds)

        async def authoritative() -> EvaluationView:
            while True:
                view = await self.inspect()
                if view.completion in {"complete", "cancelled", "needs_attention"}:
                    return view
                await asyncio.sleep(0.05)

        return await _wait(
            scope="evaluation", resource_id=self.experiment_id, waiter=authoritative,
            watch=lambda ready: self._watch_prepared(cursor, include_event_content, ready),
            on_event=on_event, cursor=cursor, timeout_seconds=timeout_seconds,
            close_timeout_seconds=close_timeout_seconds,
            register=self._evaluations._register_observation,
            release=self._evaluations._release_observation,
        )

    def watch(
        self, *, cursor: str | None = None, include_content: bool = False,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        return self._watch_prepared(cursor, include_content, None)

    def _watch_prepared(
        self, cursor: str | None, include_content: bool, ready: asyncio.Event | None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        if not isinstance(include_content, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        cursors = {} if cursor is None else decode_evaluation_watch_cursor(
            self._evaluations._namespace, self._principal.tenant_id, self.experiment_id,
            cursor, include_content=include_content,
        )
        return self._watch(cursors, include_content, ready, cursor)

    async def _watch(
        self, cursors: dict[str, str], include_content: bool,
        ready: asyncio.Event | None, initial_cursor: str | None,
    ) -> AsyncIterator[TaskGraphRunEvent]:
        owner = self._evaluations
        streams: dict[str, AsyncIterator[TaskGraphRunEvent]] = {}
        tasks: dict[str, asyncio.Task[TaskGraphRunEvent]] = {}
        cancelled_by_owner: set[asyncio.Task[TaskGraphRunEvent]] = set()
        known: set[str] = set()
        inactive_sequences: dict[str, int] = {}
        last_cursor = initial_cursor
        finite = False

        def scoped_error(error: BaseException) -> BaseException:
            if not isinstance(error, ObservationError):
                return error
            scoped = ObservationError(
                error.origin, cursor=last_cursor, cause_code=error.cause_code,
                safe_details=error.safe_details, diagnostics=error.diagnostics,
            )
            scoped.__cause__ = error.__cause__
            return scoped

        async def close_streams() -> None:
            errors = await _drain_stream_tasks(
                tasks.values(), cancelled_by_owner=cancelled_by_owner,
                map_error=lambda task, error: scoped_error(error),
            )
            for stream in streams.values():
                try:
                    await stream.aclose()
                except BaseException as error:
                    if not _is_observation_cleanup(error):
                        error = scoped_error(error)
                        errors.append(error)
                        _report_observation_error(error)
            tasks.clear()
            streams.clear()
            if errors:
                fatal = next((error for error in errors if not isinstance(error, ObservationError)), errors[0])
                raise fatal

        async def start(graph_id: str, stream: AsyncIterator[TaskGraphRunEvent], prepared: asyncio.Event | None) -> None:
            streams[graph_id] = stream
            task = asyncio.create_task(stream.__anext__(), name=f"evaluation-graph-{graph_id}")
            tasks[graph_id] = task
            if prepared is None:
                # A finite reader validates its starting watermark before its first item/EOF.
                try:
                    await asyncio.shield(task)
                except StopAsyncIteration:
                    pass
                return
            waiting = asyncio.create_task(prepared.wait())
            try:
                await asyncio.wait({waiting, task}, return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    try:
                        task.result()
                    except StopAsyncIteration:
                        pass
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)

        try:
            while True:
                if not finite:
                    record = await owner._record(self.experiment_id, self._principal)
                    members = {intent.submission.graph.graph_id for intent in record.intents if intent.confirmed}
                    if set(cursors) - members:
                        raise AIError(ErrorCode.CURSOR_INVALID)
                    view = await self.inspect()
                    finite = view.completion in {"complete", "cancelled", "needs_attention"}
                    if finite:
                        record = await owner._record(self.experiment_id, self._principal)
                        members = {intent.submission.graph.graph_id for intent in record.intents if intent.confirmed}
                        # Freeze every graph's durable vector before delivering the final drain.
                        snapshots = {
                            graph_id: await owner._replay_graph(
                                graph_id, self._principal, cursors.get(graph_id), include_content,
                            ) for graph_id in sorted(members)
                        }
                        await close_streams()
                        for graph_id, stream in snapshots.items():
                            await start(graph_id, stream, None)
                    else:
                        pending_members = members - known
                        for graph_id, sequence in tuple(inactive_sequences.items()):
                            state = await owner._graph.state(graph_id, principal=self._principal)
                            if state.event_seq > sequence:
                                pending_members.add(graph_id)
                                inactive_sequences.pop(graph_id)
                        preparation_failure: ObservationError | None = None
                        for graph_id in sorted(pending_members):
                            prepared = asyncio.Event()
                            stream = owner._watch_graph(
                                graph_id, self._principal, cursors.get(graph_id), include_content, prepared,
                            )
                            try:
                                await start(graph_id, stream, prepared)
                            except ObservationError as error:
                                if error.origin != "stream" or error.safe_details.get("phase") == "cleanup":
                                    raise
                                if preparation_failure is None:
                                    preparation_failure = error
                        if preparation_failure is not None:
                            if ready is not None:
                                ready.set()
                            raise preparation_failure
                    known.update(members)
                    if ready is not None:
                        ready.set()
                if not tasks:
                    if finite:
                        return
                    await asyncio.sleep(0.05)
                    continue
                done, _ = await asyncio.wait(
                    set(tasks.values()), timeout=None if finite else 0.05,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    if task.cancelled():
                        task.result()
                    error = task.exception()
                    if error is not None and not isinstance(error, (StopAsyncIteration, ObservationError)):
                        raise error
                for graph_id in sorted(tuple(tasks)):
                    task = tasks[graph_id]
                    if task not in done:
                        continue
                    try:
                        item = task.result()
                    except StopAsyncIteration:
                        tasks.pop(graph_id)
                        await streams.pop(graph_id).aclose()
                        if not finite:
                            graph_cursor = cursors.get(graph_id)
                            inactive_sequences[graph_id] = 0 if graph_cursor is None else decode_graph_watch_cursor(
                                owner._namespace, self._principal.tenant_id, graph_id,
                                graph_cursor, include_content=include_content,
                            )[0]
                        continue
                    if item.cursor is None:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    next_cursors = {**cursors, graph_id: item.cursor}
                    next_cursor = encode_evaluation_watch_cursor(
                        owner._namespace, self._principal.tenant_id, self.experiment_id,
                        include_content=include_content, graph_cursors=next_cursors,
                    )
                    cursors = next_cursors
                    last_cursor = next_cursor
                    yield replace(item, cursor=next_cursor)
                    tasks[graph_id] = asyncio.create_task(streams[graph_id].__anext__())
        except ObservationError as error:
            scoped = ObservationError(
                error.origin, cursor=last_cursor, cause_code=error.cause_code,
                safe_details=error.safe_details, diagnostics=error.diagnostics,
            )
            raise scoped from error.__cause__
        finally:
            active_error = sys.exc_info()[1]
            try:
                await _await_stream_cleanup(close_streams(), active_error)
            except BaseException as error:
                if (isinstance(error, asyncio.CancelledError) or active_error is None
                        or isinstance(active_error, GeneratorExit) or _is_observation_cleanup(active_error)
                        or isinstance(active_error, ObservationError) and active_error.origin == "stream"):
                    raise scoped_error(error)

    async def trials(
        self, *, filters: TrialFilter | None = None, cursor: str | None = None, limit: int = 100,
    ) -> Page[TrialView]:
        return await self._evaluations._page(self.experiment_id, self._principal, filters or TrialFilter(), cursor, limit)

    async def scores(
        self, *, filters: ScoreFilter | None = None, cursor: str | None = None, limit: int = 100,
    ) -> Page[ScoreAttemptView]:
        return await self._evaluations._page(self.experiment_id, self._principal, filters or ScoreFilter(), cursor, limit)

    async def preview_report(self) -> EvaluationReport:
        """Return the current report without publishing a durable report artifact."""
        return (await self._evaluations._build_snapshot(self.experiment_id, self._principal))[1]

    async def create_report(self) -> EvaluationReport:
        return (await self._evaluations._snapshot(self.experiment_id, self._principal))[1]

    async def cancel(self, *, idempotency_key: str) -> EvaluationView:
        return await self._evaluations._cancel(self.experiment_id, self._principal, idempotency_key)

    async def rescore(
        self, request: RescoreRequest, *, engine: "TaskEngine[AppT]",
        trial_scope: EvaluationTrialScopeCallback[ScopeAppT] | None = None,
    ) -> "EvaluationRun":
        return await self._evaluations._rescore(self.experiment_id, self._principal, request, engine,
                                               trial_scope=trial_scope)

    async def submit_human_score(self, request: HumanScoreRequest) -> ScoreAttemptView:
        return await self._evaluations._human_score(self.experiment_id, self._principal, request)


__all__ = ["RuntimeEvaluations", "EvaluationRun"]
