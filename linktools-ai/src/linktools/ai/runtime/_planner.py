#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of generic TaskNodes."""

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai.messages import UserContent

from ..agent import bind_output, restore_output
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExecutionStatus,
    JsonValue,
    Principal,
    ResourceKind,
    ResourceRef,
    TaskStatus,
    WorkspaceFileInput,
    ThinkingValue,
    canonical_sha256,
    deterministic_id,
    normalize_json_value,
    normalize_thinking,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore
from ..task import (
    TaskBindingContract,
    TaskDependency,
    TaskDependencyState,
    TaskEffectResolution,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphState,
    TaskLease,
    TaskNode,
    Task,
    TaskExpansionContext,
    TaskExpander,
    TaskExpanderRef,
    TaskNodeContext,
    TaskNodeInvocation,
    TaskNodeRunError,
    TaskNodeRunControl,
    TaskNodeRunResult,
    TaskResultRecord,
    TaskResultRef,
    TaskRef,
)
from ._agent_task import RuntimeAgentTaskRunner
from ._agent_task_input import AgentTaskInput
from ._input import (
    CanonicalUserInput,
    ExecutionInputMaterializer,
    decode_task_prompt_draft,
    task_prompt_draft,
    validate_user_input,
)
from .state._codec import encode_domain
from .state._contracts import (
    StoredUserInput,
    TaskAdmissionRepository,
    TaskPreparedInputRecord,
)
from ._object import RuntimeObjectKeyFactory
from ._task_graph_binding_capture import (
    TaskGraphBindingCapture,
    TaskGraphBindingCaptureStore,
    builtin_task_declaration,
    task_declaration_semantics,
)
from .service_api import ExecutionService, ExecutionView
from .service_api import ExecutionResult
from .state import ArtifactRecord, ArtifactRepositories, RuntimeDomain

_logger = environ.get_logger("ai.runtime.planner")
AppT = TypeVar("AppT")
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class _TaskCallableAdapter:
    def __init__(self, task: Task[AppT]) -> None:
        if task.function is None:
            raise TypeError("runner-backed Task cannot use function adapter")
        self._task = task

    @property
    def effect_policy(self) -> str:
        return self._task.effect_policy

    @property
    def output_type(self) -> type[BaseModel] | None:
        return self._task.output_type

    @property
    def reconcile(self) -> object:
        return self._task.reconcile_callback

    def normalize(self, input: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return self._task.normalize(input)

    async def run(self, context: TaskNodeContext[AppT]) -> JsonValue:
        function = self._task.function
        if function is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await function(context)

    async def cancel(self, context: TaskNodeContext[AppT]) -> None:
        callback = self._task.cancel_callback
        if callback is not None:
            await callback(context)


class _TaskRunnerAdapter:
    def __init__(self, task: Task[AppT]) -> None:
        runner = task.runner
        if runner is None:
            raise TypeError("function-backed Task cannot use runner adapter")
        self.task = task
        self.runner = runner
        self.effect_policy = task.effect_policy
        self.output_type = task.output_type
        self.reconcile = task.contract["reconcile"]

    def normalize(self, input: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if isinstance(self.runner, RuntimeAgentTaskRunner):
            return self.runner.normalize(input)
        return self.task.normalize(input)

    def normalize_durable(
        self,
        input: Mapping[str, JsonValue],
    ) -> Mapping[str, JsonValue]:
        if isinstance(self.runner, RuntimeAgentTaskRunner):
            return self.runner.normalize_durable(input)
        return self.normalize(input)


class _DeferredInputHandler:
    _ref = TaskRef.deferred_input()
    id = _ref.id
    revision = _ref.revision
    effect_policy = "none"
    output_type = None
    reconcile = None

    def normalize(self, input: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        return normalize_json_value(dict(input))

    async def run(self, context: TaskNodeContext[AppT]) -> JsonValue:
        del context
        raise AIError(ErrorCode.TASK_NOT_READY)

    async def cancel(self, context: TaskNodeContext[AppT]) -> None:
        del context


class _TaskArtifactPublisher:
    def __init__(
        self,
        state: ArtifactRepositories,
        object_store: ObjectStore,
        object_key_factory: RuntimeObjectKeyFactory,
        *,
        principal: Principal,
        graph_id: str,
        node_id: str,
        execution_id: str,
    ) -> None:
        self._state = state
        self._object_store = object_store
        self._object_key_factory = object_key_factory
        self._principal = principal
        self._graph_id = graph_id
        self._node_id = node_id
        self._execution_id = execution_id

    async def publish(
        self,
        name: str,
        chunks: AsyncIterator[bytes],
        *,
        media_type: str,
        expected_size: int,
        expected_digest: str,
    ) -> ArtifactRecord:
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(media_type, str)
            or not media_type
            or isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        artifact_id = deterministic_id(
            "task-artifact",
            self._execution_id,
            name,
        )
        existing = await self._state.records.get_metadata(
            artifact_id,
            tenant_id=self._principal.tenant_id,
        )
        if existing is not None:
            if (
                existing.execution_id != self._execution_id
                or existing.digest != expected_digest
                or existing.media_type != media_type
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return existing
        object_key = self._object_key_factory.key(
            RuntimeDomain.ARTIFACT,
            self._principal.tenant_id,
            expected_digest,
        )
        stat = await self._object_store.put(
            object_key,
            chunks,
            expected_size=expected_size,
            expected_digest=expected_digest,
        )
        object_ref = ObjectRef("runtime", object_key, stat.digest, stat.size)
        record = ArtifactRecord(
            artifact_id,
            self._execution_id,
            self._principal.tenant_id,
            f"task:{self._graph_id}:{self._node_id}",
            media_type,
            object_ref,
            datetime.now(timezone.utc),
        )
        return await self._state.records.put_metadata(record)


class _TaskStateReader(Protocol):
    async def get_header(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None: ...

    async def graph_state(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphState | None: ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]: ...

    async def get_prepared_input(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
    ) -> TaskPreparedInputRecord | None: ...

    async def publish_prepared_input(
        self,
        lease: TaskLease,
        record: TaskPreparedInputRecord,
        *,
        tenant_id: str,
    ) -> TaskPreparedInputRecord: ...


class _TaskExpansionContext:
    __slots__ = (
        "_graph_id",
        "_output",
        "_principal",
        "_source_node",
    )

    def __init__(
        self,
        principal: Principal,
        graph_id: str,
        source_node: TaskNode,
        output: JsonValue,
    ) -> None:
        self._principal = principal
        self._graph_id = graph_id
        self._source_node = source_node
        self._output = normalize_json_value(output)

    @property
    def principal(self) -> Principal:
        return self._principal

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def source_node(self) -> TaskNode:
        return self._source_node

    @property
    def output(self) -> JsonValue:
        return normalize_json_value(self._output)

class RuntimeTaskNodeRunner(Generic[AppT]):
    """Interpret admitted TaskNodes using the captured Runtime handler map."""

    def __init__(
        self,
        execution: ExecutionService,
        *,
        namespace: str,
        app: AppT,
        authorization: AuthorizationPolicy,
        task_state: _TaskStateReader,
        task_admissions: TaskAdmissionRepository,
        task_objects: ObjectStore,
        artifact_state: ArtifactRepositories | None = None,
        artifact_objects: ObjectStore | None = None,
        object_key_factory: RuntimeObjectKeyFactory,
        binding_captures: TaskGraphBindingCaptureStore,
        tasks: Sequence[Task[AppT]] = (),
        expanders: Sequence[TaskExpander] = (),
        task_durable: bool = False,
        execution_durable: bool = True,
        recovery_durable: bool = True,
        input_materializer: ExecutionInputMaterializer | None = None,
    ) -> None:
        self._app = app
        self._namespace = namespace
        self._authorization = authorization
        self._execution = execution
        self._task_state = task_state
        self._task_admissions = task_admissions
        self._task_objects = task_objects
        self._input_materializer = input_materializer
        self._artifact_state = artifact_state
        self._artifact_objects = artifact_objects
        self._object_key_factory = object_key_factory
        if not isinstance(binding_captures, TaskGraphBindingCaptureStore):
            raise TypeError("binding_captures must be TaskGraphBindingCaptureStore")
        self._binding_capture_store = binding_captures
        self._admitted_binding_captures: dict[str, TaskGraphBindingCapture] = {}
        self._definition_lock = asyncio.Lock()
        self._active_definitions: dict[
            str,
            tuple[
                Mapping[tuple[str, int], Task[AppT]],
                Mapping[tuple[str, int], TaskExpander],
            ],
        ] = {}
        self._pending_definition_activations: dict[str, int] = {}
        self._deferred_input = _DeferredInputHandler()
        self._default_definitions = self._definition_maps(tasks, expanders)
        self._task_durable = task_durable
        self._execution_durable = execution_durable
        self._recovery_durable = recovery_durable

    def _definition_maps(
        self,
        tasks: Sequence[Task[AppT]],
        expanders: Sequence[TaskExpander],
    ) -> tuple[
        Mapping[tuple[str, int], Task[AppT]],
        Mapping[tuple[str, int], TaskExpander],
    ]:
        task_map: dict[tuple[str, int], Task[AppT]] = {}
        for task in tasks:
            if not isinstance(task, Task):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            identity = (task.ref.id, task.ref.revision)
            if identity in task_map:
                raise AIError(ErrorCode.BINDING_CONFLICT)
            task_map[identity] = task
        expander_map: dict[tuple[str, int], TaskExpander] = {}
        for expander in expanders:
            if not isinstance(expander, TaskExpander):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            identity = (expander.id, expander.revision)
            if identity in expander_map:
                raise AIError(ErrorCode.BINDING_CONFLICT)
            expander_map[identity] = expander
        return (
            MappingProxyType(dict(sorted(task_map.items()))),
            MappingProxyType(dict(sorted(expander_map.items()))),
        )

    async def activate_graph(
        self,
        graph: TaskGraph,
        tasks: Sequence[Task[AppT]],
        expanders: Sequence[TaskExpander],
        *,
        track_pre_admission: bool = False,
    ) -> object | None:
        proposed = self._definition_maps(tasks, expanders)
        task_map, expander_map = proposed
        async with self._definition_lock:
            active = self._active_definitions.get(graph.graph_id)
            if active is None:
                self._active_definitions[graph.graph_id] = proposed
                active = proposed
            else:
                current_tasks, current_expanders = active
                required_tasks = {
                    (node.task.id, node.task.revision)
                    for node in graph.nodes
                    if node.task is not None
                }
                required_expanders = {
                    (node.expander.id, node.expander.revision)
                    for node in graph.nodes
                    if node.expander is not None
                }
                for identity in required_tasks:
                    previous = current_tasks.get(identity)
                    replacement = task_map.get(identity)
                    if previous is None and identity == (
                        self._deferred_input.id,
                        self._deferred_input.revision,
                    ):
                        continue
                    if (
                        previous is None
                        or replacement is None
                        or task_declaration_semantics(
                            {
                                "id": previous.id,
                                "revision": previous.revision,
                                **dict(previous.contract),
                            }
                        )
                        != task_declaration_semantics(
                            {
                                "id": replacement.id,
                                "revision": replacement.revision,
                                **dict(replacement.contract),
                            }
                        )
                    ):
                        missing_node = next(
                            (
                                node
                                for node in graph.nodes
                                if node.task is not None
                                and (node.task.id, node.task.revision) == identity
                            ),
                            None,
                        )
                        raise AIError(
                            ErrorCode.BINDING_NOT_REGISTERED,
                            safe_details={
                                "graph_id": graph.graph_id,
                                "node_id": (
                                    None
                                    if missing_node is None
                                    else missing_node.node_id
                                ),
                                "role": "task",
                                "task_id": identity[0],
                                "task_revision": identity[1],
                                "ref": f"{identity[0]}@{identity[1]}",
                            },
                        )
                for identity in required_expanders:
                    previous = current_expanders.get(identity)
                    replacement = expander_map.get(identity)
                    if previous is None or replacement is None:
                        missing_node = next(
                            (
                                node
                                for node in graph.nodes
                                if node.expander is not None
                                and (node.expander.id, node.expander.revision) == identity
                            ),
                            None,
                        )
                        raise AIError(
                            ErrorCode.BINDING_NOT_REGISTERED,
                            safe_details={
                                "graph_id": graph.graph_id,
                                "node_id": (
                                    None
                                    if missing_node is None
                                    else missing_node.node_id
                                ),
                                "role": "expander",
                                "expander_id": identity[0],
                                "expander_revision": identity[1],
                                "ref": f"{identity[0]}@{identity[1]}",
                            },
                        )
            if track_pre_admission:
                self._pending_definition_activations[graph.graph_id] = (
                    self._pending_definition_activations.get(graph.graph_id, 0) + 1
                )
                return active
            return None

    async def finish_graph_activation(
        self,
        graph_id: str,
        tenant_id: str,
        activation: object,
        *,
        admitted: bool,
    ) -> None:
        persisted = admitted
        if not persisted:
            persisted = (
                await self._task_admissions.get(
                    graph_id,
                    tenant_id=tenant_id,
                )
            ) is not None
        async with self._definition_lock:
            pending = self._pending_definition_activations.get(graph_id)
            if pending is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if pending == 1:
                del self._pending_definition_activations[graph_id]
            else:
                self._pending_definition_activations[graph_id] = pending - 1
            if (
                not persisted
                and pending == 1
                and self._active_definitions.get(graph_id) is activation
            ):
                self._active_definitions.pop(graph_id, None)
                self._admitted_binding_captures.pop(graph_id, None)

    def _definitions_for(
        self,
        graph_id: str,
    ) -> tuple[
        Mapping[tuple[str, int], Task[AppT]],
        Mapping[tuple[str, int], TaskExpander],
    ]:
        return self._active_definitions.get(graph_id, self._default_definitions)

    def _artifact_publisher(
        self,
        principal: Principal,
        graph_id: str,
        node_id: str,
        execution_id: str,
    ) -> "_TaskArtifactPublisher | None":
        if self._artifact_state is None or self._artifact_objects is None:
            return None
        return _TaskArtifactPublisher(
            self._artifact_state,
            self._artifact_objects,
            self._object_key_factory,
            principal=principal,
            graph_id=graph_id,
            node_id=node_id,
            execution_id=execution_id,
        )

    async def capture_admission(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> TaskGraph:
        tasks, expanders = self._definitions_for(admission.graph_id)
        binding_capture = await self._binding_capture_store.capture(
            admission,
            graph,
            tasks=tuple(tasks.values()),
            expanders=tuple(expanders.values()),
        )
        self._admitted_binding_captures[admission.graph_id] = binding_capture
        return TaskGraph(
            graph.graph_id,
            tuple(
                [
                    await self._materialize_node_input(
                        node,
                        admission.principal.tenant_id,
                        tasks,
                    )
                    for node in graph.nodes
                ]
            ),
        )

    async def _materialize_node_input(
        self,
        node: TaskNode,
        tenant_id: str,
        tasks: Mapping[tuple[str, int], Task[AppT]],
    ) -> TaskNode:
        reference = node.task
        if reference is None:
            return node
        task = tasks.get((reference.id, reference.revision))
        if task is None or task.contract.get("type") != "agent":
            return node
        task_input = AgentTaskInput.from_mapping(node.input)
        config = task.contract.get("config")
        input_mode = config.get("input_mode") if isinstance(config, Mapping) else None
        if task_input.stored_prompt is not None:
            return node
        if input_mode == "projected":
            if not task_input.files:
                return node
            materializer = self._input_materializer
            if materializer is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            body = dict(task_input)
            body["files"] = list(
                await materializer.canonicalize_files(task_input.files)
            )
            return TaskNode.from_resolved(
                node.node_id,
                node.dependencies,
                task=node.task,
                input=body,
                budget_cost=node.budget_cost,
                expander=node.expander,
                input_refs=node.input_refs,
                timeout_seconds=node.timeout_seconds,
                max_attempts=node.max_attempts,
                retry_delay_seconds=node.retry_delay_seconds,
                output_contract=node.output_contract,
                effect_policy=node.effect_policy,
                reconcile=node.reconcile,
                dependency_policy=node.dependency_policy,
                failure_policy=node.failure_policy,
            )
        if not task_input.files and isinstance(task_input.prompt, str):
            return node
        materializer = self._input_materializer
        if materializer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        canonical = await materializer.canonicalize_input(task_input.prompt)
        content = await materializer.materialize(canonical, task_input.files)
        stored = await materializer.store(content, tenant_id=tenant_id)
        body = dict(task_input)
        body["prompt"] = {
            "kind": "stored-user-content-v1",
            "intent": "task-admission",
            "source_intent_digest": canonical_sha256(task_input["prompt"]),
            "value": encode_domain(stored),
        }
        _logger.info("task input resolved: node=%s", node.node_id)
        return TaskNode.from_resolved(
            node.node_id,
            node.dependencies,
            task=node.task,
            input=body,
            budget_cost=node.budget_cost,
            expander=node.expander,
            input_refs=node.input_refs,
            timeout_seconds=node.timeout_seconds,
            max_attempts=node.max_attempts,
            retry_delay_seconds=node.retry_delay_seconds,
            output_contract=node.output_contract,
            effect_policy=node.effect_policy,
            reconcile=node.reconcile,
            dependency_policy=node.dependency_policy,
            failure_policy=node.failure_policy,
        )

    async def restore_prepared_agent_prompt(
        self,
        value: StoredUserInput,
    ) -> CanonicalUserInput:
        materializer = self._input_materializer
        if materializer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await materializer.restore(value)

    async def store_prepared_agent_prompt(
        self,
        value: CanonicalUserInput,
        *,
        files: Sequence[str],
        tenant_id: str,
    ) -> StoredUserInput:
        materializer = self._input_materializer
        if materializer is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        canonical = await materializer.canonicalize_input(value)
        materialized = await materializer.materialize(canonical, files)
        return await materializer.store(materialized, tenant_id=tenant_id)

    async def get_prepared_agent_input(
        self,
        invocation: TaskNodeInvocation,
    ) -> TaskPreparedInputRecord | None:
        task_ref = invocation.node.task
        if task_ref is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        admission = await self._task_admissions.get(
            invocation.graph_id,
            tenant_id=invocation.principal.tenant_id,
        )
        if admission is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = await self._task_state.get_prepared_input(
            invocation.graph_id,
            invocation.node.node_id,
            tenant_id=invocation.principal.tenant_id,
        )
        if record is None:
            return None
        if (
            record.admission_digest != admission.initial_request_digest
            or record.task_ref != task_ref
            or record.tenant_id != invocation.principal.tenant_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for name, reference in record.source_refs:
            if await self.read_input_result_ref(invocation, name) != reference:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return record

    async def publish_prepared_agent_input(
        self,
        invocation: TaskNodeInvocation,
        *,
        input_identity: str,
        source_refs: tuple[tuple[str, TaskResultRef], ...],
        stored_user_input: StoredUserInput,
        final_input_digest: str,
        request_identity: str,
    ) -> TaskPreparedInputRecord:
        lease = invocation.task_lease
        task_ref = invocation.node.task
        if lease is None or task_ref is None:
            raise AIError(ErrorCode.TASK_FENCE_STALE)
        admission = await self._task_admissions.get(
            invocation.graph_id,
            tenant_id=invocation.principal.tenant_id,
        )
        if admission is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        candidate = TaskPreparedInputRecord(
            invocation.graph_id,
            invocation.node.node_id,
            invocation.principal.tenant_id,
            admission.initial_request_digest,
            task_ref,
            input_identity,
            source_refs,
            stored_user_input,
            final_input_digest,
            request_identity,
            lease.fence,
        )
        return await self._task_state.publish_prepared_input(
            lease,
            candidate,
            tenant_id=invocation.principal.tenant_id,
        )

    async def load_admission(
        self,
        admission: TaskGraphAdmission,
    ) -> None:
        binding_capture = await self._binding_capture_store.load(admission)
        self._admitted_binding_captures[admission.graph_id] = binding_capture

    def _require_binding_capture(
        self,
        graph_id: str,
    ) -> TaskGraphBindingCapture:
        binding_capture = self._admitted_binding_captures.get(graph_id)
        if binding_capture is None:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={"graph_id": graph_id},
            )
        return binding_capture

    def _validate_task_declaration(
        self,
        task_id: str,
        task_revision: int,
        *,
        graph_id: str,
        node_id: str,
    ) -> Mapping[str, JsonValue]:
        identity = (task_id, task_revision)
        capture = self._require_binding_capture(graph_id)
        declared = capture.tasks.get(identity)
        tasks, _expanders = self._definitions_for(graph_id)
        task = tasks.get(identity)
        current = None if task is None else {
            "id": task.id,
            "revision": task.revision,
            **dict(task.contract),
        }
        if task is None:
            current = builtin_task_declaration(task_id, task_revision)
        if declared is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "task",
                    "ref": f"{task_id}@{task_revision}",
                },
            )
        if current is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "task",
                    "ref": f"{task_id}@{task_revision}",
                },
            )
        if task_declaration_semantics(declared) != task_declaration_semantics(current):
            reason = "task_declaration_changed"
            if declared.get("effect_policy") != current.get("effect_policy"):
                reason = "task_effect_changed"
            elif declared.get("output_contract") != current.get("output_contract"):
                reason = "task_output_contract_changed"
            elif declared.get("reconcile") != current.get("reconcile"):
                reason = "task_reconcile_changed"
            elif declared.get("cancel") != current.get("cancel"):
                reason = "task_cancel_changed"
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "kind": "task",
                    "task_id": task_id,
                    "task_revision": task_revision,
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "task",
                    "reason": reason,
                },
            )
        return declared

    def _validate_expander_declaration(
        self,
        reference: TaskExpanderRef,
        *,
        graph_id: str,
        node_id: str,
    ) -> Mapping[str, JsonValue]:
        identity = (reference.id, reference.revision)
        capture = self._require_binding_capture(graph_id)
        declared = capture.expanders.get(identity)
        _tasks, expanders = self._definitions_for(graph_id)
        expander = expanders.get(identity)
        current = (
            None
            if expander is None
            else {"version": 1, "id": expander.id, "revision": expander.revision}
        )
        if declared is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "expander",
                    "ref": f"{reference.id}@{reference.revision}",
                },
            )
        if current is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "expander",
                    "ref": f"{reference.id}@{reference.revision}",
                },
            )
        if dict(declared) != current:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "kind": "task_expander",
                    "expander_id": reference.id,
                    "expander_revision": reference.revision,
                    "graph_id": graph_id,
                    "node_id": node_id,
                    "role": "expander",
                    "reason": "task_expander_declaration_changed",
                },
            )
        return declared

    async def prepare_node(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        principal: Principal,
    ) -> None:
        self._validate_durability(
            node,
            graph_id=graph_id,
            request=False,
        )
        acquired: list[tuple[str, str]] = []
        try:
            for execution_id in await self._input_ref_execution_ids(
                node,
                principal=principal,
                authorize=True,
            ):
                hold_id = _task_dependency_hold_id(graph_id, execution_id)
                created = await self._execution.acquire_dependency_hold(
                    execution_id,
                    tenant_id=principal.tenant_id,
                    hold_id=hold_id,
                )
                if created:
                    acquired.append((execution_id, hold_id))
        except BaseException:
            for execution_id, hold_id in reversed(acquired):
                try:
                    await self._execution.release_dependency_hold(
                        execution_id,
                        tenant_id=principal.tenant_id,
                        hold_id=hold_id,
                    )
                except BaseException as cleanup_error:  # noqa: BLE001
                    _logger.warning(
                        "task dependency hold rollback failed: graph=%s "
                        "execution=%s error=%s",
                        graph_id,
                        execution_id,
                        type(cleanup_error).__name__,
                    )
            raise

    async def prepare_graph(
        self,
        graph_state: TaskGraphState,
        *,
        principal: Principal,
    ) -> None:
        if graph_state.graph_id == "":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared: list[TaskNode] = []
        try:
            for node, node_state in zip(
                graph_state.nodes,
                graph_state.node_states,
                strict=True,
            ):
                if node_state.node_id != node.node_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self.prepare_node(
                    node,
                    graph_id=graph_state.graph_id,
                    principal=principal,
                )
                prepared.append(node)
        except BaseException:
            await self._release_nodes_dependencies(
                graph_state.graph_id,
                prepared,
                tenant_id=principal.tenant_id,
            )
            raise

    async def release_graph_dependencies(
        self,
        graph_state: TaskGraphState,
        *,
        tenant_id: str,
    ) -> None:
        await self._release_nodes_dependencies(
            graph_state.graph_id,
            graph_state.nodes,
            tenant_id=tenant_id,
        )
        self._admitted_binding_captures.pop(graph_state.graph_id, None)
        async with self._definition_lock:
            self._active_definitions.pop(graph_state.graph_id, None)

    async def _release_nodes_dependencies(
        self,
        graph_id: str,
        nodes: Sequence[TaskNode],
        *,
        tenant_id: str,
    ) -> None:
        releases: set[tuple[str, str]] = set()
        for node in nodes:
            for execution_id in await self._input_ref_execution_ids(
                node,
                tenant_id=tenant_id,
                authorize=False,
            ):
                releases.add(
                    (
                        execution_id,
                        _task_dependency_hold_id(graph_id, execution_id),
                    )
                )
        for execution_id, hold_id in sorted(releases):
            await self._execution.release_dependency_hold(
                execution_id,
                tenant_id=tenant_id,
                hold_id=hold_id,
            )

    async def _input_ref_execution_ids(
        self,
        node: TaskNode,
        *,
        principal: "Principal | None" = None,
        tenant_id: "str | None" = None,
        authorize: bool,
    ) -> tuple[str, ...]:
        resolved_tenant = (
            principal.tenant_id
            if principal is not None
            else tenant_id
        )
        if resolved_tenant is None:
            raise TypeError("tenant identity is required")
        grouped: dict[str, list[TaskResultRef]] = {}
        for reference in node.input_refs.values():
            if (
                reference.namespace != self._namespace
                or reference.tenant_id != resolved_tenant
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            grouped.setdefault(reference.graph_id, []).append(reference)

        execution_ids: set[str] = set()
        for source_graph_id, references in sorted(grouped.items()):
            if authorize:
                if principal is None:
                    raise TypeError("principal is required for authorization")
                header = await self._task_state.get_header(
                    source_graph_id,
                    tenant_id=resolved_tenant,
                )
                if header is None:
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                if (
                    header.kind is not ResourceKind.TASK_GRAPH
                    or header.id != source_graph_id
                    or header.tenant_id != resolved_tenant
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                await self._authorization.authorize(
                    principal,
                    AuthorizationAction.TASK_READ,
                    header,
                )
            node_ids = tuple(
                dict.fromkeys(
                    reference.node_id
                    for reference in references
                )
            )
            records = await self._task_state.get_results(
                source_graph_id,
                node_ids,
                tenant_id=resolved_tenant,
            )
            graph_state = await self._task_state.graph_state(
                source_graph_id,
                tenant_id=resolved_tenant,
            )
            if graph_state is None:
                raise AIError(ErrorCode.TASK_NOT_READY)
            states = {
                node_state.node_id: node_state
                for node_state in graph_state.node_states
            }
            for reference in references:
                record = records.get(reference.node_id)
                node_state = states.get(reference.node_id)
                if (
                    record is None
                    or node_state is None
                    or node_state.status is not TaskStatus.SUCCEEDED
                    or node_state.result_digest != reference.result_digest
                    or node_state.execution_id is None
                    or (
                        record.execution_id is not None
                        and record.execution_id != node_state.execution_id
                    )
                ):
                    raise AIError(ErrorCode.TASK_NOT_READY)
                execution_id = node_state.execution_id
                if authorize:
                    assert principal is not None
                    execution = await self._execution.inspect(
                        execution_id,
                        principal=principal,
                    )
                    if execution.status is not ExecutionStatus.SUCCEEDED:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                execution_ids.add(execution_id)
        return tuple(sorted(execution_ids))

    def admit_node(self, node: TaskNode, *, graph_id: str) -> TaskNode:
        task_id, task_revision, body = _parse_node(node, request=True)
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=graph_id,
            node_id=node.node_id,
            request=True,
        )
        if node.expander is not None:
            self._resolve_expander(
                node.expander,
                graph_id=graph_id,
                request=True,
                node_id=node.node_id,
            )
        try:
            normalized = handler.normalize(body)
            canonical_body = _normalize_handler_body(normalized)
            if (
                isinstance(handler, _TaskRunnerAdapter)
                and isinstance(handler.runner, RuntimeAgentTaskRunner)
                and AgentTaskInput.from_mapping(canonical_body).stored_prompt
                is not None
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        handler_output_type = getattr(handler, "output_type", None)
        if (
            handler_output_type is not None
            and node.output_type is not None
            and _output_contract(handler, None)
            != _output_contract(None, node.output_type)
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return TaskNode.from_resolved(
            node.node_id,
            node.dependencies,
            task=node.task,
            input=canonical_body,
            budget_cost=node.budget_cost,
            expander=node.expander,
            input_refs=node.input_refs,
            timeout_seconds=node.timeout_seconds,
            max_attempts=node.max_attempts,
            retry_delay_seconds=node.retry_delay_seconds,
            output_contract=_output_contract(handler, node.output_type),
            effect_policy=_handler_effect_policy(handler, request=True),
            reconcile=_handler_has_reconcile(handler),
            dependency_policy=node.dependency_policy,
            failure_policy=node.failure_policy,
        )

    def admit_request(self, graph: TaskGraph) -> TaskGraph:
        canonical = TaskGraph(
            graph.graph_id,
            tuple(
                self.admit_node(node, graph_id=graph.graph_id)
                for node in graph.nodes
            ),
        )
        for node in canonical.nodes:
            self._validate_durability(
                node,
                graph_id=graph.graph_id,
                request=True,
            )
        return canonical

    def validate_input(self, node: TaskNode, value: JsonValue) -> None:
        del value
        task_id, task_revision, _body = _parse_node(node, request=False)
        if (
            task_id != self._deferred_input.id
            or task_revision != self._deferred_input.revision
        ):
            raise AIError(ErrorCode.TASK_NOT_READY)

    def validate_effect_resolution(
        self,
        node: TaskNode,
        resolution: TaskEffectResolution,
    ) -> None:
        del resolution
        if node.effect_policy != "non_replay_safe":
            raise AIError(ErrorCode.TASK_NOT_READY)

    def validate_recovery(self, graph_state: TaskGraphState) -> None:
        for node, node_state in zip(
            graph_state.nodes,
            graph_state.node_states,
            strict=True,
        ):
            if node_state.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            }:
                continue
            self._validate_durability(
                node,
                graph_id=graph_state.graph_id,
                request=False,
            )
            task_id, task_revision, body = _parse_node(node, request=False)
            task_declaration = self._validate_task_declaration(
                task_id,
                task_revision,
                graph_id=graph_state.graph_id,
                node_id=node.node_id,
            )
            handler = self._handler(
                task_id,
                task_revision,
                graph_id=graph_state.graph_id,
                node_id=node.node_id,
                request=False,
            )
            if node.expander is not None:
                self._validate_expander_declaration(
                    node.expander,
                    graph_id=graph_state.graph_id,
                    node_id=node.node_id,
                )
                self._resolve_expander(
                    node.expander,
                    graph_id=graph_state.graph_id,
                    request=False,
                    node_id=node.node_id,
                )
            if node.effect_policy != _handler_effect_policy(handler, request=False):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph_state.graph_id,
                        "node_id": node.node_id,
                        "reason": "task_effect_changed",
                    },
                )
            declared_output = task_declaration["output_contract"]
            if (
                isinstance(declared_output, Mapping)
                and declared_output.get("kind") == "schema"
                and node.output_contract != _output_contract(handler, None)
            ):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph_state.graph_id,
                        "node_id": node.node_id,
                        "reason": "task_output_contract_changed",
                    },
                )
            if (
                isinstance(declared_output, Mapping)
                and declared_output.get("kind") == "json"
                and node.output_contract is not None
            ):
                try:
                    _restore_output_contract(node.output_contract)
                except AIError as error:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        safe_details={
                            "graph_id": graph_state.graph_id,
                            "node_id": node.node_id,
                            "reason": "task_node_output_contract_invalid",
                        },
                    ) from error
            if node.reconcile != _handler_has_reconcile(handler):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph_state.graph_id,
                        "node_id": node.node_id,
                        "reason": "task_reconcile_changed",
                    },
            )
            try:
                normalized_body = (
                    handler.normalize_durable(body)
                    if isinstance(handler, _TaskRunnerAdapter)
                    else handler.normalize(body)
                )
                canonical_body = _normalize_handler_body(normalized_body)
            except (AIError, TypeError, ValueError) as error:
                if (
                    isinstance(error, AIError)
                    and error.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
                ):
                    raise
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph_state.graph_id,
                        "node_id": node.node_id,
                        "task_id": task_id,
                        "task_revision": task_revision,
                    },
                ) from error
            canonical = TaskNode.from_resolved(
                node.node_id,
                node.dependencies,
                task=node.task,
                input=canonical_body,
                budget_cost=node.budget_cost,
                expander=node.expander,
                input_refs=node.input_refs,
                timeout_seconds=node.timeout_seconds,
                max_attempts=node.max_attempts,
                retry_delay_seconds=node.retry_delay_seconds,
                output_contract=node.output_contract,
                effect_policy=node.effect_policy,
                reconcile=node.reconcile,
                dependency_policy=node.dependency_policy,
                failure_policy=node.failure_policy,
            )
            if canonical.input != node.input:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": graph_state.graph_id,
                        "node_id": node.node_id,
                        "task_id": task_id,
                        "task_revision": task_revision,
                    },
                )

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        dependency_states = invocation.dependency_states
        task_id, task_revision, body = _parse_node(node, request=False)
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=graph_id,
            node_id=node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            runner_result = await handler.runner.run(invocation, control=control)
            return await self._complete_runner_result(invocation, runner_result)
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
            dependency_states=dependency_states,
            principal=principal,
            graph_id=graph_id,
        )
        binding = _task_binding(node, handler, task_id, task_revision)
        idempotency_key = _custom_idempotency_key(
            graph_id,
            node,
            principal,
            dependencies,
            dependency_states,
        )
        handle = await self._execution.start_task(
            binding,
            principal=principal,
            input=body,
            idempotency_key=idempotency_key,
            correlation=correlation,
        )
        execution_id = handle.execution_id
        if control.execution_id is None:
            await control.bind_execution(execution_id)
        elif control.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        if handler is self._deferred_input:
            view = await self._execution.inspect(execution_id, principal=principal)
            if view.status is ExecutionStatus.SUCCEEDED:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                return await self._complete_output(
                    node,
                    result.output,
                    execution_id=execution_id,
                    principal=principal,
                    graph_id=graph_id,
                )
            if view.status is ExecutionStatus.STARTED:
                await self._execution.defer_task_input(
                    execution_id,
                    principal=principal,
                    wait_id=execution_id,
                )
            elif view.status is not ExecutionStatus.WAITING_DEFERRED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await control.handoff_execution(
                execution_id,
                occupies_concurrency=False,
            )
            return TaskNodeRunResult(
                canonical_sha256({"wait_id": execution_id}),
                execution_id,
                deferred=True,
            )

        return await self._run_custom_execution(
            node,
            handler,
            body,
            dependencies,
            dependency_states,
            principal=principal,
            correlation=correlation,
            graph_id=graph_id,
            execution_id=execution_id,
        )

    async def _run_custom_execution(
        self,
        node: TaskNode,
        handler: _TaskCallableAdapter,
        body: Mapping[str, JsonValue],
        dependencies: Mapping[str, TaskDependency],
        dependency_states: Mapping[str, TaskDependencyState],
        *,
        principal: Principal,
        correlation: Mapping[str, str | int],
        graph_id: str,
        execution_id: str,
    ) -> TaskNodeRunResult:
        view = await self._execution.inspect(execution_id, principal=principal)
        if view.status is ExecutionStatus.SUCCEEDED:
            result = await self._execution.result(execution_id, principal=principal)
            return await self._complete_output(
                node,
                result.output,
                execution_id=execution_id,
                principal=principal,
                graph_id=graph_id,
            )
        if view.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await self._execution.result(execution_id, principal=principal)
            raise TaskNodeRunError(
                ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                execution_id,
                safe_details=result.safe_error_details,
            )
        if view.status is ExecutionStatus.RECOVERY_REQUIRED:
            return await self._reconcile_custom_execution(
                node,
                handler,
                body,
                dependencies,
                dependency_states,
                principal=principal,
                correlation=correlation,
                graph_id=graph_id,
                execution_id=execution_id,
            )
        if view.status is ExecutionStatus.WAITING_RETRY:
            claimed = await self._execution.claim_task_attempt(
                execution_id,
                principal=principal,
            )
            if claimed.status is ExecutionStatus.WAITING_RETRY:
                if claimed.task_next_attempt_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return TaskNodeRunResult(
                    canonical_sha256(
                        {
                            "execution_id": execution_id,
                            "retry_at": claimed.task_next_attempt_at.isoformat(),
                        }
                    ),
                    execution_id,
                    retry_at=claimed.task_next_attempt_at,
                )
            if claimed.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                raise TaskNodeRunError(
                    ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                    execution_id,
                    safe_details=result.safe_error_details,
                )
            if claimed.status is ExecutionStatus.SUCCEEDED:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                return await self._complete_output(
                    node,
                    result.output,
                    execution_id=execution_id,
                    principal=principal,
                    graph_id=graph_id,
                )
            if (
                claimed.status is not ExecutionStatus.STARTED
                or claimed.task_attempt <= view.task_attempt
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            if view.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if view.task_attempt > 0 and node.effect_policy == "non_replay_safe":
                await self._execution.require_task_recovery(
                    execution_id,
                    principal=principal,
                    error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                )
                raise TaskNodeRunError(ErrorCode.TASK_EFFECT_UNKNOWN, execution_id)
            claimed = await self._execution.claim_task_attempt(
                execution_id,
                principal=principal,
            )
            if claimed.status is ExecutionStatus.RECOVERY_REQUIRED:
                raise TaskNodeRunError(
                    ErrorCode.TASK_EFFECT_UNKNOWN,
                    execution_id,
                )
            if claimed.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                raise TaskNodeRunError(
                    ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                    execution_id,
                    safe_details=result.safe_error_details,
                )
            if claimed.status is ExecutionStatus.SUCCEEDED:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                return await self._complete_output(
                    node,
                    result.output,
                    execution_id=execution_id,
                    principal=principal,
                    graph_id=graph_id,
                )
            if claimed.status is not ExecutionStatus.STARTED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if claimed.task_attempt <= view.task_attempt:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


        context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            execution_id,
            body,
            dependencies,
            _custom_idempotency_key(
                graph_id,
                node,
                principal,
                dependencies,
                dependency_states,
            ),
            lambda dependency: self._read_dependency(
                dependency,
                principal=principal,
            ),
            correlation,
            artifacts=self._artifact_publisher(
                principal,
                graph_id,
                node.node_id,
                execution_id,
            ),
            dependency_states=dependency_states,
        )
        try:
            timeout = None
            if claimed.task_deadline_at is not None:
                timeout = (
                    claimed.task_deadline_at - datetime.now(timezone.utc)
                ).total_seconds()
                if timeout <= 0:
                    raise asyncio.TimeoutError
            raw_output = (
                await handler.run(context)
                if timeout is None
                else await asyncio.wait_for(handler.run(context), timeout=timeout)
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as error:
            return await self._settle_custom_failure(
                node,
                execution_id,
                principal,
                AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT),
                unknown_effect=node.effect_policy == "non_replay_safe",
                cause=error,
                attempt=claimed,
            )
        except AIError as error:
            return await self._settle_custom_failure(
                node,
                execution_id,
                principal,
                error,
                unknown_effect=node.effect_policy == "non_replay_safe",
                attempt=claimed,
            )
        except Exception as error:  # noqa: BLE001
            return await self._settle_custom_failure(
                node,
                execution_id,
                principal,
                AIError(ErrorCode.TASK_NODE_FAILED),
                unknown_effect=node.effect_policy == "non_replay_safe",
                cause=error,
                attempt=claimed,
            )

        try:
            output = normalize_json_value(raw_output)
            _validate_task_output(node, output)
        except (AIError, TypeError, ValueError) as error:
            failure = (
                error
                if isinstance(error, AIError)
                else AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            )
            if node.effect_policy == "non_replay_safe":
                await self._execution.require_task_recovery(
                    execution_id,
                    principal=principal,
                    error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                    attempt=claimed,
                )
                raise TaskNodeRunError(
                    ErrorCode.TASK_EFFECT_UNKNOWN,
                    execution_id,
                ) from error
            await self._execution.fail_task(
                execution_id,
                principal=principal,
                error=failure,
                attempt=claimed,
            )
            raise TaskNodeRunError(
                failure.code,
                execution_id,
                safe_details=failure.safe_details,
            ) from error

        result = await self._execution.complete_task(
            execution_id,
            principal=principal,
            output=output,
            attempt=claimed,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._complete_output(
            node,
            result.output,
            execution_id=execution_id,
            principal=principal,
            graph_id=graph_id,
        )

    async def _settle_custom_failure(
        self,
        node: TaskNode,
        execution_id: str,
        principal: Principal,
        error: AIError,
        *,
        unknown_effect: bool,
        cause: BaseException | None = None,
        attempt: ExecutionView,
    ) -> TaskNodeRunResult:
        if unknown_effect:
            await self._execution.require_task_recovery(
                execution_id,
                principal=principal,
                error_code=ErrorCode.TASK_EFFECT_UNKNOWN.value,
                attempt=attempt,
            )
            raised = TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
            )
            if cause is not None:
                raise raised from cause
            raise raised from error

        view = await self._execution.inspect(execution_id, principal=principal)
        retryable = (
            error.retryable
            and view.task_attempt < node.max_attempts
            and error.code is not ErrorCode.EXECUTION_WAIT_TIMEOUT
        )
        if retryable:
            retry = await self._execution.schedule_task_retry(
                execution_id,
                principal=principal,
                error_code=error.code.value,
                attempt=attempt,
            )
            if retry.status is ExecutionStatus.FAILED:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                raise TaskNodeRunError(
                    ErrorCode(result.error_code or error.code.value),
                    execution_id,
                    safe_details=result.safe_error_details,
                ) from cause or error
            if retry.task_next_attempt_at is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskNodeRunResult(
                canonical_sha256(
                    {
                        "execution_id": execution_id,
                        "retry_at": retry.task_next_attempt_at.isoformat(),
                    }
                ),
                execution_id,
                retry_at=retry.task_next_attempt_at,
            )
        await self._execution.fail_task(
            execution_id,
            principal=principal,
            error=error,
            attempt=attempt,
        )
        raised = TaskNodeRunError(
            error.code,
            execution_id,
            safe_details=error.safe_details,
        )
        if cause is not None:
            raise raised from cause
        raise raised from error

    async def _reconcile_custom_execution(
        self,
        node: TaskNode,
        handler: _TaskCallableAdapter,
        body: Mapping[str, JsonValue],
        dependencies: Mapping[str, TaskDependency],
        dependency_states: Mapping[str, TaskDependencyState],
        *,
        principal: Principal,
        correlation: Mapping[str, str | int],
        graph_id: str,
        execution_id: str,
    ) -> TaskNodeRunResult:
        reconcile = getattr(handler, "reconcile", None)
        if reconcile is None:
            raise TaskNodeRunError(ErrorCode.TASK_EFFECT_UNKNOWN, execution_id)
        context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            execution_id,
            body,
            dependencies,
            _custom_idempotency_key(
                graph_id,
                node,
                principal,
                dependencies,
                dependency_states,
            ),
            lambda dependency: self._read_dependency(
                dependency,
                principal=principal,
            ),
            correlation,
            artifacts=None,
            dependency_states=dependency_states,
        )
        try:
            resolution = await reconcile(context)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            safe_details: dict[str, JsonValue] = {
                "reconcile_exception_type": type(error).__name__,
            }
            if isinstance(error, AIError):
                safe_details["reconcile_error_code"] = error.code.value
                safe_details["reconcile_error_details"] = dict(error.safe_details)
            _logger.error(
                "task effect reconciliation failed: graph=%s node=%s execution=%s type=%s",
                graph_id,
                node.node_id,
                execution_id,
                type(error).__name__,
                exc_info=True,
            )
            raise TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
                safe_details=safe_details,
            ) from error
        if not isinstance(resolution, TaskEffectResolution):
            _logger.error(
                "task effect reconciliation returned an invalid value: graph=%s node=%s execution=%s type=%s",
                graph_id,
                node.node_id,
                execution_id,
                type(resolution).__name__,
            )
            raise TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
                safe_details={
                    "reconcile_result_type": type(resolution).__name__,
                },
            )
        if resolution.kind == "unknown":
            raise TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
                safe_details={"reconcile_outcome": "unknown"},
            )

        if node.effect_policy == "non_replay_safe":
            reconciled = await self._execution.resolve_task_effect(
                execution_id,
                principal=principal,
                resolution=resolution,
            )
        elif resolution.kind == "not_applied":
            reconciled = await self._execution.resume_task_not_applied(
                execution_id,
                principal=principal,
            )
        else:
            reconciled = None

        if reconciled is not None:
            if reconciled.status is ExecutionStatus.WAITING_RETRY:
                if reconciled.task_next_attempt_at is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return TaskNodeRunResult(
                    canonical_sha256(
                        {
                            "execution_id": execution_id,
                            "retry_at": reconciled.task_next_attempt_at.isoformat(),
                        }
                    ),
                    execution_id,
                    retry_at=reconciled.task_next_attempt_at,
                )
            if reconciled.status in {
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            }:
                result = await self._execution.result(
                    execution_id,
                    principal=principal,
                )
                raise TaskNodeRunError(
                    ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                    execution_id,
                    safe_details=result.safe_error_details,
                )
            if reconciled.status is ExecutionStatus.RECOVERY_REQUIRED:
                raise TaskNodeRunError(
                    ErrorCode.TASK_EFFECT_UNKNOWN,
                    execution_id,
                )
            if reconciled.status is not ExecutionStatus.SUCCEEDED:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result = await self._execution.result(
                execution_id,
                principal=principal,
            )
            return await self._complete_output(
                node,
                result.output,
                execution_id=execution_id,
                principal=principal,
                graph_id=graph_id,
            )

        output = normalize_json_value(resolution.value)
        try:
            _validate_task_output(node, output)
        except (AIError, TypeError, ValueError) as error:
            failure = (
                error
                if isinstance(error, AIError)
                else AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            )
            await self._execution.fail_task(
                execution_id,
                principal=principal,
                error=failure,
            )
            raise TaskNodeRunError(
                failure.code,
                execution_id,
                safe_details=failure.safe_details,
            ) from error
        result = await self._execution.complete_task(
            execution_id,
            principal=principal,
            output=output,
        )
        return await self._complete_output(
            node,
            result.output,
            execution_id=execution_id,
            principal=principal,
            graph_id=graph_id,
        )

    async def supply_input(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        value: JsonValue,
    ) -> TaskNodeRunResult:
        task_id, task_revision, _body = _parse_node(
            invocation.node,
            request=False,
        )
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=invocation.graph_id,
            node_id=invocation.node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            runner_result = await handler.runner.supply_input(
                invocation,
                execution_id,
                value,
            )
            return await self._complete_runner_result(invocation, runner_result)
        view = await self._execution.supply_task_input(
            execution_id,
            principal=invocation.principal,
            value=value,
        )
        if view.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = await self._execution.result(
            execution_id,
            principal=invocation.principal,
        )
        return await self._complete_output(
            invocation.node,
            result.output,
            execution_id=execution_id,
            principal=invocation.principal,
            graph_id=invocation.graph_id,
        )

    async def resolve_effect(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
        resolution: TaskEffectResolution,
    ) -> "TaskNodeRunResult | None":
        task_id, task_revision, _body = _parse_node(
            invocation.node,
            request=False,
        )
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=invocation.graph_id,
            node_id=invocation.node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            runner_result = await handler.runner.resolve_effect(
                invocation,
                execution_id,
                resolution,
            )
            if runner_result is None:
                return None
            return await self._complete_runner_result(invocation, runner_result)
        view = await self._execution.resolve_task_effect(
            execution_id,
            principal=invocation.principal,
            resolution=resolution,
        )
        if view.status is ExecutionStatus.RECOVERY_REQUIRED:
            return None
        if view.status is ExecutionStatus.WAITING_RETRY:
            if view.task_next_attempt_at is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return TaskNodeRunResult(
                canonical_sha256(
                    {
                        "execution_id": execution_id,
                        "retry_at": view.task_next_attempt_at.isoformat(),
                    }
                ),
                execution_id,
                retry_at=view.task_next_attempt_at,
            )
        if view.status is ExecutionStatus.SUCCEEDED:
            result = await self._execution.result(
                execution_id,
                principal=invocation.principal,
            )
            return await self._complete_output(
                invocation.node,
                result.output,
                execution_id=execution_id,
                principal=invocation.principal,
                graph_id=invocation.graph_id,
            )
        if view.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await self._execution.result(
                execution_id,
                principal=invocation.principal,
            )
            raise TaskNodeRunError(
                ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                execution_id,
                safe_details=result.safe_error_details,
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        node = invocation.node
        task_id, task_revision, _body = _parse_node(
            node,
            request=False,
        )
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=invocation.graph_id,
            node_id=invocation.node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            runner_result = await handler.runner.wait_bound(
                invocation,
                execution_id,
            )
            return await self._complete_runner_result(invocation, runner_result)
        view = await self._execution.inspect(
            execution_id,
            principal=invocation.principal,
        )
        if view.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = await self._execution.wait(
            execution_id,
            principal=invocation.principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        return await self._complete_output(
            node,
            result.output,
            execution_id=execution_id,
            principal=invocation.principal,
            graph_id=invocation.graph_id,
        )

    async def inspect_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult | None:
        view = await self._execution.inspect(
            execution_id,
            principal=invocation.principal,
        )
        if view.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if view.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
            )
        if view.status in {
            ExecutionStatus.STARTED,
            ExecutionStatus.WAITING_RETRY,
            ExecutionStatus.WAITING_DEFERRED,
        }:
            return None
        node = invocation.node
        task_id, task_revision, _body = _parse_node(
            node,
            request=False,
        )
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=invocation.graph_id,
            node_id=node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            return await self.wait_bound(invocation, execution_id)
        if view.status is ExecutionStatus.SUCCEEDED:
            result = await self._execution.result(
                execution_id,
                principal=invocation.principal,
            )
            return await self._complete_output(
                node,
                result.output,
                execution_id=execution_id,
                principal=invocation.principal,
                graph_id=invocation.graph_id,
            )
        if view.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            result = await self._execution.result(
                execution_id,
                principal=invocation.principal,
            )
            raise TaskNodeRunError(
                ErrorCode(result.error_code or ErrorCode.TASK_NODE_FAILED.value),
                execution_id,
                safe_details=result.safe_error_details,
            )
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        graph_id = invocation.graph_id
        node = invocation.node
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        dependency_states = invocation.dependency_states
        execution_id = invocation.execution_id
        if execution_id is None:
            graph_state = await self._task_state.graph_state(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if graph_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            node_state = next(
                (
                    value
                    for value in graph_state.node_states
                    if value.node_id == node.node_id
                ),
                None,
            )
            if node_state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution_id = node_state.execution_id
        if execution_id is None:
            return

        execution_view = await self._execution.inspect(
            execution_id,
            principal=principal,
        )
        if execution_view.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
        }:
            return
        execution_cancelled = execution_view.status is ExecutionStatus.CANCELLED
        if execution_view.status is ExecutionStatus.RECOVERY_REQUIRED:
            raise TaskNodeRunError(
                ErrorCode.TASK_EFFECT_UNKNOWN,
                execution_id,
            )

        task_id, task_revision, body = _parse_node(node, request=False)
        handler = self._handler(
            task_id,
            task_revision,
            graph_id=graph_id,
            node_id=node.node_id,
            request=False,
        )
        if isinstance(handler, _TaskRunnerAdapter):
            if execution_cancelled and isinstance(
                handler.runner,
                RuntimeAgentTaskRunner,
            ):
                return
            await handler.runner.cancel(
                replace(invocation, execution_id=execution_id),
            )
            return
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
            dependency_states=dependency_states,
            principal=principal,
            graph_id=graph_id,
        )
        context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            execution_id,
            body,
            dependencies,
            _custom_idempotency_key(
                graph_id,
                node,
                principal,
                dependencies,
                dependency_states,
            ),
            lambda dependency: self._read_dependency(
                dependency,
                principal=principal,
            ),
            correlation,
            artifacts=self._artifact_publisher(
                principal,
                graph_id,
                node.node_id,
                execution_id,
            ),
            dependency_states=dependency_states,
        )
        await handler.cancel(context)
        if not execution_cancelled:
            await self._execution.cancel_task(
                execution_id,
                principal=principal,
            )


    async def get_result_record(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
    ) -> TaskResultRecord | None:
        records = await self.get_result_records(
            graph_id,
            (node_id,),
            tenant_id=tenant_id,
        )
        return records.get(node_id)

    async def get_result_records(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]:
        return await self._task_state.get_results(
            graph_id,
            node_ids,
            tenant_id=tenant_id,
        )

    async def result_payload_size(
        self,
        record: TaskResultRecord,
        *,
        principal: Principal,
    ) -> int:
        if record.execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._execution.result_payload_size(
            record.execution_id,
            principal=principal,
        )

    async def read_execution_failure(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionResult | None:
        view = await self._execution.inspect(execution_id, principal=principal)
        if view.status not in {
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return None
        return await self._execution.result(execution_id, principal=principal)

    async def read_input_result(
        self,
        invocation: TaskNodeInvocation,
        name: str,
    ) -> JsonValue:
        reference, record = await self._input_result_source(invocation, name)
        del reference
        if record.execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self.read_result_record(record, principal=invocation.principal)

    async def read_input_result_ref(
        self,
        invocation: TaskNodeInvocation,
        name: str,
    ) -> TaskResultRef:
        reference, _record = await self._input_result_source(invocation, name)
        return reference

    async def _input_result_source(
        self,
        invocation: TaskNodeInvocation,
        name: str,
    ) -> tuple[TaskResultRef, TaskResultRecord]:
        node = invocation.node
        if name in node.dependencies:
            state = invocation.dependency_states.get(name)
            if state is not None and state.status is not TaskStatus.SUCCEEDED:
                raise AIError(ErrorCode.TASK_DEPENDENCY_FAILED)
            dependency = invocation.dependency_results.get(name)
            if dependency is None:
                if state is None or state.status is TaskStatus.SUCCEEDED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                raise AIError(ErrorCode.TASK_DEPENDENCY_FAILED)
            source_graph_id = invocation.graph_id
            source_node_id = name
            expected_digest = dependency.result_digest
        else:
            source = node.input_refs.get(name)
            if source is None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if (
                source.namespace != self._namespace
                or source.tenant_id != invocation.principal.tenant_id
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            source_graph_id = source.graph_id
            source_node_id = source.node_id
            expected_digest = source.result_digest
            header = await self._task_state.get_header(
                source_graph_id,
                tenant_id=invocation.principal.tenant_id,
            )
            if header is None:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if (
                header.kind is not ResourceKind.TASK_GRAPH
                or header.id != source_graph_id
                or header.tenant_id != invocation.principal.tenant_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._authorization.authorize(
                invocation.principal,
                AuthorizationAction.TASK_READ,
                header,
            )

        records = await self._task_state.get_results(
            source_graph_id,
            (source_node_id,),
            tenant_id=invocation.principal.tenant_id,
        )
        graph_state = await self._task_state.graph_state(
            source_graph_id,
            tenant_id=invocation.principal.tenant_id,
        )
        if graph_state is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        node_state = next(
            (
                state
                for state in graph_state.node_states
                if state.node_id == source_node_id
            ),
            None,
        )
        record = records.get(source_node_id)
        if node_state is not None and node_state.status in {
            TaskStatus.FAILED,
            TaskStatus.BLOCKED,
            TaskStatus.CANCELLED,
        }:
            raise AIError(ErrorCode.TASK_DEPENDENCY_FAILED)
        if (
            node_state is None
            or node_state.status is not TaskStatus.SUCCEEDED
            or node_state.result_digest != expected_digest
            or node_state.execution_id is None
            or record is None
            or record.result_digest != expected_digest
            or record.execution_id != node_state.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        reference = TaskResultRef(
            self._namespace,
            invocation.principal.tenant_id,
            source_graph_id,
            source_node_id,
            expected_digest,
        )
        return reference, record

    async def read_result_record(
        self,
        record: TaskResultRecord,
        *,
        principal: "Principal | None" = None,
    ) -> JsonValue:
        if principal is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = await self._execution.result(
            record.execution_id,
            principal=principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = result.output
        if canonical_sha256(output) != record.result_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return output

    async def _complete_runner_result(
        self,
        invocation: TaskNodeInvocation,
        result: TaskNodeRunResult,
    ) -> TaskNodeRunResult:
        if not isinstance(result, TaskNodeRunResult) or result.expanded_nodes:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if result.deferred or result.retry_at is not None:
            return result
        execution_id = result.execution_id
        if execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            execution_result = await self._execution.result(
                execution_id,
                principal=invocation.principal,
            )
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if (
            execution_result.status is not ExecutionStatus.SUCCEEDED
            or canonical_sha256(execution_result.output) != result.result_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._complete_output(
            invocation.node,
            execution_result.output,
            execution_id=execution_id,
            principal=invocation.principal,
            graph_id=invocation.graph_id,
        )

    async def _complete_output(
        self,
        node: TaskNode,
        output: JsonValue,
        *,
        execution_id: str | None,
        principal: Principal,
        graph_id: str,
    ) -> TaskNodeRunResult:
        if execution_id is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        normalized = normalize_json_value(output)
        _validate_task_output(node, normalized)
        digest = canonical_sha256(normalized)
        expanded_nodes = self._expand_nodes(
            node,
            normalized,
            principal=principal,
            graph_id=graph_id,
        )
        task_definitions, _expanders = self._definitions_for(graph_id)
        expanded_nodes = tuple([
            await self._materialize_node_input(
                expanded,
                principal.tenant_id,
                task_definitions,
            )
            for expanded in expanded_nodes
        ])
        return TaskNodeRunResult(
            digest,
            execution_id,
            expanded_nodes=expanded_nodes,
        )

    def _expand_nodes(
        self,
        source_node: TaskNode,
        output: JsonValue,
        *,
        principal: Principal,
        graph_id: str,
    ) -> tuple[TaskNode, ...]:
        if source_node.expander is None:
            return ()
        self._validate_expander_declaration(
            source_node.expander,
            graph_id=graph_id,
            node_id=source_node.node_id,
        )
        expander = self._resolve_expander(
            source_node.expander,
            graph_id=graph_id,
            request=False,
            node_id=source_node.node_id,
        )
        context_impl = _TaskExpansionContext(
            principal,
            graph_id,
            source_node,
            output,
        )
        context: TaskExpansionContext = context_impl
        try:
            expanded = expander.expand(context)
        except AIError as error:
            if error.code is ErrorCode.BINDING_NOT_REGISTERED:
                raise
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_failed",
            ) from error
        except (TypeError, ValueError) as error:
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_output_invalid",
            ) from error
        if not isinstance(expanded, Sequence) or isinstance(
            expanded,
            (str, bytes, bytearray),
        ):
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="expander_output_invalid",
            )
        raw_nodes = tuple(expanded)
        raw_ids = tuple(
            node.node_id for node in raw_nodes if isinstance(node, TaskNode)
        )
        if len(raw_ids) != len(raw_nodes) or len(set(raw_ids)) != len(raw_ids):
            raise _expansion_error(
                graph_id,
                source_node.node_id,
                reason="duplicate_node_id",
            )
        nodes: list[TaskNode] = []
        for raw_node in raw_nodes:
            try:
                task_id, task_revision, _body = _parse_node(
                    raw_node,
                    request=True,
                )
            except AIError as error:
                raise _expansion_error(
                    graph_id,
                    source_node.node_id,
                    reason="node_admission_invalid",
                    conflict=raw_node.node_id,
                ) from error
            self._validate_task_declaration(
                task_id,
                task_revision,
                graph_id=graph_id,
                node_id=raw_node.node_id,
            )
            if raw_node.expander is not None:
                self._validate_expander_declaration(
                    raw_node.expander,
                    graph_id=graph_id,
                    node_id=raw_node.node_id,
                )
            try:
                admitted = self.admit_node(raw_node, graph_id=graph_id)
                self._validate_durability(
                    admitted,
                    graph_id=graph_id,
                    request=False,
                )
                nodes.append(admitted)
            except AIError as error:
                if error.code is ErrorCode.BINDING_NOT_REGISTERED:
                    raise
                raise _expansion_error(
                    graph_id,
                    source_node.node_id,
                    reason="node_admission_invalid",
                    conflict=raw_node.node_id,
                ) from error
        result = tuple(sorted(nodes, key=lambda item: item.node_id))
        _logger.info(
            "task graph expansion produced nodes: graph=%s source=%s nodes=%s",
            graph_id,
            source_node.node_id,
            tuple(node.node_id for node in result),
        )
        return result

    def _resolve_expander(
        self,
        reference: TaskExpanderRef,
        *,
        graph_id: str,
        request: bool,
        node_id: str,
    ) -> TaskExpander:
        _tasks, expanders = self._definitions_for(graph_id)
        expander = expanders.get((reference.id, reference.revision))
        if expander is not None:
            return expander
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.BINDING_NOT_REGISTERED,
            safe_details={
                "graph_id": graph_id,
                "node_id": node_id,
                "role": "expander",
                "ref": f"{reference.id}@{reference.revision}",
            },
        )

    def _validate_durability(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        request: bool,
    ) -> None:
        if node.input_refs and not (
            self._task_durable and self._execution_durable
        ):
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                safe_details={
                    "phase": "task_dependency_durability",
                    "graph_id": graph_id,
                    "node_id": node.node_id,
                    "request": request,
                },
            )
        task_reference = node.task
        task_id = None if task_reference is None else task_reference.id
        task_revision = None if task_reference is None else task_reference.revision
        task_runner = self._definitions_for(graph_id)[0].get(
            (task_id, task_revision),
        ) if task_id is not None and task_revision is not None else None
        is_agent_task = task_runner is not None and isinstance(
            task_runner.runner,
            RuntimeAgentTaskRunner,
        )
        if not self._task_durable or not is_agent_task:
            return
        if self._execution_durable and self._recovery_durable:
            return
        raise AIError(
            ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
            safe_details={
                "phase": "task_durability_validation",
                "graph_id": graph_id,
                "node_id": node.node_id,
                "request": request,
            },
        )

    async def _dependencies(
        self,
        node: TaskNode,
        *,
        dependency_results: Mapping[str, TaskDependencyResult],
        dependency_states: Mapping[str, TaskDependencyState],
        principal: Principal,
        graph_id: str,
    ) -> dict[str, TaskDependency]:
        del graph_id
        if set(dependency_states) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        dependency_status = node.dependency_status(
            {
                dependency_id: state.status
                for dependency_id, state in dependency_states.items()
            }
        )
        if dependency_status is not TaskStatus.READY:
            raise AIError(ErrorCode.TASK_NOT_READY)
        required = {
            dependency_id
            for dependency_id, state in dependency_states.items()
            if state.status is TaskStatus.SUCCEEDED
        }
        if set(dependency_results) != required:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values: dict[str, TaskDependency] = {}
        for dependency_id in sorted(required):
            dependency = dependency_results[dependency_id]
            values[dependency_id] = TaskDependency(
                dependency_id,
                dependency.result_digest,
                dependency.execution_id,
            )

        grouped: dict[str, list[tuple[str, TaskResultRef]]] = {}
        for name, reference in node.input_refs.items():
            if (
                reference.namespace != self._namespace
                or reference.tenant_id != principal.tenant_id
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            grouped.setdefault(reference.graph_id, []).append((name, reference))

        for source_graph_id, entries in sorted(grouped.items()):
            node_ids = tuple(
                dict.fromkeys(
                    reference.node_id
                    for _, reference in entries
                )
            )
            records = await self._task_state.get_results(
                source_graph_id,
                node_ids,
                tenant_id=principal.tenant_id,
            )
            graph_state = await self._task_state.graph_state(
                source_graph_id,
                tenant_id=principal.tenant_id,
            )
            states = (
                {}
                if graph_state is None
                else {
                    node_state.node_id: node_state
                    for node_state in graph_state.node_states
                }
            )
            for name, reference in entries:
                record = records.get(reference.node_id)
                node_state = states.get(reference.node_id)
                if (
                    record is None
                    or node_state is None
                    or node_state.status is not TaskStatus.SUCCEEDED
                    or node_state.result_digest != reference.result_digest
                    or node_state.execution_id is None
                    or (
                        record.execution_id is not None
                        and record.execution_id != node_state.execution_id
                    )
                ):
                    raise AIError(ErrorCode.TASK_NOT_READY)
                execution_id = node_state.execution_id
                execution = await self._execution.inspect(
                    execution_id,
                    principal=principal,
                )
                if execution.status is not ExecutionStatus.SUCCEEDED:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                values[name] = TaskDependency(
                    reference.node_id,
                    reference.result_digest,
                    execution_id,
                )
        return values

    async def _read_dependency(
        self,
        dependency: TaskDependency,
        *,
        principal: Principal,
    ) -> JsonValue:
        result = await self._execution.result(
            dependency.execution_id,
            principal=principal,
        )
        if (
            result.status is not ExecutionStatus.SUCCEEDED
            or canonical_sha256(result.output) != dependency.result_digest
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result.output

    def _handler(
        self,
        task_id: str,
        task_revision: int,
        *,
        graph_id: str,
        node_id: str,
        request: bool,
    ) -> _TaskCallableAdapter | _TaskRunnerAdapter | _DeferredInputHandler:
        if (
            task_id == self._deferred_input.id
            and task_revision == self._deferred_input.revision
        ):
            return self._deferred_input
        tasks, _expanders = self._definitions_for(graph_id)
        task = tasks.get((task_id, task_revision))
        if task is not None:
            if task.function is not None:
                return _TaskCallableAdapter(task)
            if task.runner is not None:
                return _TaskRunnerAdapter(task)
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.BINDING_NOT_REGISTERED,
            safe_details={
                "graph_id": graph_id,
                "node_id": node_id,
                "role": "task",
                "ref": f"{task_id}@{task_revision}",
            },
        )


def _parse_node(
    node: TaskNode,
    *,
    request: bool,
) -> tuple[str, int, dict[str, JsonValue]]:
    reference = node.task
    if reference is None:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    return reference.id, reference.revision, dict(node.input)


def _expansion_error(
    graph_id: str,
    source_node_id: str,
    *,
    reason: str,
    conflict: str | None = None,
) -> AIError:
    details: dict[str, JsonValue] = {
        "phase": "task_graph_expansion",
        "reason": reason,
        "graph_id": graph_id,
        "source_node_id": source_node_id,
    }
    if conflict is not None:
        details["conflict"] = conflict
    return AIError(ErrorCode.TASK_DAG_INVALID, safe_details=details)


def _normalize_handler_body(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise TypeError("task handler normalize must return a mapping")
    normalized = normalize_json_value(_copy_json_mappings(value))
    if not isinstance(normalized, dict):
        raise TypeError("task handler normalize must return a mapping")
    return normalized


def _copy_json_mappings(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _copy_json_mappings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_json_mappings(item) for item in value]
    return value


def _execution_failure(result: ExecutionResult) -> TaskNodeRunError:
    if result.status not in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if result.error_code is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        code = ErrorCode(result.error_code)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return TaskNodeRunError(
        code,
        result.execution_id,
        safe_details=result.safe_error_details,
    )


def _dependency_identity_payload(
    node: TaskNode,
    dependencies: Mapping[str, TaskDependency],
    dependency_states: Mapping[str, TaskDependencyState],
) -> list[dict[str, JsonValue]]:
    if node.dependency_policy == "all_succeeded":
        return [
            {
                "node_id": dependency_id,
                "result_digest": dependencies[dependency_id].result_digest,
            }
            for dependency_id in sorted(dependencies)
        ]
    if node.dependency_policy not in {"all_terminal", "any_succeeded"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(dependency_states) != set(node.dependencies):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    node_dependencies = set(node.dependencies)
    result: list[dict[str, JsonValue]] = []
    for dependency_id in sorted(node_dependencies | set(dependencies)):
        if dependency_id not in node_dependencies:
            result.append(
                {
                    "node_id": dependency_id,
                    "result_digest": dependencies[dependency_id].result_digest,
                }
            )
            continue
        state = dependency_states[dependency_id]
        if state.status is TaskStatus.SUCCEEDED:
            dependency = dependencies.get(dependency_id)
            if (
                dependency is None
                or dependency.result_digest != state.result_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        elif dependency_id in dependencies:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result.append(
            {
                "node_id": dependency_id,
                **state.to_payload(),
            }
        )
    return result


def _task_dependency_hold_id(
    graph_id: str,
    execution_id: str,
) -> str:
    return "task-ref:" + canonical_sha256(
        {
            "graph_id": graph_id,
            "source_execution_id": execution_id,
        }
    )


def _custom_idempotency_key(
    graph_id: str,
    node: TaskNode,
    principal: Principal,
    dependencies: Mapping[str, TaskDependency],
    dependency_states: Mapping[str, TaskDependencyState],
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "graph_id": graph_id,
            "node_id": node.node_id,
            "input": node.input,
            "dependencies": _dependency_identity_payload(
                node,
                dependencies,
                dependency_states,
            ),
            "principal": principal_identity_payload(principal),
        }
    )



def _task_binding(
    node: TaskNode,
    handler: object,
    task_id: str,
    task_revision: int,
) -> TaskBindingContract:
    output_contract: Mapping[str, JsonValue] = (
        {"kind": "json"}
        if node.output_contract is None
        else dict(node.output_contract)
    )
    return TaskBindingContract(
        task_id,
        task_revision,
        node.effect_policy,
        output_contract,
        node.timeout_seconds,
        node.max_attempts,
        node.retry_delay_seconds,
        node.reconcile,
    )


def _validate_task_output(node: TaskNode, output: JsonValue) -> None:
    if node.output_contract is not None:
        _restore_output_contract(node.output_contract).validate_payload(output)


def _handler_effect_policy(handler: object, *, request: bool) -> str:
    value = getattr(handler, "effect_policy", None)
    if value not in {"none", "replay_safe", "non_replay_safe"}:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    return value


def _output_contract(
    handler: object,
    output_type: object | None,
) -> dict[str, JsonValue] | None:
    if isinstance(handler, _TaskRunnerAdapter):
        declared = handler.task.contract.get("output_contract")
        if isinstance(declared, Mapping) and declared.get("kind") == "schema":
            schema = declared.get("schema")
            if not isinstance(schema, Mapping):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            contract: dict[str, JsonValue] = {
                "mode": "structured",
                "schema": dict(schema),
            }
            _restore_output_contract(contract)
            if (
                output_type is not None
                and _output_contract(None, output_type) != contract
            ):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            return contract
    handler_output_type = getattr(handler, "output_type", None)
    output = output_type if handler_output_type is None else handler_output_type
    if output is None:
        return None
    binding = bind_output(cast("type[BaseModel]", output))
    return {
        "mode": binding.mode,
        "schema": binding.schema_definition,
    }


def _handler_has_reconcile(handler: object) -> bool:
    if isinstance(handler, _TaskRunnerAdapter):
        return handler.task.contract["reconcile"] is True
    return getattr(handler, "reconcile", None) is not None


def _restore_output_contract(
    contract: Mapping[str, JsonValue],
):
    try:
        return restore_output(contract["mode"], contract["schema"])
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


__all__ = ["RuntimeTaskNodeRunner"]
