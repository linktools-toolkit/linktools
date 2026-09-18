#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned interpretation of generic TaskNodes."""

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar, cast

from linktools.core import environ
from pydantic import BaseModel
from pydantic_ai.messages import UserContent

from ..agent import (
    AgentCatalog,
    AgentCompiler,
    AgentDefinition,
    bind_output,
    restore_output,
)
from ..capability import TaskExpander, TaskExpansionContext
from ..core import (
    ExecutionStatus,
    JsonValue,
    Principal,
    TaskStatus,
    ThinkingValue,
    canonical_json_bytes,
    canonical_sha256,
    deterministic_id,
    normalize_json_value,
    normalize_thinking,
    principal_identity_payload,
    validate_agent_id,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ..task import (
    TaskDependency,
    TaskDependencyResult,
    TaskGraph,
    TaskGraphSnapshot,
    TaskNode,
    TaskExpanderRef,
    TaskNodeContext,
    TaskNodeHandler,
    TaskNodeInvocation,
    TaskNodeRunError,
    TaskNodeRunControl,
    TaskNodeRunResult,
    TaskResultRecord,
    TaskResultRef,
)
from ._agent_task import _AgentTaskNodeHandler, _execution_failure
from ._input import CanonicalUserInput, task_prompt_draft, validate_user_input
from ._object import RuntimeObjectKeyFactory, put_runtime_object, read_runtime_object
from .service_api import ExecutionService, SessionService
from .state import ArtifactRecord, ArtifactState, RuntimeDomain

_logger = environ.get_logger("ai.runtime.planner")
AppT = TypeVar("AppT")
_TASK_TYPE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_RESERVED_EXPANDER_ID_PREFIX = "linktools.ai."
_DEFERRED_INPUT_TYPE = "linktools.ai.input"
_DEFERRED_INPUT_VERSION = 1


class _DeferredInputHandler:
    type = _DEFERRED_INPUT_TYPE
    version = _DEFERRED_INPUT_VERSION
    effect = "none"

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
        state: ArtifactState,
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
        data = bytearray()
        async for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            data.extend(chunk)
            if len(data) > expected_size:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        value = bytes(data)
        digest = hashlib.sha256(value).hexdigest()
        if len(value) != expected_size or digest != expected_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        artifact_id = deterministic_id(
            "task-artifact",
            self._execution_id,
            name,
            expected_digest,
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
        object_ref = await put_runtime_object(
            self._object_store,
            self._object_key_factory,
            RuntimeDomain.ARTIFACT,
            self._principal.tenant_id,
            value,
        )
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
    async def snapshot_graph(
        self,
        graph_id: str,
        *,
        tenant_id: str,
    ) -> TaskGraphSnapshot | None: ...

    async def get_results(
        self,
        graph_id: str,
        node_ids: tuple[str, ...],
        *,
        tenant_id: str,
    ) -> Mapping[str, TaskResultRecord]: ...


class _TaskExpansionContext:
    __slots__ = (
        "_build_agent_task",
        "_generated_agent_tasks",
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
        *,
        build_agent_task: Callable[..., TaskNode],
    ) -> None:
        self._principal = principal
        self._graph_id = graph_id
        self._source_node = source_node
        self._output = output
        self._build_agent_task = build_agent_task
        self._generated_agent_tasks: dict[str, TaskNode] = {}

    def agent_task(
        self,
        agent_id: str,
        node_id: str,
        user_prompt: str | Sequence[UserContent],
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
        files: Sequence[str] = (),
        session_id: str | None = None,
        memory_scope: str | None = None,
    ) -> TaskNode:
        node = self._build_agent_task(
            agent_id,
            node_id,
            validate_user_input(user_prompt),
            dependencies=dependencies,
            budget_cost=budget_cost,
            output=output,
            planning=planning,
            thinking=thinking,
            expander=expander,
            files=files,
            session_id=session_id,
            memory_scope=memory_scope,
        )
        self._generated_agent_tasks[node.node_id] = node
        return node

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
        return self._output

    def _generated_agent_task(self, node_id: str) -> TaskNode | None:
        return self._generated_agent_tasks.get(node_id)


class RuntimeTaskNodeRunner(Generic[AppT]):
    """Interpret admitted TaskNodes using the frozen Runtime handler map."""

    def __init__(
        self,
        execution: ExecutionService,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        *,
        session: SessionService | None = None,
        namespace: str,
        app: AppT,
        task_state: _TaskStateReader,
        task_objects: ObjectStore,
        artifact_state: ArtifactState | None = None,
        artifact_objects: ObjectStore | None = None,
        object_key_factory: RuntimeObjectKeyFactory,
        payload_policy: PayloadPolicy,
        handlers: Sequence[TaskNodeHandler[AppT]] = (),
        expanders: Sequence[TaskExpander] = (),
        release_dependency_hold: Callable[..., Awaitable[None]] | None = None,
        task_durable: bool = False,
        execution_durable: bool = True,
        recovery_durable: bool = True,
    ) -> None:
        self._app = app
        self._namespace = namespace
        self._catalog = catalog
        self._compiler = compiler
        self._execution = execution
        self._task_state = task_state
        self._task_objects = task_objects
        self._artifact_state = artifact_state
        self._artifact_objects = artifact_objects
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy
        self._agent = _AgentTaskNodeHandler(
            execution,
            catalog,
            compiler,
            session=session,
            release_dependency_hold=release_dependency_hold,
        )
        self._deferred_input = _DeferredInputHandler()
        values: dict[tuple[str, int], TaskNodeHandler[AppT]] = {}
        for handler in handlers:
            key = _external_handler_identity(handler)
            if key in values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            values[key] = handler
        self._handlers = MappingProxyType(values)
        expander_values: dict[tuple[str, int], TaskExpander] = {}
        for expander in expanders:
            key = _external_expander_identity(expander)
            if key in expander_values:
                raise AIError(ErrorCode.CAPABILITY_CONFLICT)
            expander_values[key] = expander
        self._expanders = MappingProxyType(expander_values)
        self._task_durable = task_durable
        self._execution_durable = execution_durable
        self._recovery_durable = recovery_durable
        self._materialization_tasks: set[asyncio.Task[StoredPayload]] = set()
        self._background_failure: AIError | None = None

    @property
    def pending_background_tasks(self) -> tuple[asyncio.Task[object], ...]:
        materializations = tuple(
            cast("asyncio.Task[object]", task)
            for task in self._materialization_tasks
            if not task.done()
        )
        return (*self._agent.pending_background_tasks, *materializations)

    @property
    def pending_cancelled_tasks(self) -> tuple[asyncio.Task[object], ...]:
        return self._agent.pending_cancelled_tasks

    @property
    def background_failure(self) -> AIError | None:
        if self._background_failure is not None:
            return AIError(
                self._background_failure.code,
                category=self._background_failure.category,
                retryable=self._background_failure.retryable,
                operation_id=self._background_failure.operation_id,
                safe_details=dict(self._background_failure.safe_details),
                diagnostics=self._background_failure.diagnostics,
            )
        return self._agent.background_failure

    def admit_node(self, node: TaskNode) -> TaskNode:
        task_type, task_version, body = _parse_node(node, request=True)
        handler = self._handler(task_type, task_version, request=True)
        if node.expander is not None:
            self._resolve_expander(node.expander, request=True)
        try:
            normalized = handler.normalize(body)
            canonical_body = _normalize_handler_body(normalized)
        except (AIError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        return TaskNode(
            node.node_id,
            node.dependencies,
            input={
                "type": task_type,
                "version": task_version,
                **canonical_body,
            },
            budget_cost=node.budget_cost,
            expander=node.expander,
            input_refs=node.input_refs,
            timeout_seconds=node.timeout_seconds,
            max_attempts=node.max_attempts,
            retry_delay_seconds=node.retry_delay_seconds,
            output_schema=node.output_schema,
            output_contract=_output_contract(handler, node.output_schema),
            effect=_handler_effect(handler),
        )

    def validate_request(self, graph: TaskGraph) -> None:
        for node in graph.nodes:
            canonical = self.admit_node(node)
            if canonical.input != node.input:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            self._validate_durability(node, graph_id=graph.graph_id, request=True)

    def validate_recovery(self, snapshot: TaskGraphSnapshot) -> None:
        for node, state in zip(snapshot.nodes, snapshot.node_states, strict=True):
            if state.status in {
                TaskStatus.SUCCEEDED,
                TaskStatus.FAILED,
                TaskStatus.BLOCKED,
                TaskStatus.CANCELLED,
            }:
                continue
            self._validate_durability(
                node,
                graph_id=snapshot.graph_id,
                request=False,
            )
            task_type, task_version, body = _parse_node(node, request=False)
            try:
                handler = self._handler(task_type, task_version, request=False)
            except AIError as error:
                if error.code is not ErrorCode.CAPABILITY_REQUIRED_MISSING:
                    raise
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "task",
                        "task_type": task_type,
                        "task_version": task_version,
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                    },
                ) from error
            if node.expander is not None:
                self._resolve_expander(node.expander, request=False)
            if node.effect != _handler_effect(handler):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                        "reason": "task_effect_changed",
                    },
                )
            handler_output = getattr(handler, "output", None)
            if handler_output is not None and node.output_contract != _output_contract(
                handler,
                None,
            ):
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                        "reason": "task_output_contract_changed",
                    },
                )
            if handler is self._agent:
                canonical_body = self._agent.validate_recovery(
                    body,
                    graph_id=snapshot.graph_id,
                    node_id=node.node_id,
                )
            else:
                try:
                    canonical_body = _normalize_handler_body(handler.normalize(body))
                except (AIError, TypeError, ValueError) as error:
                    raise AIError(
                        ErrorCode.STORAGE_INTEGRITY_ERROR,
                        safe_details={
                            "graph_id": snapshot.graph_id,
                            "node_id": node.node_id,
                            "task_type": task_type,
                            "task_version": task_version,
                        },
                    ) from error
            canonical = TaskNode(
                node.node_id,
                node.dependencies,
                input={
                    "type": task_type,
                    "version": task_version,
                    **canonical_body,
                },
                budget_cost=node.budget_cost,
                expander=node.expander,
                input_refs=node.input_refs,
                timeout_seconds=node.timeout_seconds,
                max_attempts=node.max_attempts,
                retry_delay_seconds=node.retry_delay_seconds,
                output_schema=node.output_schema,
                output_contract=node.output_contract,
                effect=node.effect,
            )
            if canonical.input != node.input:
                raise AIError(
                    ErrorCode.STORAGE_INTEGRITY_ERROR,
                    safe_details={
                        "graph_id": snapshot.graph_id,
                        "node_id": node.node_id,
                        "task_type": task_type,
                        "task_version": task_version,
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
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
            tenant_id=principal.tenant_id,
            graph_id=graph_id,
        )
        execution_id: str | None = None
        if handler is self._deferred_input:
            wait_id = _wait_execution_id(
                graph_id,
                node.node_id,
                principal.tenant_id,
            )
            await control.handoff_execution(wait_id)
            return TaskNodeRunResult(
                canonical_sha256({"wait_id": wait_id}),
                wait_id,
                deferred=True,
            )
        if handler is self._agent:
            output, execution_id = await self._agent.run_node(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
                dependencies=dependencies,
                control=control,
            )
        else:
            execution_id = _task_execution_id(
                graph_id,
                node.node_id,
                principal.tenant_id,
            )
            idempotency_key = _custom_idempotency_key(
                graph_id,
                node,
                principal,
                dependencies,
            )
            task_context = TaskNodeContext(
                self._app,
                principal,
                graph_id,
                node.node_id,
                execution_id,
                body,
                dependencies,
                idempotency_key,
                correlation,
                artifacts=self._artifact_publisher(
                    principal,
                    graph_id,
                    node.node_id,
                    execution_id,
                ),
            )
            raw_output = await self._run_custom_handler(
                handler,
                task_context,
                node=node,
                execution_id=execution_id,
            )
            try:
                output = normalize_json_value(raw_output)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        return await self._complete_output(
            node,
            output,
            execution_id=execution_id,
            principal=principal,
            graph_id=graph_id,
        )

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

    async def _run_custom_handler(
        self,
        handler: TaskNodeHandler[AppT],
        context: TaskNodeContext[AppT],
        *,
        node: TaskNode,
        execution_id: str,
    ) -> JsonValue:
        attempts = 0
        effect = node.effect
        while True:
            attempts += 1
            try:
                if node.timeout_seconds is None:
                    return await handler.run(context)
                return await asyncio.wait_for(
                    handler.run(context),
                    timeout=node.timeout_seconds,
                )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError as error:
                if effect == "non_replay_safe":
                    raise TaskNodeRunError(
                        ErrorCode.TASK_EFFECT_UNKNOWN,
                        execution_id,
                    ) from error
                if attempts >= node.max_attempts:
                    raise AIError(ErrorCode.EXECUTION_WAIT_TIMEOUT) from error
            except AIError as error:
                retryable = bool(error.retryable)
                if (
                    attempts >= node.max_attempts
                    or not retryable
                    or effect == "non_replay_safe"
                ):
                    if effect == "non_replay_safe":
                        raise TaskNodeRunError(
                            ErrorCode.TASK_EFFECT_UNKNOWN,
                            execution_id,
                        ) from error
                    raise
            except Exception as error:  # noqa: BLE001
                if effect == "non_replay_safe":
                    raise TaskNodeRunError(
                        ErrorCode.TASK_EFFECT_UNKNOWN,
                        execution_id,
                    ) from error
                raise AIError(ErrorCode.TASK_NODE_FAILED) from error
            if node.retry_delay_seconds:
                await asyncio.sleep(node.retry_delay_seconds)

    async def wait_bound(
        self,
        invocation: TaskNodeInvocation,
        execution_id: str,
    ) -> TaskNodeRunResult:
        node = invocation.node
        task_type, task_version, _body = _parse_node(node, request=False)
        if (task_type, task_version) != (self._agent.type, self._agent.version):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        view = await self._execution.inspect(
            execution_id,
            principal=invocation.principal,
        )
        if view.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if view.status is ExecutionStatus.RECOVERY_REQUIRED:
            await self._execution.recover(
                execution_id,
                principal=invocation.principal,
            )
        result = await self._execution.wait(
            execution_id,
            principal=invocation.principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._complete_output(
            node,
            result.output,
            execution_id=execution_id,
            principal=invocation.principal,
            graph_id=invocation.graph_id,
        )

    def build_agent_task(
        self,
        agent_digest: str,
        node_id: str,
        user_prompt: CanonicalUserInput,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
        files: Sequence[str] = (),
        session_id: str | None = None,
        memory_scope: str | None = None,
    ) -> TaskNode:
        definition = self._root_definition(agent_digest)
        if planning is not None and not isinstance(planning, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        resolved_planning = (
            definition.spec.planning if planning is None else planning
        )
        resolved_thinking = (
            definition.spec.thinking
            if thinking is None
            else normalize_thinking(thinking)
        )
        binding = self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": self._agent.type,
                "version": self._agent.version,
                "binding": binding.snapshot.to_payload(),
                "user_prompt": task_prompt_draft(user_prompt),
                "mode": "run",
                "planning": resolved_planning,
                "thinking": resolved_thinking,
                "files": list(files),
                "session_id": session_id,
                "memory_scope": memory_scope,
            },
            budget_cost=budget_cost,
            expander=expander,
            output_schema=output,
        )

    def _build_agent_task_by_id(
        self,
        agent_id: str,
        node_id: str,
        user_prompt: CanonicalUserInput,
        *,
        dependencies: tuple[str, ...] = (),
        budget_cost: int = 1,
        output: type[BaseModel] | None = None,
        planning: bool | None = None,
        thinking: ThinkingValue | None = None,
        expander: TaskExpanderRef | None = None,
        files: Sequence[str] = (),
        session_id: str | None = None,
        memory_scope: str | None = None,
    ) -> TaskNode:
        validate_agent_id(agent_id)
        try:
            definition = self._catalog.root_definition(agent_id)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={"kind": "agent", "agent_id": agent_id},
            ) from error
        return self.build_agent_task(
            definition.digest,
            node_id,
            user_prompt,
            dependencies=dependencies,
            budget_cost=budget_cost,
            output=output,
            planning=planning,
            thinking=thinking,
            expander=expander,
            files=files,
            session_id=session_id,
            memory_scope=memory_scope,
        )

    async def cancel(self, invocation: TaskNodeInvocation) -> None:
        node = invocation.node
        graph_id = invocation.graph_id
        principal = invocation.principal
        correlation = invocation.correlation
        dependency_results = invocation.dependency_results
        task_type, task_version, body = _parse_node(node, request=False)
        handler = self._handler(task_type, task_version, request=False)
        dependencies = await self._dependencies(
            node,
            dependency_results=dependency_results,
            tenant_id=principal.tenant_id,
            graph_id=graph_id,
        )
        if handler is self._agent:
            snapshot = await self._task_state.snapshot_graph(
                graph_id,
                tenant_id=principal.tenant_id,
            )
            if snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            state = next(
                (
                    value
                    for value in snapshot.node_states
                    if value.node_id == node.node_id
                ),
                None,
            )
            if state is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await self._agent.cancel_node(
                node,
                graph_id=graph_id,
                principal=principal,
                correlation=correlation,
                dependencies=dependencies,
                durable_execution_id=state.execution_id,
            )
            return
        task_context = TaskNodeContext(
            self._app,
            principal,
            graph_id,
            node.node_id,
            _task_execution_id(graph_id, node.node_id, principal.tenant_id),
            body,
            dependencies,
            _custom_idempotency_key(graph_id, node, principal, dependencies),
            correlation,
            artifacts=self._artifact_publisher(
                principal,
                graph_id,
                node.node_id,
                _task_execution_id(graph_id, node.node_id, principal.tenant_id),
            ),
        )
        await handler.cancel(task_context)

    async def get_result_record(
        self,
        graph_id: str,
        node_id: str,
        *,
        tenant_id: str,
    ) -> TaskResultRecord | None:
        records = await self._task_state.get_results(
            graph_id,
            (node_id,),
            tenant_id=tenant_id,
        )
        return records.get(node_id)

    async def read_result_record(self, record: TaskResultRecord) -> JsonValue:
        output = await self._read_payload(record.payload)
        if canonical_sha256(output) != record.result_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return output

    async def _complete_output(
        self,
        node: TaskNode,
        output: JsonValue,
        *,
        execution_id: str | None,
        principal: Principal,
        graph_id: str,
    ) -> TaskNodeRunResult:
        normalized = normalize_json_value(output)
        if node.output_contract is not None:
            _restore_output_contract(node.output_contract).validate_payload(normalized)
        elif isinstance(node.output_schema, type) and issubclass(
            node.output_schema,
            BaseModel,
        ):
            bind_output(node.output_schema).validate_payload(normalized)
        digest = canonical_sha256(normalized)
        payload = await self._materialize_result(
            normalized,
            tenant_id=principal.tenant_id,
            graph_id=graph_id,
            node_id=node.node_id,
        )
        if payload.digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        expanded_nodes = self._expand_nodes(
            node,
            normalized,
            principal=principal,
            graph_id=graph_id,
        )
        return TaskNodeRunResult(
            digest,
            execution_id,
            payload,
            expanded_nodes,
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
        expander = self._resolve_expander(source_node.expander, request=False)
        context_impl = _TaskExpansionContext(
            principal,
            graph_id,
            source_node,
            output,
            build_agent_task=self._build_agent_task_by_id,
        )
        context: TaskExpansionContext = context_impl
        try:
            expanded = expander.expand(context)
        except AIError as error:
            if error.code is ErrorCode.CAPABILITY_REQUIRED_MISSING:
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
            task_type = raw_node.input.get("type")
            if task_type == self._agent.type:
                generated = context_impl._generated_agent_task(raw_node.node_id)
                if generated != raw_node:
                    raise _expansion_error(
                        graph_id,
                        source_node.node_id,
                        reason="agent_task_builder_required",
                        conflict=raw_node.node_id,
                    )
            try:
                admitted = self.admit_node(raw_node)
                self._validate_durability(
                    admitted,
                    graph_id=graph_id,
                    request=False,
                )
                nodes.append(admitted)
            except AIError as error:
                if error.code is ErrorCode.CAPABILITY_REQUIRED_MISSING:
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
        request: bool,
    ) -> TaskExpander:
        expander = self._expanders.get((reference.id, reference.version))
        if expander is not None:
            return expander
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={
                "kind": "task_expander",
                "expander_id": reference.id,
                "expander_version": reference.version,
            },
        )

    def _validate_durability(
        self,
        node: TaskNode,
        *,
        graph_id: str,
        request: bool,
    ) -> None:
        task_type = node.input.get("type")
        if not self._task_durable or task_type != self._agent.type:
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

    def _root_definition(self, agent_digest: str) -> AgentDefinition:
        for agent_id in self._catalog.root_ids:
            definition = self._catalog.root_definition(agent_id)
            if definition.digest == agent_digest:
                return definition
        raise AIError(
            ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"kind": "agent", "agent_digest": agent_digest},
        )

    async def _dependencies(
        self,
        node: TaskNode,
        *,
        dependency_results: Mapping[str, TaskDependencyResult],
        tenant_id: str,
        graph_id: str,
    ) -> dict[str, TaskDependency]:
        if set(dependency_results) != set(node.dependencies):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        values: dict[str, TaskDependency] = {}
        for dependency_id in sorted(node.dependencies):
            dependency = dependency_results[dependency_id]
            if dependency.result_payload is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            output = await self._read_payload(dependency.result_payload)
            execution_id = dependency.execution_id
            if execution_id == _task_execution_id(
                graph_id,
                dependency_id,
                tenant_id,
            ):
                execution_id = None
            values[dependency_id] = TaskDependency(
                dependency_id,
                output,
                dependency.result_digest,
                execution_id,
            )
        if node.input_refs:
            grouped: dict[str, list[tuple[str, TaskResultRef]]] = {}
            for name, reference in node.input_refs.items():
                if (
                    reference.namespace != self._namespace
                    or reference.tenant_id != tenant_id
                ):
                    raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                grouped.setdefault(reference.graph_id, []).append((name, reference))
            for graph_id, entries in grouped.items():
                records = await self._task_state.get_results(
                    graph_id,
                    tuple(reference.node_id for _, reference in entries),
                    tenant_id=tenant_id,
                )
                snapshot = await self._task_state.snapshot_graph(
                    graph_id,
                    tenant_id=tenant_id,
                )
                states = (
                    {} if snapshot is None else {
                        state.node_id: state for state in snapshot.node_states
                    }
                )
                for name, reference in entries:
                    record = records.get(reference.node_id)
                    state = states.get(reference.node_id)
                    if (
                        record is None
                        or state is None
                        or state.status is not TaskStatus.SUCCEEDED
                        or state.result_digest != reference.result_digest
                    ):
                        raise AIError(ErrorCode.TASK_NOT_READY)
                    output = await self._read_payload(record.payload)
                    execution_id = state.execution_id
                    if execution_id == _task_execution_id(
                        graph_id,
                        reference.node_id,
                        tenant_id,
                    ):
                        execution_id = None
                    values[name] = TaskDependency(
                        reference.node_id,
                        output,
                        reference.result_digest,
                        execution_id,
                    )
        return values

    async def _materialize_result(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
        graph_id: str,
        node_id: str,
    ) -> StoredPayload:
        task = asyncio.create_task(
            self._materialize_result_inner(output, tenant_id=tenant_id),
            name=f"task-result-materialize-{graph_id}-{node_id}",
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._consume_materialization(task, graph_id=graph_id, node_id=node_id)
            else:
                self._materialization_tasks.add(task)

                def consume(done: asyncio.Task[StoredPayload]) -> None:
                    try:
                        self._consume_materialization(
                            done,
                            graph_id=graph_id,
                            node_id=node_id,
                        )
                    finally:
                        self._materialization_tasks.discard(done)

                task.add_done_callback(consume)
            raise

    def _consume_materialization(
        self,
        task: asyncio.Task[StoredPayload],
        *,
        graph_id: str,
        node_id: str,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as error:  # noqa: BLE001
            if self._background_failure is not None:
                return
            details = dict(error.safe_details) if isinstance(error, AIError) else {}
            details.setdefault("phase", "task_result_materialize")
            details.setdefault("graph_id", graph_id)
            details.setdefault("node_id", node_id)
            self._background_failure = AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                safe_details=details,
            )

    async def _materialize_result_inner(
        self,
        output: JsonValue,
        *,
        tenant_id: str,
    ) -> StoredPayload:
        inline = StoredPayload.inline_json(output)
        if payload_fits_inline(inline, self._payload_policy):
            return inline
        data = canonical_json_bytes(output)
        reference = await put_runtime_object(
            self._task_objects,
            self._object_key_factory,
            RuntimeDomain.TASK,
            tenant_id,
            data,
        )
        return StoredPayload.object(reference)

    async def _read_payload(self, payload: StoredPayload) -> JsonValue:
        try:
            if payload.kind == "inline":
                value = payload.decode()
            else:
                if payload.ref is None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                data = await read_runtime_object(self._task_objects, payload.ref)
                value = json.loads(data.decode("utf-8"))
            normalized = normalize_json_value(value)
        except AIError:
            raise
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if canonical_sha256(normalized) != payload.digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return normalized

    def _handler(
        self,
        task_type: str,
        task_version: int,
        *,
        request: bool,
    ) -> TaskNodeHandler[AppT] | _AgentTaskNodeHandler:
        if task_type == self._agent.type and task_version == self._agent.version:
            return self._agent
        if (
            task_type == self._deferred_input.type
            and task_version == self._deferred_input.version
        ):
            return self._deferred_input
        handler = self._handlers.get((task_type, task_version))
        if handler is not None:
            return handler
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.CAPABILITY_REQUIRED_MISSING,
            safe_details={"task_type": task_type, "task_version": task_version},
        )


def _parse_node(
    node: TaskNode,
    *,
    request: bool,
) -> tuple[str, int, dict[str, JsonValue]]:
    payload = node.input
    task_type = payload.get("type")
    task_version = payload.get("version")
    if (
        not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID
            if request
            else ErrorCode.STORAGE_INTEGRITY_ERROR
        )
    body = {
        key: value for key, value in payload.items() if key not in {"type", "version"}
    }
    return task_type, task_version, body


def _external_handler_identity(handler: TaskNodeHandler[object]) -> tuple[str, int]:
    task_type = handler.type
    task_version = handler.version
    if (
        not isinstance(task_type, str)
        or _TASK_TYPE.fullmatch(task_type) is None
        or task_type.startswith("linktools.ai.")
        or not isinstance(task_version, int)
        or isinstance(task_version, bool)
        or task_version < 1
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return task_type, task_version


def _external_expander_identity(expander: TaskExpander) -> tuple[str, int]:
    try:
        reference = TaskExpanderRef(expander.id, expander.version)
    except (TypeError, ValueError, AttributeError) as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error
    if reference.id.startswith(_RESERVED_EXPANDER_ID_PREFIX):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return reference.id, reference.version


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
    normalized = normalize_json_value(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError("task handler normalize must return a mapping")
    if "type" in normalized or "version" in normalized:
        raise ValueError("task handler normalize returned reserved fields")
    return normalized


def _custom_idempotency_key(
    graph_id: str,
    node: TaskNode,
    principal: Principal,
    dependencies: Mapping[str, TaskDependency],
) -> str:
    return canonical_sha256(
        {
            "version": 1,
            "graph_id": graph_id,
            "node_id": node.node_id,
            "input": node.input,
            "dependencies": [
                {
                    "node_id": dependency_id,
                    "result_digest": dependencies[dependency_id].result_digest,
                }
                for dependency_id in sorted(node.dependencies)
            ],
            "principal": principal_identity_payload(principal),
        }
    )


def _task_execution_id(graph_id: str, node_id: str, tenant_id: str) -> str:
    return deterministic_id("task-execution", graph_id, node_id, tenant_id)


def _wait_execution_id(graph_id: str, node_id: str, tenant_id: str) -> str:
    return f"wait:{deterministic_id('task-wait', graph_id, node_id, tenant_id)}"


def _handler_effect(handler: object) -> str:
    value = getattr(handler, "effect", "none")
    return value if value in {"none", "replay_safe", "non_replay_safe"} else "none"


def _output_contract(
    handler: object,
    output_schema: object | None,
) -> dict[str, JsonValue] | None:
    output = getattr(handler, "output", None) or output_schema
    if output is None:
        return None
    binding = bind_output(cast("type[BaseModel]", output))
    return {
        "mode": binding.mode,
        "schema": binding.schema_definition,
    }


def _restore_output_contract(
    contract: Mapping[str, JsonValue],
):
    try:
        return restore_output(contract["mode"], contract["schema"])
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


__all__ = ["RuntimeTaskNodeRunner"]
