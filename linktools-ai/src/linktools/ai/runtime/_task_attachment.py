#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task v2 attachment preparation, adoption, and execution integration."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, cast
from weakref import WeakKeyDictionary

from pydantic_ai.messages import BinaryContent, UserContent

import linktools.ai.runtime._attachment as attachment_runtime
import linktools.ai.runtime._attachment_admission as admission_runtime
import linktools.ai.runtime._planner as planner_runtime
import linktools.ai.runtime._runtime_service as runtime_service
from ..agent import AgentBindingSnapshot
from ..capability import WorkspaceAccess
from ..core import (
    ExecutionMode,
    JsonValue,
    Principal,
    ThinkingValue,
    canonical_json_bytes,
    canonical_sha256,
    idempotency_key_digest,
    normalize_json_value,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..task import DefaultTaskService, TaskGraph, TaskGraphLaunch, TaskGraphSnapshot, TaskNode
from ._attachment import DefaultAttachmentService, InputPreparer
from ._attachment_context import ManagedAdmission, admission_scope
from ._input import _decode_user_content
from .service_api import ExecutionRequest
from .state import (
    InputPrepareRecord,
    InputTarget,
    Locator,
    PathOrigin,
    PreparedInput,
    RuntimeDomain,
    input_v2_digest,
    record_key_digest,
)
from .state._attachment_codec import _entry, _entry_json, _input, _input_json
from .state._attachment_repository import AttachmentRepository, _project_owner_record
from .state._codec import _decode_domain
from .state._contracts import ExecutionStartReservation
from .state._repositories import ExecutionRepositoryImpl, replace_checked
from .state._store import StateTransaction

_AGENT_TASK_TYPE = "linktools.ai.agent"
_TASK_V2 = 2
_DRAFT_FIELDS = frozenset(
    {
        "stage",
        "binding",
        "prompt",
        "attachments",
        "mode",
        "planning",
        "thinking",
    }
)
_PREPARED_FIELDS = frozenset(
    {
        "stage",
        "binding",
        "user_prompt",
        "user_prompt_codec",
        "attachment_manifest",
        "input_digest",
        "path_origin",
        "mode",
        "planning",
        "thinking",
    }
)

_installed = False
_original_runtime_init: Any = None
_original_admit_graph: Any = None
_original_arm_graph: Any = None
_original_runner_handler: Any = None
_original_agent_normalize: Any = None
_original_agent_validate_recovery: Any = None
_original_agent_prepare_request: Any = None
_original_agent_run_node: Any = None
_original_agent_cancel_node: Any = None
_original_reserve_start: Any = None
_original_iter_object_refs: Any = None
_coordinators: "WeakKeyDictionary[object, _TaskAttachmentCoordinator]" = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class _PreparedTaskValue:
    binding: Mapping[str, JsonValue]
    user_prompt: Any
    attachment_manifest: tuple[Any, ...]
    input_digest: str
    path_origin: PathOrigin
    mode: ExecutionMode
    planning: bool
    thinking: ThinkingValue


class _GraphAttachmentRepository(AttachmentRepository):
    """Use the exact (task-admission-key, node-id) identity for graph preparation."""

    @staticmethod
    def scope(admission_key: str, node_id: str) -> str:
        return canonical_json_bytes([admission_key, node_id]).decode("utf-8")

    def owner_key(self, admission_key: str, node_id: str) -> str:
        return self._key("input_prepare", [admission_key, node_id]).hex()

    async def reserve_prepare(
        self,
        scope: str,
        idempotency_key: str,
        candidate: InputPrepareRecord,
    ) -> tuple[str, InputPrepareRecord]:
        del idempotency_key
        try:
            raw = json.loads(scope)
        except json.JSONDecodeError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(not isinstance(value, str) or not value for value in raw)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        admission_key, node_id = raw
        owner_key = self.owner_key(admission_key, node_id)
        key = bytes.fromhex(owner_key)

        async def mutate(transaction: StateTransaction) -> InputPrepareRecord:
            stored = await transaction.get_record(key)
            if stored is not None:
                if stored.kind != "input_prepare":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current = await self._decode_prepare(stored)
                if (
                    current.intent_digest != candidate.intent_digest
                    or current.path_origin != candidate.path_origin
                ):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return current
            await transaction.insert_record(
                self._stored(
                    "input_prepare",
                    [admission_key, node_id],
                    candidate,
                    state=candidate.status,
                )
            )
            return candidate

        return owner_key, await self._store.mutate(mutate)


class _TaskAttachmentCoordinator:
    def __init__(
        self,
        service: DefaultTaskService,
        task_runtime: planner_runtime.RuntimeTaskNodeRunner,
        attachment_service: DefaultAttachmentService,
        workspace: Any,
    ) -> None:
        self.service = service
        self.task_runtime = task_runtime
        self.state = attachment_service._state
        self.workspace = workspace
        self.namespace = self.state.namespace
        self.tenant_id = self.state.tenant_id
        self.repository = _GraphAttachmentRepository(
            self.state.execution.executions.state_store,
            namespace=self.namespace,
            tenant_id=self.tenant_id,
        )

    def admission_key(self, graph_id: str) -> str:
        return record_key_digest(
            self.namespace,
            self.tenant_id,
            RuntimeDomain.TASK.value,
            "task_admission",
            graph_id,
        ).hex()

    async def prepare_graph(
        self,
        graph: TaskGraph,
        *,
        principal: Principal,
        idempotency_key: str,
    ) -> TaskGraph:
        existing = await self.state.task.admissions.get(
            graph.graph_id,
            tenant_id=principal.tenant_id,
        )
        snapshot = await self.state.task.tasks.snapshot_graph(
            graph.graph_id,
            tenant_id=principal.tenant_id,
        )
        if existing is not None or snapshot is not None:
            if existing is None or snapshot is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if (
                existing.principal != principal
                or existing.operation_id != idempotency_key_digest(idempotency_key)
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return await self._reuse_existing(graph, snapshot)

        admission_key = self.admission_key(graph.graph_id)
        prepared: dict[str, TaskNode] = {}
        for node in sorted(graph.nodes, key=lambda value: value.node_id):
            if _is_v2_draft(node):
                body = _draft_body(node)
                prompt, attachments = _decode_draft_prompt(body)
                preparer = InputPreparer(
                    self.repository,
                    self.state,
                    attachment_runtime.RuntimeObjectKeyFactory(self.namespace),
                    self.workspace,
                )
                value = await preparer.prepare(
                    prompt,
                    attachments,
                    principal=principal,
                    scope=self.repository.scope(admission_key, node.node_id),
                    idempotency_key=idempotency_key,
                )
                prepared[node.node_id] = _prepared_node(node, body, value)
            else:
                prepared[node.node_id] = self.task_runtime.admit_node(node)
        return TaskGraph(
            graph.graph_id,
            tuple(prepared[node.node_id] for node in graph.nodes),
        )

    async def _reuse_existing(
        self,
        requested: TaskGraph,
        snapshot: TaskGraphSnapshot,
    ) -> TaskGraph:
        if requested.graph_id != snapshot.graph_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        existing = {node.node_id: node for node in snapshot.nodes}
        if set(existing) != {node.node_id for node in requested.nodes}:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        admission_key = self.admission_key(requested.graph_id)
        for node in requested.nodes:
            current = existing[node.node_id]
            if (
                current.dependencies != node.dependencies
                or current.budget_cost != node.budget_cost
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if not _is_v2_draft(node):
                if self.task_runtime.admit_node(node) != current:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                continue
            body = _draft_body(node)
            prepared = _prepared_task_value(current)
            if (
                dict(prepared.binding) != dict(cast(Mapping[str, JsonValue], body["binding"]))
                or prepared.mode != body["mode"]
                or prepared.planning is not body["planning"]
                or prepared.thinking != body["thinking"]
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            prompt, attachments = _decode_draft_prompt(body)
            owner_key = self.repository.owner_key(admission_key, node.node_id)
            record = await self.repository.get_prepare(
                owner_key,
                tenant_id=self.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.intent_digest != attachment_runtime.input_intent_digest(
                prompt,
                attachments,
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            if record.status not in {"READY", "ADOPTED"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.status == "READY":
                if record.input is None or not _prepared_matches(record.input, prepared):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            elif record.input is not None or record.slots:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskGraph(requested.graph_id, snapshot.nodes)

    async def confirm_launch(self, launch: TaskGraphLaunch) -> None:
        admission = await self.state.task.admissions.get(
            launch.graph.graph_id,
            tenant_id=launch.principal.tenant_id,
        )
        snapshot = await self.state.task.tasks.snapshot_graph(
            launch.graph.graph_id,
            tenant_id=launch.principal.tenant_id,
        )
        if admission is None or snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if tuple(snapshot.nodes) != tuple(launch.graph.nodes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if admission.bind(launch.graph) != launch:
                raise ValueError("task launch does not match durable admission")
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

        admission_key = self.admission_key(launch.graph.graph_id)
        target = Locator("state:task", "records", admission_key)
        for node in sorted(launch.graph.nodes, key=lambda value: value.node_id):
            if not _is_v2_prepared(node):
                continue
            prepared = _prepared_task_value(node)
            owner_key = self.repository.owner_key(admission_key, node.node_id)
            current = await self.repository.get_prepare(
                owner_key,
                tenant_id=self.tenant_id,
            )
            if current is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            expected_target = InputTarget(target, node.node_id)
            if current.status == "ADOPTED":
                if (
                    current.target != expected_target
                    or current.input is not None
                    or current.slots
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                continue
            if (
                current.status != "READY"
                or current.input is None
                or not _prepared_matches(current.input, prepared)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            adopted = InputPrepareRecord(
                1,
                current.intent_digest,
                current.path_origin,
                "ADOPTED",
                (),
                None,
                expected_target,
                None,
            )
            await self.repository.compare_and_swap_prepare(
                owner_key,
                expected=current,
                next_record=adopted,
            )


def _is_v2_draft(node: TaskNode) -> bool:
    value = node.input
    return (
        isinstance(value, Mapping)
        and value.get("type") == _AGENT_TASK_TYPE
        and value.get("version") == _TASK_V2
        and value.get("stage") == "draft"
    )


def _is_v2_prepared(node: TaskNode) -> bool:
    value = node.input
    return (
        isinstance(value, Mapping)
        and value.get("type") == _AGENT_TASK_TYPE
        and value.get("version") == _TASK_V2
        and value.get("stage") == "prepared"
    )


def _draft_body(node: TaskNode) -> dict[str, JsonValue]:
    if not _is_v2_draft(node):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    body = {key: value for key, value in node.input.items() if key not in {"type", "version"}}
    if set(body) != _DRAFT_FIELDS:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    binding = body["binding"]
    if not isinstance(binding, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    try:
        snapshot = AgentBindingSnapshot.from_payload(binding)
    except (AIError, TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    body["binding"] = cast(JsonValue, snapshot.to_payload())
    if body["stage"] != "draft":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if not isinstance(body["attachments"], list) or any(
        not isinstance(path, str) or not path for path in body["attachments"]
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if body["mode"] not in {"run", "plan"} or not isinstance(body["planning"], bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if body["mode"] == "plan" and body["planning"] is not True:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return body


def _decode_draft_prompt(body: Mapping[str, JsonValue]) -> tuple[Any, tuple[str, ...]]:
    prompt = body["prompt"]
    if not isinstance(prompt, Mapping):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    codec = prompt.get("codec")
    value = prompt.get("value")
    if codec == "text":
        if set(prompt) != {"codec", "value"} or not isinstance(value, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        user_prompt: Any = value
    elif codec == "sequence":
        if set(prompt) != {"codec", "value"} or not isinstance(value, list) or not value:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        items: list[UserContent] = []
        for raw in value:
            if not isinstance(raw, Mapping):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            kind = raw.get("kind")
            if kind == "text":
                if set(raw) != {"kind", "text"} or not isinstance(raw.get("text"), str):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                items.append(cast(str, raw["text"]))
            elif kind == "binary":
                if set(raw) != {
                    "kind",
                    "media_type",
                    "identifier",
                    "vendor_metadata",
                    "data_b64",
                }:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                data_b64 = raw["data_b64"]
                media_type = raw["media_type"]
                identifier = raw["identifier"]
                metadata = raw["vendor_metadata"]
                if (
                    not isinstance(data_b64, str)
                    or not isinstance(media_type, str)
                    or not media_type
                    or identifier is not None and not isinstance(identifier, str)
                    or metadata is not None and not isinstance(metadata, Mapping)
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                try:
                    data = base64.b64decode(data_b64, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
                if not data or base64.b64encode(data).decode("ascii") != data_b64:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                items.append(
                    BinaryContent(
                        data,
                        media_type=media_type,
                        identifier=cast(str | None, identifier),
                        vendor_metadata=(
                            None
                            if metadata is None
                            else cast(dict[str, Any], dict(metadata))
                        ),
                    )
                )
            elif kind == "native":
                if set(raw) != {"kind", "codec", "value"} or raw.get("codec") != "pydantic-user-content-v1":
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                native = raw.get("value")
                if not isinstance(native, Mapping):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                decoded = _decode_user_content(dict(native))
                if len(decoded) != 1:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                items.append(decoded[0])
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        user_prompt = tuple(items)
    else:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return user_prompt, tuple(cast(list[str], body["attachments"]))


def _prepared_node(
    node: TaskNode,
    draft: Mapping[str, JsonValue],
    prepared: PreparedInput,
) -> TaskNode:
    body: dict[str, JsonValue] = {
        "stage": "prepared",
        "binding": cast(JsonValue, dict(cast(Mapping[str, JsonValue], draft["binding"]))),
        "user_prompt": _input_json(prepared.user_prompt),
        "user_prompt_codec": prepared.user_prompt_codec,
        "attachment_manifest": [_entry_json(entry) for entry in prepared.attachment_manifest],
        "input_digest": prepared.input_digest,
        "path_origin": prepared.path_origin.to_json(),
        "mode": cast(JsonValue, draft["mode"]),
        "planning": cast(JsonValue, draft["planning"]),
        "thinking": cast(JsonValue, draft["thinking"]),
    }
    return TaskNode(
        node.node_id,
        node.dependencies,
        input={"type": _AGENT_TASK_TYPE, "version": 2, **body},
        budget_cost=node.budget_cost,
    )


def _prepared_task_value(node: TaskNode) -> _PreparedTaskValue:
    if not _is_v2_prepared(node):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    body = {key: value for key, value in node.input.items() if key not in {"type", "version"}}
    if set(body) != _PREPARED_FIELDS or body.get("stage") != "prepared":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding = body["binding"]
    manifest_raw = body["attachment_manifest"]
    if not isinstance(binding, Mapping) or not isinstance(manifest_raw, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        snapshot = AgentBindingSnapshot.from_payload(binding)
        prompt = _input(body["user_prompt"])
        manifest = tuple(_entry(value) for value in manifest_raw)
        origin = PathOrigin.from_json(body["path_origin"])
    except (AIError, TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    digest = body["input_digest"]
    if (
        body["user_prompt_codec"] != "linktools-input-v2"
        or not isinstance(digest, str)
        or input_v2_digest(prompt, manifest) != digest
        or body["mode"] not in {"run", "plan"}
        or not isinstance(body["planning"], bool)
        or body["mode"] == "plan" and body["planning"] is not True
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _PreparedTaskValue(
        cast(Mapping[str, JsonValue], snapshot.to_payload()),
        prompt,
        manifest,
        digest,
        origin,
        cast(ExecutionMode, body["mode"]),
        cast(bool, body["planning"]),
        cast(ThinkingValue, body["thinking"]),
    )


def _normalized_prepared_body(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    node = TaskNode(
        "normalize",
        (),
        input={"type": _AGENT_TASK_TYPE, "version": 2, **dict(body)},
    )
    value = _prepared_task_value(node)
    return {
        "stage": "prepared",
        "binding": cast(JsonValue, dict(value.binding)),
        "user_prompt": _input_json(value.user_prompt),
        "user_prompt_codec": "linktools-input-v2",
        "attachment_manifest": [_entry_json(entry) for entry in value.attachment_manifest],
        "input_digest": value.input_digest,
        "path_origin": value.path_origin.to_json(),
        "mode": value.mode,
        "planning": value.planning,
        "thinking": value.thinking,
    }


def _prepared_matches(value: PreparedInput, prepared: _PreparedTaskValue) -> bool:
    return (
        value.user_prompt_codec == "linktools-input-v2"
        and value.user_prompt == prepared.user_prompt
        and value.attachment_manifest == prepared.attachment_manifest
        and value.input_digest == prepared.input_digest
        and value.path_origin == prepared.path_origin
    )


def _task_target(namespace: str, tenant_id: str, graph_id: str, node_id: str) -> InputTarget:
    key = record_key_digest(
        namespace,
        tenant_id,
        RuntimeDomain.TASK.value,
        "task_admission",
        graph_id,
    ).hex()
    return InputTarget(Locator("state:task", "records", key), node_id)


def _derived_execution_prepared(
    handler: Any,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    dependencies: Mapping[str, Any],
) -> tuple[str, ExecutionRequest, PreparedInput, InputTarget]:
    prepared = _prepared_task_value(node)
    snapshot = AgentBindingSnapshot.from_payload(prepared.binding)
    binding = handler._catalog.register_binding(handler._compiler.restore(snapshot))
    prompt = prepared.user_prompt
    dependency_payload = {
        dependency_id: dependencies[dependency_id].output
        for dependency_id in sorted(node.dependencies)
    }
    if dependency_payload:
        text = (
            "\n\nUpstream task results (JSON, keyed by task id):\n"
            + planner_runtime._canonical_json(dependency_payload)
        )
        prompt = replace(prompt, parts=(*prompt.parts, attachment_runtime.InputTextPart("text", text)))
    derived_digest = input_v2_digest(prompt, prepared.attachment_manifest)
    idempotency_key = canonical_sha256(
        {
            "version": 1,
            "graph_id": graph_id,
            "node_id": node.node_id,
            "binding_digest": binding.digest,
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
    execution_prepared = PreparedInput(
        1,
        "linktools-input-v2",
        prompt,
        prepared.attachment_manifest,
        canonical_sha256(
            {
                "contract": "task-execution-v2",
                "graph_id": graph_id,
                "node_id": node.node_id,
                "node_input_digest": prepared.input_digest,
                "dependency_digests": [
                    dependencies[dependency_id].result_digest
                    for dependency_id in sorted(node.dependencies)
                ],
            }
        ),
        derived_digest,
        prepared.path_origin,
    )
    transport = attachment_runtime.prepared_user_prompt_transport(execution_prepared)
    request = ExecutionRequest(
        user_prompt=str(transport),
        user_prompt_codec=transport.codec,
        principal=principal,
        idempotency_key=idempotency_key,
        memory_scope=None,
        mode=prepared.mode,
        planning=prepared.planning,
        thinking=prepared.thinking,
    )
    return (
        binding.digest,
        request,
        execution_prepared,
        _task_target(handler._namespace, principal.tenant_id, graph_id, node.node_id),
    )


async def _reserve_start_from_adopted_source(
    self: ExecutionRepositoryImpl,
    reservation: ExecutionStartReservation,
):
    admission = admission_runtime._managed_admission.get()
    source_target = None if admission is None else getattr(admission, "source_target", None)
    if admission is None or source_target is None:
        return await _original_reserve_start(self, reservation)
    if reservation.idempotency.scope != admission.scope:
        return await _original_reserve_start(self, reservation)
    if reservation.idempotency.idempotency_key_digest != idempotency_key_digest(admission.idempotency_key):
        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    prepared = admission.prepared
    owner_key = admission_runtime._prepared_owner(prepared)
    effective = replace(
        reservation,
        execution=replace(
            reservation.execution,
            attachment_manifest=prepared.attachment_manifest,
            input_digest=prepared.input_digest,
            path_origin=prepared.path_origin,
        ),
    )
    attachments = AttachmentRepository(
        self.state_store,
        namespace=self._namespace,
        tenant_id=self._tenant_id,
    )

    async def mutate(transaction: StateTransaction):
        stored = await transaction.get_record(bytes.fromhex(owner_key))
        if stored is None or stored.kind != "input_prepare":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = await attachments._decode_prepare(stored)
        if (
            current.status != "ADOPTED"
            or current.target != source_target
            or current.input is not None
            or current.slots
            or current.path_origin != prepared.path_origin
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result = await admission_runtime._original_reserve_start(self, effective)
        admission_runtime._validate_execution_input(result.execution, prepared)
        return result

    return await self.state_store.mutate(mutate)


async def _runtime_admit_graph(
    self: runtime_service.Runtime,
    graph: TaskGraph,
    *,
    principal: Principal | None,
    idempotency_key: str,
    limits: Any,
    correlation: Any,
):
    coordinator = _coordinators.get(self.task)
    if coordinator is None or not any(_is_v2_draft(node) for node in graph.nodes):
        return await _original_admit_graph(
            self,
            graph,
            principal=principal,
            idempotency_key=idempotency_key,
            limits=limits,
            correlation=correlation,
        )
    resolved = self._resolve_principal(principal)
    prepared = await coordinator.prepare_graph(
        graph,
        principal=resolved,
        idempotency_key=idempotency_key,
    )
    return await _original_admit_graph(
        self,
        prepared,
        principal=resolved,
        idempotency_key=idempotency_key,
        limits=limits,
        correlation=correlation,
    )


async def _task_arm_graph(self: DefaultTaskService, launch: TaskGraphLaunch) -> None:
    coordinator = _coordinators.get(self)
    if coordinator is not None:
        await coordinator.confirm_launch(launch)
    await _original_arm_graph(self, launch)


def _runtime_init(self: runtime_service.Runtime, *args: Any, **kwargs: Any) -> None:
    _original_runtime_init(self, *args, **kwargs)
    attachment_service = self.attachments
    task_runtime = self._task_node_runtime
    if (
        isinstance(attachment_service, DefaultAttachmentService)
        and isinstance(task_runtime, planner_runtime.RuntimeTaskNodeRunner)
    ):
        _coordinators[self.task] = _TaskAttachmentCoordinator(
            self.task,
            task_runtime,
            attachment_service,
            self.workspace,
        )


def _runner_handler(self: planner_runtime.RuntimeTaskNodeRunner, task_type: str, task_version: int):
    if task_type == _AGENT_TASK_TYPE and task_version == 2:
        return self._agent
    return _original_runner_handler(self, task_type, task_version)


def _agent_normalize(self: Any, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    if body.get("stage") is None:
        return _original_agent_normalize(self, body)
    if body.get("stage") != "prepared":
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return _normalized_prepared_body(body)


def _agent_validate_recovery(
    self: Any,
    body: Mapping[str, JsonValue],
    *,
    graph_id: str,
    node_id: str,
) -> dict[str, JsonValue]:
    if body.get("stage") is None:
        return _original_agent_validate_recovery(
            self,
            body,
            graph_id=graph_id,
            node_id=node_id,
        )
    del graph_id, node_id
    return _normalized_prepared_body(body)


def _agent_prepare_request(
    self: Any,
    node: TaskNode,
    *,
    graph_id: str,
    principal: Principal,
    correlation: Any,
    dependencies: Mapping[str, Any],
):
    if not _is_v2_prepared(node):
        return _original_agent_prepare_request(
            self,
            node,
            graph_id=graph_id,
            principal=principal,
            correlation=correlation,
            dependencies=dependencies,
        )
    binding, request, _prepared, _target = _derived_execution_prepared(
        self,
        node,
        graph_id=graph_id,
        principal=principal,
        dependencies=dependencies,
    )
    return binding, replace(request, correlation=correlation)


async def _agent_run_node(self: Any, node: TaskNode, **kwargs: Any):
    if not _is_v2_prepared(node):
        return await _original_agent_run_node(self, node, **kwargs)
    graph_id = cast(str, kwargs["graph_id"])
    principal = cast(Principal, kwargs["principal"])
    dependencies = cast(Mapping[str, Any], kwargs["dependencies"])
    _binding, request, prepared, target = _derived_execution_prepared(
        self,
        node,
        graph_id=graph_id,
        principal=principal,
        dependencies=dependencies,
    )
    with admission_scope(
        ManagedAdmission(
            "execution.run",
            request.idempotency_key,
            prepared,
            source_target=target,
        )
    ):
        return await _original_agent_run_node(self, node, **kwargs)


async def _agent_cancel_node(self: Any, node: TaskNode, **kwargs: Any) -> None:
    if not _is_v2_prepared(node):
        await _original_agent_cancel_node(self, node, **kwargs)
        return
    graph_id = cast(str, kwargs["graph_id"])
    principal = cast(Principal, kwargs["principal"])
    dependencies = cast(Mapping[str, Any], kwargs["dependencies"])
    _binding, request, prepared, target = _derived_execution_prepared(
        self,
        node,
        graph_id=graph_id,
        principal=principal,
        dependencies=dependencies,
    )
    with admission_scope(
        ManagedAdmission(
            "execution.run",
            request.idempotency_key,
            prepared,
            source_target=target,
        )
    ):
        await _original_agent_cancel_node(self, node, **kwargs)


def _install_task_object_visitor() -> None:
    import linktools.ai.runtime.state._codec as codec_runtime

    global _original_iter_object_refs
    if _original_iter_object_refs is not None:
        return
    _original_iter_object_refs = codec_runtime._iter_runtime_object_refs

    def visit(value: object, domain: RuntimeDomain, codec: Any):
        if isinstance(value, Mapping) and value.get("$dataclass") == "task_node":
            fields = value.get("fields")
            if isinstance(fields, Mapping):
                input_value = fields.get("input")
                try:
                    decoded = _decode_domain(input_value, Any, codec, persisted=True)
                except AIError:
                    decoded = None
                if isinstance(decoded, Mapping) and decoded.get("type") == _AGENT_TASK_TYPE and decoded.get("version") == 2 and decoded.get("stage") == "prepared":
                    manifest = decoded.get("attachment_manifest")
                    if isinstance(manifest, list):
                        for raw in manifest:
                            entry = _entry(raw)
                            try:
                                yield RuntimeDomain(entry.content.domain), entry.content.object
                            except ValueError as error:
                                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        yield from _original_iter_object_refs(value, domain, codec)

    codec_runtime._iter_runtime_object_refs = visit


def install_task_attachments() -> None:
    """Install Task v2 attachment integration after base attachment admission."""
    global _installed
    global _original_admit_graph
    global _original_agent_cancel_node
    global _original_agent_normalize
    global _original_agent_prepare_request
    global _original_agent_run_node
    global _original_agent_validate_recovery
    global _original_arm_graph
    global _original_reserve_start
    global _original_runner_handler
    global _original_runtime_init
    if _installed:
        return
    _original_runtime_init = runtime_service.Runtime.__init__
    _original_admit_graph = runtime_service.Runtime._admit_graph
    _original_arm_graph = DefaultTaskService._arm_graph
    _original_runner_handler = planner_runtime.RuntimeTaskNodeRunner._handler
    _original_agent_normalize = planner_runtime._AgentTaskNodeHandler.normalize
    _original_agent_validate_recovery = planner_runtime._AgentTaskNodeHandler.validate_recovery
    _original_agent_prepare_request = planner_runtime._AgentTaskNodeHandler._prepare_request
    _original_agent_run_node = planner_runtime._AgentTaskNodeHandler.run_node
    _original_agent_cancel_node = planner_runtime._AgentTaskNodeHandler.cancel_node
    _original_reserve_start = ExecutionRepositoryImpl.reserve_start

    runtime_service.Runtime.__init__ = _runtime_init
    runtime_service.Runtime._admit_graph = _runtime_admit_graph
    DefaultTaskService._arm_graph = _task_arm_graph
    planner_runtime.RuntimeTaskNodeRunner._handler = _runner_handler
    planner_runtime._AgentTaskNodeHandler.normalize = _agent_normalize
    planner_runtime._AgentTaskNodeHandler.validate_recovery = _agent_validate_recovery
    planner_runtime._AgentTaskNodeHandler._prepare_request = _agent_prepare_request
    planner_runtime._AgentTaskNodeHandler.run_node = _agent_run_node
    planner_runtime._AgentTaskNodeHandler.cancel_node = _agent_cancel_node
    ExecutionRepositoryImpl.reserve_start = _reserve_start_from_adopted_source
    _install_task_object_visitor()
    _installed = True


__all__ = ["install_task_attachments"]
