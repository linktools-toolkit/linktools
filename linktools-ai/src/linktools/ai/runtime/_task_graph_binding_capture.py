#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable Task and expander declarations admitted with a TaskGraph."""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from ..agent import AgentBindingContract
from ..core import (
    ImmutableJsonMapping,
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, read_object
from ..task import Task, TaskExpander, TaskGraph, TaskGraphAdmission, TaskNode
from ._runtime_identity import task_graph_binding_capture_key
from .state._task_graph_binding_capture import (
    TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION,
    TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS,
    read_task_graph_binding_capture_declarations,
    task_declaration_identity,
    task_expander_declaration_identity,
)

_KIND = "task-definition-capture"
_VERSION = TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_BUILTIN_TASKS: dict[tuple[str, int], dict[str, JsonValue]] = {
    ("linktools.ai.input", 1): {
        "version": 1,
        "id": "linktools.ai.input",
        "revision": 1,
        "type": "input",
        "effect_policy": "none",
        "output_contract": {"kind": "json"},
        "reconcile": False,
    },
}
def builtin_task_declaration(
    task_id: str,
    revision: int,
) -> Mapping[str, JsonValue] | None:
    declaration = _BUILTIN_TASKS.get((task_id, revision))
    return None if declaration is None else ImmutableJsonMapping(declaration)


@dataclass(frozen=True, slots=True)
class TaskGraphBindingCapture:
    tasks: Mapping[tuple[str, int], Mapping[str, JsonValue]]
    expanders: Mapping[tuple[str, int], Mapping[str, JsonValue]]

    def __post_init__(self) -> None:
        tasks = _freeze_declarations(self.tasks, task_declaration_identity)
        expanders = _freeze_declarations(
            self.expanders,
            task_expander_declaration_identity,
        )
        object.__setattr__(self, "tasks", MappingProxyType(tasks))
        object.__setattr__(self, "expanders", MappingProxyType(expanders))


class TaskGraphBindingCaptureStore:
    def __init__(self, namespace: str, object_store: ObjectStore) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace is required")
        self._namespace = namespace
        self._objects = object_store

    async def capture(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
        *,
        tasks: Sequence[Task[object]] = (),
        expanders: Sequence[TaskExpander] = (),
    ) -> TaskGraphBindingCapture:
        key = self._key(admission)
        task_declarations = _task_declarations(tasks)
        expander_declarations = _expander_declarations(expanders)
        dynamic = any(node.expander is not None for node in graph.nodes)
        if dynamic:
            captured_tasks = task_declarations
            captured_expanders = expander_declarations
        else:
            required_tasks = _required_task_identities(graph)
            required_expanders = _required_expander_identities(graph)
            captured_tasks = {
                identity: task_declarations[identity]
                for identity in required_tasks
                if identity in task_declarations
            }
            captured_expanders = {
                identity: expander_declarations[identity]
                for identity in required_expanders
                if identity in expander_declarations
            }
        _validate_request_references(
            graph,
            task_declarations,
            expander_declarations,
        )

        existing = await self._objects.stat(key)
        if existing is not None:
            capture = await self._read(
                ObjectRef(
                    self._objects.store_id,
                    key,
                    existing.digest,
                    existing.size,
                ),
                admission,
            )
            _validate_required_declarations(
                capture,
                graph,
                task_declarations,
                expander_declarations,
            )
            return capture

        manifest: dict[str, JsonValue] = {
            "kind": _KIND,
            "format_version": _VERSION,
            "namespace": self._namespace,
            "tenant_id": admission.principal.tenant_id,
            "graph_id": admission.graph_id,
            "request_digest": admission.initial_request_digest,
            "tasks": [
                dict(value)
                for _identity, value in sorted(captured_tasks.items())
            ],
            "expanders": [
                dict(value)
                for _identity, value in sorted(captured_expanders.items())
            ],
        }
        payload = canonical_json_bytes(manifest)
        digest = canonical_sha256(manifest)

        async def chunks():
            yield payload

        try:
            stat = await self._objects.put(
                key,
                chunks(),
                expected_size=len(payload),
                expected_digest=digest,
            )
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_CONFLICT:
                raise
            current = await self._objects.stat(key)
            if current is None:
                raise
            capture = await self._read(
                ObjectRef(
                    self._objects.store_id,
                    key,
                    current.digest,
                    current.size,
                ),
                admission,
            )
        else:
            capture = await self._read(
                ObjectRef(
                    self._objects.store_id,
                    stat.key,
                    stat.digest,
                    stat.size,
                ),
                admission,
            )
        _validate_required_declarations(
            capture,
            graph,
            task_declarations,
            expander_declarations,
        )
        return capture

    async def load(
        self,
        admission: TaskGraphAdmission,
    ) -> TaskGraphBindingCapture:
        key = self._key(admission)
        stat = await self._objects.stat(key)
        if stat is None:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={"graph_id": admission.graph_id},
            )
        return await self._read(
            ObjectRef(
                self._objects.store_id,
                key,
                stat.digest,
                stat.size,
            ),
            admission,
        )

    async def _read(
        self,
        ref: ObjectRef,
        admission: TaskGraphAdmission,
    ) -> TaskGraphBindingCapture:
        payload = await read_object(
            self._objects,
            ref.key,
            expected_digest=ref.digest,
            expected_size=ref.size,
        )
        try:
            manifest = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if not isinstance(manifest, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        format_version = manifest.get("format_version")
        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, int)
            or format_version < 1
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if format_version != _VERSION:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if (
            set(manifest) != TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS
            or manifest.get("kind") != _KIND
            or manifest.get("namespace") != self._namespace
            or manifest.get("tenant_id") != admission.principal.tenant_id
            or manifest.get("graph_id") != admission.graph_id
            or manifest.get("request_digest")
            != admission.initial_request_digest
            or not isinstance(manifest.get("tasks"), list)
            or not isinstance(manifest.get("expanders"), list)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if canonical_json_bytes(manifest) != payload:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        tasks, expanders = read_task_graph_binding_capture_declarations(
            manifest["tasks"],
            manifest["expanders"],
        )
        return TaskGraphBindingCapture(tasks, expanders)

    def _key(self, admission: TaskGraphAdmission) -> str:
        return task_graph_binding_capture_key(
            self._namespace,
            admission.principal.tenant_id,
            admission.graph_id,
            admission.initial_request_digest,
        )


def task_declaration_semantics(
    declaration: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    semantic = dict(declaration)
    if declaration.get("type") != "agent":
        return semantic
    config = declaration.get("config")
    if not isinstance(config, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding_payload = config.get("binding_contract")
    if not isinstance(binding_payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    binding = AgentBindingContract.from_payload(binding_payload)
    semantic["config"] = {
        "agent_id": config.get("agent_id"),
        "agent_revision": config.get("agent_revision"),
        "input_mode": config.get("input_mode"),
        "binding_digest": binding.binding_digest,
    }
    return semantic


def _task_declarations(
    tasks: Sequence[Task[object]],
) -> dict[tuple[str, int], dict[str, JsonValue]]:
    declarations = dict(_BUILTIN_TASKS)
    for task in tasks:
        if not isinstance(task, Task):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        contract = {"id": task.id, "revision": task.revision, **dict(task.contract)}
        identity = task_declaration_identity(contract)
        if identity != (task.id, task.revision):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if identity in declarations:
            raise AIError(ErrorCode.BINDING_CONFLICT)
        declarations[identity] = contract
    return dict(sorted(declarations.items()))


def _expander_declarations(
    expanders: Sequence[TaskExpander],
) -> dict[tuple[str, int], dict[str, JsonValue]]:
    declarations: dict[tuple[str, int], dict[str, JsonValue]] = {}
    for expander in expanders:
        if not isinstance(expander, TaskExpander):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        contract = {"version": 1, "id": expander.id, "revision": expander.revision}
        identity = task_expander_declaration_identity(contract)
        if identity != (expander.id, expander.revision):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if identity in declarations:
            raise AIError(ErrorCode.BINDING_CONFLICT)
        declarations[identity] = contract
    return dict(sorted(declarations.items()))


def _required_task_identities(graph: TaskGraph) -> tuple[tuple[str, int], ...]:
    identities: set[tuple[str, int]] = set()
    for node in graph.nodes:
        if node.task is not None:
            identities.add(_validate_task_identity(node.task.id, node.task.revision))
    return tuple(sorted(identities))


def _required_expander_identities(
    graph: TaskGraph,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        sorted(
            {
                (node.expander.id, node.expander.revision)
                for node in graph.nodes
                if node.expander is not None
            }
        )
    )


def _validate_request_references(
    graph: TaskGraph,
    tasks: Mapping[tuple[str, int], Mapping[str, JsonValue]],
    expanders: Mapping[tuple[str, int], Mapping[str, JsonValue]],
) -> None:
    for node in graph.nodes:
        task_ref = node.task
        if task_ref is None or (task_ref.id, task_ref.revision) not in tasks:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "task",
                    "task_id": None if task_ref is None else task_ref.id,
                    "task_revision": None if task_ref is None else task_ref.revision,
                },
            )
        expander_ref = node.expander
        if expander_ref is not None and (expander_ref.id, expander_ref.revision) not in expanders:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "expander",
                    "expander_id": expander_ref.id,
                    "expander_revision": expander_ref.revision,
                },
            )


def _validate_required_declarations(
    capture: TaskGraphBindingCapture,
    graph: TaskGraph,
    tasks: Mapping[tuple[str, int], Mapping[str, JsonValue]],
    expanders: Mapping[tuple[str, int], Mapping[str, JsonValue]],
) -> None:
    for node in graph.nodes:
        task_ref = node.task
        if task_ref is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identity = (task_ref.id, task_ref.revision)
        captured = capture.tasks.get(identity)
        current = tasks.get(identity)
        if captured is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "task",
                    "ref": f"{identity[0]}@{identity[1]}",
                },
            )
        if current is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "task",
                    "ref": f"{identity[0]}@{identity[1]}",
                },
            )
        if task_declaration_semantics(captured) != task_declaration_semantics(current):
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "task",
                    "task_id": identity[0],
                    "task_revision": identity[1],
                },
            )
        expander_ref = node.expander
        if expander_ref is None:
            continue
        expander_identity = (expander_ref.id, expander_ref.revision)
        captured_expander = capture.expanders.get(expander_identity)
        current_expander = expanders.get(expander_identity)
        if captured_expander is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "expander",
                    "ref": f"{expander_identity[0]}@{expander_identity[1]}",
                },
            )
        if current_expander is None:
            raise AIError(
                ErrorCode.BINDING_NOT_REGISTERED,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "expander",
                    "ref": f"{expander_identity[0]}@{expander_identity[1]}",
                },
            )
        if dict(captured_expander) != dict(current_expander):
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "graph_id": graph.graph_id,
                    "node_id": node.node_id,
                    "role": "expander",
                    "expander_id": expander_identity[0],
                    "expander_revision": expander_identity[1],
                },
            )


def _freeze_declarations(
    values: Mapping[tuple[str, int], Mapping[str, JsonValue]],
    identity_of: Callable[[object], tuple[str, int]],
) -> dict[tuple[str, int], Mapping[str, JsonValue]]:
    if not isinstance(values, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    declarations: dict[tuple[str, int], Mapping[str, JsonValue]] = {}
    for identity, value in values.items():
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            declaration = ImmutableJsonMapping(value)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if identity_of(declaration) != identity or identity in declarations:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        declarations[identity] = declaration
    return dict(sorted(declarations.items()))


def _validate_task_identity(task_id: object, revision: object) -> tuple[str, int]:
    if (
        not isinstance(task_id, str)
        or _TASK_ID.fullmatch(task_id) is None
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return task_id, revision


__all__ = [
    "builtin_task_declaration",
    "TaskGraphBindingCapture",
    "TaskGraphBindingCaptureStore",
    "task_declaration_semantics",
]
