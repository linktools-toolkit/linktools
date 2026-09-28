#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private immutable capability capture used by admitted TaskGraphs."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, cast

from ..agent import AgentBindingContract, AgentCompiler
from ..capability import CapabilityContribution
from ..core import ImmutableJsonMapping, JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, read_object
from ..task import TaskGraph, TaskGraphAdmission, TaskNode
from ._agent_binding_resolver import _AgentBindingResolver
from ._runtime_identity import task_capability_capture_key
from .state._task_capability_capture import (
    TASK_CAPABILITY_CAPTURE_FORMAT_VERSION,
    read_task_capability_capture_declarations,
    task_declaration_identity,
    task_expander_declaration_identity,
)

_KIND = "task-capability-capture"
_VERSION = TASK_CAPABILITY_CAPTURE_FORMAT_VERSION
_TASK_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_BUILTIN_TASKS: dict[tuple[str, int], dict[str, JsonValue]] = {
    ("linktools.ai.agent", 1): {
        "version": 1,
        "id": "linktools.ai.agent",
        "revision": 1,
        "effect_policy": "none",
        "output_contract": {"kind": "json"},
        "reconcile": False,
    },
    ("linktools.ai.input", 1): {
        "version": 1,
        "id": "linktools.ai.input",
        "revision": 1,
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
    if declaration is None:
        return None
    return ImmutableJsonMapping(declaration)


@dataclass(frozen=True, slots=True)
class TaskCapabilityCapture:
    roots: Mapping[str, AgentBindingContract]
    bindings: Mapping[str, AgentBindingContract]
    tasks: Mapping[tuple[str, int], Mapping[str, JsonValue]] = field(
        default_factory=dict
    )
    expanders: Mapping[tuple[str, int], Mapping[str, JsonValue]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        roots = dict(sorted(self.roots.items()))
        bindings = dict(sorted(self.bindings.items()))
        tasks = _freeze_declarations(self.tasks, task_declaration_identity)
        expanders = _freeze_declarations(
            self.expanders,
            task_expander_declaration_identity,
        )
        if (
            any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, AgentBindingContract)
                or value.agent_spec.id != key
                for key, value in roots.items()
            )
            or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
                or not isinstance(value, AgentBindingContract)
                for key, value in bindings.items()
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object.__setattr__(self, "roots", MappingProxyType(roots))
        object.__setattr__(self, "bindings", MappingProxyType(bindings))
        object.__setattr__(self, "tasks", MappingProxyType(tasks))
        object.__setattr__(self, "expanders", MappingProxyType(expanders))


class TaskCapabilityCaptureStore:
    def __init__(
        self,
        namespace: str,
        compiler: AgentCompiler,
        binding_resolver: _AgentBindingResolver,
        object_store: ObjectStore,
        *,
        agent_task_id: str,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace is required")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        if not isinstance(binding_resolver, _AgentBindingResolver):
            raise TypeError("binding_resolver must be _AgentBindingResolver")
        if not isinstance(agent_task_id, str) or not agent_task_id:
            raise ValueError("agent_task_id is required")
        self._namespace = namespace
        self._compiler = compiler
        self._binding_resolver = binding_resolver
        self._objects = object_store
        self._agent_task_id = agent_task_id

    async def capture(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
        *,
        task_contributions: Sequence[CapabilityContribution[object]] = (),
        expander_contributions: Sequence[CapabilityContribution[object]] = (),
    ) -> TaskCapabilityCapture:
        key = self._key(admission)
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
                task_contributions,
                expander_contributions,
            )
            return capture

        task_declarations = _task_declarations(task_contributions)
        expander_declarations = _expander_declarations(expander_contributions)
        if any(node.expander is not None for node in graph.nodes):
            captured_tasks = task_declarations
            captured_expanders = expander_declarations
        else:
            captured_tasks = {
                identity: task_declarations[identity]
                for identity in _required_task_identities(graph)
                if identity in task_declarations
            }
            captured_expanders = {}
        for identity in _required_task_identities(graph):
            if identity not in captured_tasks:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)
        for identity in _required_expander_identities(graph):
            if identity not in captured_expanders:
                raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING)

        node_bindings = tuple(
            binding_contract
            for node in graph.nodes
            if (binding_contract := self._node_binding(node)) is not None
        )
        unique_bindings = {
            binding_contract.binding_digest: binding_contract
            for binding_contract in node_bindings
        }
        roots: dict[str, AgentBindingContract] = {}
        if any(node.expander is not None for node in graph.nodes):
            for agent_id in self._binding_resolver.root_ids:
                roots[agent_id] = await self._binding_resolver.resolve_root(agent_id)
        bindings: dict[str, AgentBindingContract] = {}
        for binding_digest, binding_contract in sorted(unique_bindings.items()):
            bindings[binding_digest] = await self._binding_resolver.resolve_contract(
                binding_contract
            )

        manifest: dict[str, JsonValue] = {
            "kind": _KIND,
            "format_version": _VERSION,
            "namespace": self._namespace,
            "tenant_id": admission.principal.tenant_id,
            "graph_id": admission.graph_id,
            "request_digest": admission.initial_request_digest,
            "roots": {
                key: value.to_payload()
                for key, value in roots.items()
            },
            "bindings": {
                key: value.to_payload()
                for key, value in sorted(bindings.items())
            },
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
            _validate_required_declarations(
                capture,
                graph,
                task_contributions,
                expander_contributions,
            )
            return capture
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
            task_contributions,
            expander_contributions,
        )
        return capture

    async def load(
        self,
        admission: TaskGraphAdmission,
    ) -> TaskCapabilityCapture:
        key = self._key(admission)
        stat = await self._objects.stat(key)
        if stat is None:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "task_capability_capture",
                    "graph_id": admission.graph_id,
                },
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

    def _node_binding(
        self,
        node: TaskNode,
    ) -> AgentBindingContract | None:
        if node.input.get("task_id") != self._agent_task_id:
            return None
        payload = node.input.get("binding_contract")
        if not isinstance(payload, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return AgentBindingContract.from_payload(payload)

    async def _read(
        self,
        ref: ObjectRef,
        admission: TaskGraphAdmission,
    ) -> TaskCapabilityCapture:
        payload = await read_object(
            self._objects,
            ref.key,
            expected_digest=ref.digest,
            expected_size=ref.size,
        )
        import json

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
            manifest.get("kind") != _KIND
            or manifest.get("namespace") != self._namespace
            or manifest.get("tenant_id")
            != admission.principal.tenant_id
            or manifest.get("graph_id") != admission.graph_id
            or manifest.get("request_digest")
            != admission.initial_request_digest
            or not isinstance(manifest.get("roots"), Mapping)
            or not isinstance(manifest.get("bindings"), Mapping)
            or not isinstance(manifest.get("tasks"), list)
            or not isinstance(manifest.get("expanders"), list)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        roots = {
            str(agent_id): AgentBindingContract.from_payload(value)
            for agent_id, value in cast(
                "Mapping[object, object]",
                manifest["roots"],
            ).items()
        }
        bindings = {
            str(binding_digest): AgentBindingContract.from_payload(value)
            for binding_digest, value in cast(
                "Mapping[object, object]",
                manifest["bindings"],
            ).items()
        }
        for agent_id, binding_contract in roots.items():
            if binding_contract.agent_spec.id != agent_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._compiler.restore(binding_contract)
        for source_digest, binding_contract in bindings.items():
            if (
                len(source_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in source_digest
                )
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._compiler.restore(binding_contract)
        tasks, expanders = read_task_capability_capture_declarations(
            manifest["tasks"],
            manifest["expanders"],
        )
        return TaskCapabilityCapture(roots, bindings, tasks, expanders)

    def _key(self, admission: TaskGraphAdmission) -> str:
        return task_capability_capture_key(
            self._namespace,
            admission.principal.tenant_id,
            admission.graph_id,
            admission.initial_request_digest,
        )


def _task_declarations(
    contributions: Sequence[CapabilityContribution[object]],
) -> dict[tuple[str, int], dict[str, JsonValue]]:
    declarations = dict(_BUILTIN_TASKS)
    for contribution in contributions:
        if (
            not isinstance(contribution, CapabilityContribution)
            or contribution.kind != "task"
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        contract = contribution.contract
        identity = task_declaration_identity(contract)
        if identity != (contribution.id, contribution.revision):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if identity in declarations:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        declarations[identity] = contract
    return dict(sorted(declarations.items()))


def _expander_declarations(
    contributions: Sequence[CapabilityContribution[object]],
) -> dict[tuple[str, int], dict[str, JsonValue]]:
    declarations: dict[tuple[str, int], dict[str, JsonValue]] = {}
    for contribution in contributions:
        if (
            not isinstance(contribution, CapabilityContribution)
            or contribution.kind != "task_expander"
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        contract = contribution.contract
        identity = task_expander_declaration_identity(contract)
        if identity != (contribution.id, contribution.revision):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if identity in declarations:
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        declarations[identity] = contract
    return dict(sorted(declarations.items()))


def _required_task_identities(graph: TaskGraph) -> tuple[tuple[str, int], ...]:
    identities: set[tuple[str, int]] = set()
    for node in graph.nodes:
        identities.add(
            _validate_task_identity(
                node.input.get("task_id"),
                node.input.get("task_revision"),
            )
        )
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


def _validate_required_declarations(
    capture: TaskCapabilityCapture,
    graph: TaskGraph,
    task_contributions: Sequence[CapabilityContribution[object]],
    expander_contributions: Sequence[CapabilityContribution[object]],
) -> None:
    current_tasks = _task_declarations(task_contributions)
    current_expanders = _expander_declarations(expander_contributions)
    for identity in _required_task_identities(graph):
        captured = capture.tasks.get(identity)
        current = current_tasks.get(identity)
        if captured is None or current is None:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "task",
                    "task_id": identity[0],
                    "task_revision": identity[1],
                },
            )
        if dict(captured) != current:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "kind": "task",
                    "task_id": identity[0],
                    "task_revision": identity[1],
                },
            )
    for identity in _required_expander_identities(graph):
        captured = capture.expanders.get(identity)
        current = current_expanders.get(identity)
        if captured is None or current is None:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "task_expander",
                    "expander_id": identity[0],
                    "expander_revision": identity[1],
                },
            )
        if dict(captured) != current:
            raise AIError(
                ErrorCode.STORAGE_INTEGRITY_ERROR,
                safe_details={
                    "kind": "task_expander",
                    "expander_id": identity[0],
                    "expander_revision": identity[1],
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


def _validate_task_identity(
    task_id: object,
    revision: object,
) -> tuple[str, int]:
    if (
        not isinstance(task_id, str)
        or _TASK_ID.fullmatch(task_id) is None
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if task_id.startswith("linktools.ai.") and (task_id, revision) not in _BUILTIN_TASKS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return task_id, revision


__all__ = [
    "builtin_task_declaration",
    "TaskCapabilityCapture",
    "TaskCapabilityCaptureStore",
]
