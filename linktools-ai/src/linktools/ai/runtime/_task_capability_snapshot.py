#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private immutable capability snapshot used by admitted TaskGraphs."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from ..agent import AgentBindingSnapshot, AgentCompiler
from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, read_object
from ..task import TaskGraph, TaskGraphAdmission, TaskNode
from ._binding_freeze import _RuntimeBindingFreezer
from ._runtime_identity import task_capability_snapshot_key

_KIND = "task-capability-snapshot"
_VERSION = 1


@dataclass(frozen=True, slots=True)
class FrozenTaskCapabilities:
    roots: Mapping[str, AgentBindingSnapshot]
    bindings: Mapping[str, AgentBindingSnapshot]

    def __post_init__(self) -> None:
        roots = dict(sorted(self.roots.items()))
        bindings = dict(sorted(self.bindings.items()))
        if (
            any(
                not isinstance(key, str)
                or not key
                or not isinstance(value, AgentBindingSnapshot)
                or value.agent_spec.id != key
                for key, value in roots.items()
            )
            or any(
                not isinstance(key, str)
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
                or not isinstance(value, AgentBindingSnapshot)
                for key, value in bindings.items()
            )
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object.__setattr__(self, "roots", MappingProxyType(roots))
        object.__setattr__(self, "bindings", MappingProxyType(bindings))


class TaskCapabilitySnapshotStore:
    def __init__(
        self,
        namespace: str,
        compiler: AgentCompiler,
        binding_freezer: _RuntimeBindingFreezer,
        object_store: ObjectStore,
        *,
        agent_task_type: str,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace is required")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        if not isinstance(binding_freezer, _RuntimeBindingFreezer):
            raise TypeError("binding_freezer must be _RuntimeBindingFreezer")
        if not isinstance(agent_task_type, str) or not agent_task_type:
            raise ValueError("agent_task_type is required")
        self._namespace = namespace
        self._compiler = compiler
        self._binding_freezer = binding_freezer
        self._objects = object_store
        self._agent_task_type = agent_task_type

    async def capture(
        self,
        admission: TaskGraphAdmission,
        graph: TaskGraph,
    ) -> FrozenTaskCapabilities:
        key = self._key(admission)
        existing = await self._objects.stat(key)
        if existing is not None:
            return await self._read(
                ObjectRef(
                    self._objects.store_id,
                    key,
                    existing.digest,
                    existing.size,
                ),
                admission,
            )

        node_bindings = tuple(
            snapshot
            for node in graph.nodes
            if (snapshot := self._node_binding(node)) is not None
        )
        unique_bindings = {
            snapshot.binding_digest: snapshot
            for snapshot in node_bindings
        }
        skill_snapshots: dict[tuple[str, str], ObjectRef] = {}
        roots: dict[str, AgentBindingSnapshot] = {}
        if any(node.expander is not None for node in graph.nodes):
            for agent_id in self._binding_freezer.root_ids:
                roots[agent_id] = await self._binding_freezer.freeze_root(
                    agent_id,
                    skill_snapshots=skill_snapshots,
                )
        bindings: dict[str, AgentBindingSnapshot] = {}
        for binding_digest, snapshot in sorted(unique_bindings.items()):
            bindings[binding_digest] = await self._binding_freezer.freeze_snapshot(
                snapshot,
                skill_snapshots=skill_snapshots,
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
            return await self._read(
                ObjectRef(
                    self._objects.store_id,
                    key,
                    current.digest,
                    current.size,
                ),
                admission,
            )
        return await self._read(
            ObjectRef(
                self._objects.store_id,
                stat.key,
                stat.digest,
                stat.size,
            ),
            admission,
        )

    async def load(
        self,
        admission: TaskGraphAdmission,
    ) -> FrozenTaskCapabilities:
        key = self._key(admission)
        stat = await self._objects.stat(key)
        if stat is None:
            raise AIError(
                ErrorCode.CAPABILITY_REQUIRED_MISSING,
                safe_details={
                    "kind": "task_capability_snapshot",
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
    ) -> AgentBindingSnapshot | None:
        if node.input.get("type") != self._agent_task_type:
            return None
        payload = node.input.get("binding")
        if not isinstance(payload, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return AgentBindingSnapshot.from_payload(payload)

    async def _read(
        self,
        ref: ObjectRef,
        admission: TaskGraphAdmission,
    ) -> FrozenTaskCapabilities:
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
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        roots = {
            str(agent_id): AgentBindingSnapshot.from_payload(value)
            for agent_id, value in cast(
                "Mapping[object, object]",
                manifest["roots"],
            ).items()
        }
        bindings = {
            str(binding_digest): AgentBindingSnapshot.from_payload(value)
            for binding_digest, value in cast(
                "Mapping[object, object]",
                manifest["bindings"],
            ).items()
        }
        for agent_id, snapshot in roots.items():
            if snapshot.agent_spec.id != agent_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._compiler.restore(snapshot)
        for original_digest, snapshot in bindings.items():
            if (
                len(original_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in original_digest
                )
                or original_digest != snapshot.binding_digest
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._compiler.restore(snapshot)
        return FrozenTaskCapabilities(roots, bindings)

    def _key(self, admission: TaskGraphAdmission) -> str:
        return task_capability_snapshot_key(
            self._namespace,
            admission.principal.tenant_id,
            admission.graph_id,
            admission.initial_request_digest,
        )


__all__ = [
    "FrozenTaskCapabilities",
    "TaskCapabilitySnapshotStore",
]
