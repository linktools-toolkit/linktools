#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Private immutable capability snapshot used by admitted TaskGraphs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from ..agent import (
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    SemanticPin,
)
from ..capability import (
    SkillDefinition,
    SkillSourceRegistry,
    SnapshotSkillResourceSource,
)
from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, ObjectStore, StorageRevision, read_object
from ..task import TaskGraph, TaskGraphAdmission, TaskNode

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
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        skill_sources: SkillSourceRegistry,
        object_store: ObjectStore,
        *,
        agent_task_type: str,
    ) -> None:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace is required")
        if not isinstance(catalog, AgentCatalog):
            raise TypeError("catalog must be AgentCatalog")
        if not isinstance(compiler, AgentCompiler):
            raise TypeError("compiler must be AgentCompiler")
        if not isinstance(skill_sources, SkillSourceRegistry):
            raise TypeError("skill_sources must be SkillSourceRegistry")
        if not isinstance(agent_task_type, str) or not agent_task_type:
            raise ValueError("agent_task_type is required")
        self._namespace = namespace
        self._catalog = catalog
        self._compiler = compiler
        self._skill_sources = skill_sources
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

        roots = await self._freeze_roots()
        bindings: dict[str, AgentBindingSnapshot] = {}
        for node in graph.nodes:
            snapshot = self._node_binding(node)
            if snapshot is None:
                continue
            bindings[snapshot.binding_digest] = await self._freeze_binding(
                snapshot,
                roots=roots,
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

    async def _freeze_roots(
        self,
    ) -> Mapping[str, AgentBindingSnapshot]:
        base: dict[str, AgentBindingSnapshot] = {}
        for agent_id in self._catalog.root_ids:
            definition = self._catalog.root_definition(agent_id)
            snapshot = self._compiler.bind(
                definition,
                output=None,
            ).snapshot
            base[agent_id] = await self._freeze_skills(snapshot)

        roots: dict[str, AgentBindingSnapshot] = {}
        for agent_id, snapshot in sorted(base.items()):
            children = tuple(
                self._compiler.bind_subagent(
                    self._compiler.restore(base[child_id]).definition
                ).snapshot
                for child_id in snapshot.subagent_ids
            )
            roots[agent_id] = replace(
                snapshot,
                subagent_bindings=children,
            )
        return MappingProxyType(roots)

    async def _freeze_binding(
        self,
        snapshot: AgentBindingSnapshot,
        *,
        roots: Mapping[str, AgentBindingSnapshot],
    ) -> AgentBindingSnapshot:
        frozen = await self._freeze_skills(snapshot)
        children: list[AgentBindingSnapshot] = []
        for child_id in frozen.subagent_ids:
            root = roots.get(child_id)
            if root is None:
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "agent",
                        "agent_id": child_id,
                    },
                )
            children.append(
                self._compiler.bind_subagent(
                    self._compiler.restore(root).definition
                ).snapshot
            )
        return replace(
            frozen,
            subagent_bindings=tuple(children),
        )

    async def _freeze_skills(
        self,
        snapshot: AgentBindingSnapshot,
    ) -> AgentBindingSnapshot:
        selected: list[SemanticPin] = []
        for pin in snapshot.selected:
            if pin.kind != "skill":
                selected.append(pin)
                continue
            skill = SkillDefinition.from_semantic_contract(
                cast("Mapping[str, object]", pin.contract)
            )
            source_ref = skill.source_ref
            if source_ref is None or source_ref.snapshot is not None:
                selected.append(pin)
                continue
            source = self._skill_sources.resolve(source_ref.source_id)
            if not isinstance(source, SnapshotSkillResourceSource):
                raise AIError(
                    ErrorCode.CAPABILITY_REQUIRED_MISSING,
                    safe_details={
                        "kind": "skill_snapshot",
                        "skill_id": skill.id,
                        "source_id": source_ref.source_id,
                    },
                )
            revision = await source.current_revision(source_ref.root)
            if not isinstance(revision, StorageRevision):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            reference = await source.snapshot(
                source_ref.root,
                expected_revision=revision,
                object_store=self._objects,
            )
            frozen_skill = SkillDefinition(
                skill.spec,
                source_ref.with_snapshot(reference),
            )
            selected.append(
                SemanticPin(
                    "skill",
                    pin.id,
                    frozen_skill.semantic_contract,
                )
            )
        return replace(snapshot, selected=tuple(selected))

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
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("kind") != _KIND
            or manifest.get("format_version") != _VERSION
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
            if len(original_digest) != 64:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._compiler.restore(snapshot)
        return FrozenTaskCapabilities(roots, bindings)

    def _key(self, admission: TaskGraphAdmission) -> str:
        digest = canonical_sha256(
            {
                "version": 1,
                "namespace": self._namespace,
                "tenant_id": admission.principal.tenant_id,
                "graph_id": admission.graph_id,
                "request_digest": admission.initial_request_digest,
            }
        )
        return f"v1/task-capability-snapshot/{digest}"


__all__ = [
    "FrozenTaskCapabilities",
    "TaskCapabilitySnapshotStore",
]
