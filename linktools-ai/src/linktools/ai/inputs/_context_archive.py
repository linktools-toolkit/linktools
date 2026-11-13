#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable accepted-context roots and portable context materialization."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from ..asset import AssetStoreReader
from ._instruction_codec import RepositoryInstructionCodec
from ._instruction_payload import RepositoryInstructionPayload
from ..spec import RepositoryInstructions
from ..core import JsonValue, PersistenceDomain, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..execution import RuntimePayloadRef
from ..storage import ObjectRef, ObjectStore, StoredPayload
from ..execution import ExecutionInputContext
from ..model.messages import combine_model_message_payloads
from ..persistence import NativeMessageRoot, SqlStepStore
from ..persistence.runs import ArchiveRoot, ContentArchive, PreparedArchive
from ..persistence.state import StateStorageGroup, StateStore, active_state_scope


@dataclass(frozen=True, slots=True)
class PreparedExecutionContext:
    reference: RuntimePayloadRef
    archive: ContentArchive
    records: PreparedArchive

    async def publish(self) -> None:
        scope = active_state_scope()
        if scope is None or not scope.writable or scope.group is not self.archive.store.storage_group:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self.archive.store.mutate(lambda tx: self.archive.publish(tx, self.records))


class ExecutionContextArchive:
    def __init__(self, store: StateStore, objects: ObjectStore, *, namespace: str, tenant_id: str,
                 object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None,
                 asset_sources: Mapping[str, AssetStoreReader] | None = None) -> None:
        self._store = store
        self._objects = objects
        self._object_reader = object_reader
        self._instructions = RepositoryInstructionCodec(asset_sources)
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._archive: ContentArchive | None = None
        self._native = SqlStepStore.for_state_store(store, namespace=namespace, tenant_id=tenant_id)

    async def _open(self, *, create: bool = False) -> ContentArchive:
        await self._store.initialize()
        if self._archive is not None:
            return self._archive
        if not isinstance(self._store.storage_group, StateStorageGroup):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        opened = await self._store.storage_group.open_archive(create=create)
        if opened is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope, store = opened
        self._archive = ContentArchive(store, scope, namespace=self._namespace,
                                       tenant_id=self._tenant_id, object_store=self._objects, object_reader=self._object_reader)
        if self._native is not None:
            self._native.bind_message_archive(self._archive)
        return self._archive

    async def encode_repository_instructions(
        self, value: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Project accepted instructions without copying retained Rule bytes."""
        return await self._instructions.encode(value)

    async def decode_repository_instructions(self, value: object) -> dict[str, JsonValue]:
        """Materialize an instruction projection using its original Asset owners."""
        return await self._instructions.decode(value)

    async def prepare_repository_instruction_payload(
        self, instructions: RepositoryInstructions, *, source_domain: PersistenceDomain,
    ) -> RuntimePayloadRef:
        """Retain accepted instruction identity and only verified Asset references."""
        stored = await self.encode_repository_instructions(instructions.to_payload())
        return RepositoryInstructionPayload.capture(instructions, stored).to_reference(source_domain)

    async def load_repository_instruction_payload(
        self, reference: RuntimePayloadRef, *, source_domain: PersistenceDomain | None = None,
    ) -> RepositoryInstructions:
        """Recover admission or overlay instructions from the original owners."""
        stored = RepositoryInstructionPayload.from_reference(reference, source_domain=source_domain)
        instructions = RepositoryInstructions.from_payload(
            await self.decode_repository_instructions(stored.instructions)
        )
        return instructions

    async def prepare(
        self, context: ExecutionInputContext, *, history_root: ArchiveRoot | NativeMessageRoot | None = None,
        history_digest: str | None = None,
    ) -> PreparedExecutionContext:
        archive = await self._open(create=True)
        prepared = PreparedArchive()
        semantic = context.to_payload()
        messages = semantic["history"]
        if (self._native is not None and isinstance(history_root, NativeMessageRoot)
                and history_digest == hashlib.sha256(context.history).hexdigest()
                and self._native.can_reuse_message_root(history_root, raw_digest=history_digest,
                                                        raw_size=len(context.history))):
            history = history_root
        elif self._native is not None:
            history = await self._native.prepare_archived_message_payloads(
                tuple(canonical_json_bytes([message]) for message in messages), prepared,
            )
        elif (isinstance(history_root, ArchiveRoot) and history_root.scope_id == archive.scope_id
                and history_root.content_scope == archive.content_scope
                and history_root.count == len(messages)
                and history_digest == hashlib.sha256(context.history).hexdigest()):
            history = history_root
        else:
            history = await archive.prepare_append(archive.empty_root(), tuple(
                ("model-message", canonical_json_bytes([message])) for message in messages
            ), prepared)
        values: list[tuple[str, bytes]] = []
        components: dict[str, object] = {}
        for name in ("session_metadata", "memory"):
            mapping = semantic[name]
            if mapping is None:
                components[name] = None
                continue
            if not isinstance(mapping, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            fields: list[tuple[str, int]] = []
            for key in sorted(mapping):
                fields.append((key, len(values)))
                values.append(("json", canonical_json_bytes(mapping[key])))
            components[name] = fields
        instructions = semantic["repository_instructions"]
        instruction_index = None
        if instructions is not None:
            instruction_index = len(values)
            values.append(("json", canonical_json_bytes(await self.encode_repository_instructions(instructions))))
        skill_indices = []
        for skill in semantic["preload_skills"]:
            skill_indices.append(len(values))
            values.append(("json", canonical_json_bytes(skill)))
        identities = await archive.prepare_content(values, prepared)

        def reference(index: int) -> dict[str, JsonValue]:
            return {"archive_scope": archive.scope_id, "content_scope": archive.content_scope,
                    "content_id": identities[index]}

        manifest: dict[str, JsonValue] = {
            "kind": "execution-context", "version": 1,
            "history": history.to_json(), "semantic_digest": canonical_sha256(semantic),
            "replace_history_system_prompt": context.replace_history_system_prompt,
            "unavailable_reason": context.unavailable_reason,
            "repository_instructions": None if instruction_index is None else reference(instruction_index),
            "preload_skills": [reference(index) for index in skill_indices],
        }
        for name, fields in components.items():
            manifest[name] = None if fields is None else {key: reference(index) for key, index in fields}
        return PreparedExecutionContext(RuntimePayloadRef(StoredPayload.inline_json(manifest), PersistenceDomain.EXECUTION),
                                        archive, prepared)

    async def load_repository_instructions(
        self, reference: RuntimePayloadRef,
    ) -> Mapping[str, JsonValue] | None:
        """Read the accepted instruction component without materializing model history."""
        try:
            if (reference.source_domain is not PersistenceDomain.EXECUTION
                    or reference.payload.kind != "inline" or reference.payload.encoding != "json"):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = reference.payload.decode()
            if not isinstance(value, Mapping) or value.get("kind") != "execution-context" or value.get("version") != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            item = value["repository_instructions"]
            if item is None:
                return None
            archive = await self._open()
            if (not isinstance(item, Mapping) or item.get("archive_scope") != archive.scope_id
                    or item.get("content_scope") != archive.content_scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            bodies = await archive.read_contents((item["content_id"],), codec="json")
            instructions = json.loads(bodies[0])
            if not isinstance(instructions, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await self.decode_repository_instructions(instructions)
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    async def load(self, reference: RuntimePayloadRef) -> ExecutionInputContext:
        if reference.payload.kind != "inline":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = reference.payload.decode()
        if not isinstance(value, Mapping) or value.get("kind") != "execution-context" or value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = await self._open()
        try:
            wire = value["history"]
            if isinstance(wire, Mapping) and wire.get("type") == "native_message_root":
                root = NativeMessageRoot.from_json(wire)
                if self._native is None:
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                message_bytes = (await self._native.read_message_roots((root,)))[0]
            else:
                root = ArchiveRoot.from_json(wire)
                message_bytes = (await archive.read_ranges((root,), codec="model-message"))[0]
            history = combine_model_message_payloads(message_bytes)
            references: list[Mapping[str, JsonValue]] = []
            maps: dict[str, list[tuple[str, int]] | None] = {}
            for name in ("session_metadata", "memory"):
                fields = value[name]
                if fields is None:
                    maps[name] = None
                    continue
                if not isinstance(fields, Mapping):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                items = []
                for key, item in fields.items():
                    items.append((key, len(references)))
                    references.append(item)
                maps[name] = items
            instruction_index = None
            if value["repository_instructions"] is not None:
                instruction_index = len(references)
                references.append(value["repository_instructions"])
            skill_indices = []
            skills = value["preload_skills"]
            if not isinstance(skills, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for skill in skills:
                skill_indices.append(len(references))
                references.append(skill)
            ids = []
            for item in references:
                if (not isinstance(item, Mapping) or item.get("archive_scope") != archive.scope_id
                        or item.get("content_scope") != archive.content_scope):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                ids.append(item["content_id"])
            bodies = tuple(json.loads(raw) for raw in await archive.read_contents(ids, codec="json"))
            fields = {name: None if items is None else {key: bodies[index] for key, index in items}
                      for name, items in maps.items()}
            result = ExecutionInputContext(history,
                session_metadata=fields["session_metadata"], memory=fields["memory"],
                repository_instructions=None if instruction_index is None else await self.decode_repository_instructions(bodies[instruction_index]),
                replace_history_system_prompt=value["replace_history_system_prompt"],
                unavailable_reason=value["unavailable_reason"],
                preload_skills=tuple(bodies[index] for index in skill_indices))
            if result.digest != value["semantic_digest"]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["ExecutionContextArchive", "PreparedExecutionContext"]
