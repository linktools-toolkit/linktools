#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable accepted-context roots and portable context materialization."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass

from ..core import JsonValue, PersistenceDomain, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..execution import RuntimePayloadRef
from ..storage import ObjectRef, ObjectStore, StoredPayload
from ..execution import ExecutionInputContext
from ..model.messages import decode_model_messages
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
                 object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None) -> None:
        self._store = store
        self._objects = objects
        self._object_reader = object_reader
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._archive: ContentArchive | None = None

    async def _open(self, *, create: bool = False) -> ContentArchive:
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
        return self._archive

    async def prepare(
        self, context: ExecutionInputContext, *, history_root: ArchiveRoot | None = None,
        history_digest: str | None = None,
    ) -> PreparedExecutionContext:
        archive = await self._open(create=True)
        prepared = PreparedArchive()
        messages = tuple(canonical_json_bytes([message]) for message in json.loads(context.history))
        if (history_root is not None and history_root.scope_id == archive.scope_id
                and history_root.content_scope == archive.content_scope
                and history_root.count == len(messages)
                and history_digest == hashlib.sha256(context.history).hexdigest()):
            history = history_root
        else:
            history = await archive.prepare_append(archive.empty_root(), tuple(
                ("model-message", message) for message in messages
            ), prepared)
        semantic = context.to_payload()
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
            values.append(("json", canonical_json_bytes(instructions)))
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
        }
        for name, fields in components.items():
            manifest[name] = None if fields is None else {key: reference(index) for key, index in fields}
        return PreparedExecutionContext(RuntimePayloadRef(StoredPayload.inline_json(manifest), PersistenceDomain.EXECUTION),
                                        archive, prepared)

    async def load(self, reference: RuntimePayloadRef) -> ExecutionInputContext:
        if reference.payload.kind != "inline":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = reference.payload.decode()
        if not isinstance(value, Mapping) or value.get("kind") != "execution-context" or value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        archive = await self._open()
        try:
            root = ArchiveRoot.from_json(value["history"])
            message_bytes = (await archive.read_ranges((root,), codec="model-message"))[0]
            decoded = tuple(decode_model_messages(raw) for raw in message_bytes)
            if any(len(messages) != 1 for messages in decoded):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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
            ids = []
            for item in references:
                if (not isinstance(item, Mapping) or item.get("archive_scope") != archive.scope_id
                        or item.get("content_scope") != archive.content_scope):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                ids.append(item["content_id"])
            bodies = tuple(json.loads(raw) for raw in await archive.read_contents(ids, codec="json"))
            fields = {name: None if items is None else {key: bodies[index] for key, index in items}
                      for name, items in maps.items()}
            history = b"[" + b",".join(raw[1:-1] for raw in message_bytes) + b"]"
            result = ExecutionInputContext(history,
                session_metadata=fields["session_metadata"], memory=fields["memory"],
                repository_instructions=None if instruction_index is None else bodies[instruction_index],
                replace_history_system_prompt=value["replace_history_system_prompt"],
                unavailable_reason=value["unavailable_reason"])
            if result.digest != value["semantic_digest"]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["ExecutionContextArchive", "PreparedExecutionContext"]
