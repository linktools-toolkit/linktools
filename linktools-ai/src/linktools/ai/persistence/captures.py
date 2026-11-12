#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoped state ownership and reference closure for immutable input captures."""

from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import cast
import json

from ..agent import AgentInputCaptureRef
from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ..task import TaskDependencyCapture, TaskGraphCaptureRef, TaskGraphTemplateRef, TaskInvocationInputRef
from ..storage import ObjectRef, StoredPayload
from ..model import StoredUserInput
from ..model.attachments import store_attachment_payload, read_attachment_payload
from ._archive import archive_content_scope, decode_archive_scope, validate_archive_content_identity, validate_archive_group_id, ContentArchive, ContentRecord, PreparedArchive
from ._payload import decode_payload_bytes
from ._capture_binding_content import (
    BINDING_CODEC, capture_binding_references, expand_capture_bindings, map_capture_binding_slots,
    prepare_capture_bindings, restore_capture_bindings,
)
from .codec import binding_reference, decode_domain, encode_domain, decode_envelope, encode_envelope
from .state import (StateGroupTransaction, StateStore, StoredRecord, active_state_scope,
                    active_state_group_transaction, record_key, scope_key)

_CAPTURE_TYPES = {
    "agent_input_capture_ref": (AgentInputCaptureRef, "agent"),
    "task_invocation_input_ref": (TaskInvocationInputRef, "task"),
    "task_graph_capture_ref": (TaskGraphCaptureRef, "graph"),
    "task_graph_template_ref": (TaskGraphTemplateRef, "template"),
}


CAPTURE_RECORD_KINDS = frozenset({"input_capture", "task_graph_binding_capture"})
TASK_BINDING_CODEC = "task-graph-binding"


def capture_record(namespace: str, tenant_id: str, key: str, value: JsonValue, *,
                   storage_revision: int = 0, source_digest: str | None = None) -> StoredRecord:
    if not isinstance(key, str) or not key or not key.isascii() or len(key) > 512:
        raise ValueError("capture key must contain 1..512 ASCII bytes")
    kind = "task_graph_binding_capture" if key.startswith("v1/task-graph-binding-capture/") else "input_capture"
    owner = value.get("principal") if isinstance(value, Mapping) else None
    scope = (scope_key(namespace, tenant_id, "task", kind, "owner", (owner["principal_id"],))
             if isinstance(owner, Mapping) and isinstance(owner.get("principal_id"), str) else None)
    payload = {"namespace": namespace, "tenant_id": tenant_id, "key": key, "value": value}
    if source_digest is not None:
        payload["source_digest"] = source_digest
    return StoredRecord(record_key(namespace, tenant_id, "task", kind, (key,)),
                        scope, None, kind, key.encode("ascii"), None, storage_revision, None, 0, None,
                        encode_envelope({"type": kind, "payload": payload}))


def _capture_record_payload(record: StoredRecord) -> Mapping[str, JsonValue]:
    envelope = decode_envelope(record.data).value
    payload = envelope.get("payload")
    if (record.kind not in CAPTURE_RECORD_KINDS or envelope.get("type") != record.kind
            or not isinstance(payload, Mapping) or not {"namespace", "tenant_id", "key", "value"}.issubset(payload)
            or set(payload) - {"namespace", "tenant_id", "key", "value", "source_digest"}
            or not isinstance(payload["key"], str) or not 1 <= len(payload["key"]) <= 512 or not payload["key"].isascii()
            or payload["key"].encode("ascii") != record.sort_key):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return payload


def capture_record_key(record: StoredRecord) -> str:
    """Return the validated logical capture path owned by this record."""
    return cast(str, _capture_record_payload(record)["key"])


def capture_record_value(record: StoredRecord) -> JsonValue:
    return _capture_record_payload(record)["value"]


async def read_capture_record(store: StateStore, *, namespace: str, tenant_id: str,
                              key: str) -> StoredRecord | None:
    identity = capture_record(namespace, tenant_id, key, None).key
    return await store.read(lambda transaction: transaction.get_record(identity))


async def read_capture_value(store: StateStore, *, namespace: str, tenant_id: str, key: str,
                             read_object: Callable[[ObjectRef], Awaitable[bytes]]) -> JsonValue | None:
    """Capture an owner and its binding bytes together before reading attachments."""
    group = store.storage_group

    async def capture(_transactions: StateGroupTransaction) -> tuple[StoredRecord | None, bool]:
        record = await read_capture_record(store, namespace=namespace, tenant_id=tenant_id, key=key)
        if record is None:
            return None, False
        if capture_record_scope(record) != (namespace, tenant_id):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = capture_record_value(record)
        if not capture_binding_references(key, value):
            return record, False
        opened, active = group.opened_archive, active_state_scope()
        if opened is None or active is None or opened[1] not in active.transactions:
            return None, True
        archive = ContentArchive(opened[1], opened[0], namespace=namespace, tenant_id=tenant_id, object_store=None)
        value = await restore_capture_bindings(key, value, archive)
        return capture_record(namespace, tenant_id, key, value, storage_revision=record.storage_revision,
                              source_digest=capture_record_digest(record)), False

    if active_state_scope() is None:
        record, requires_archive = await group.read_many((store,), capture)
    else:
        record, requires_archive = await capture(active_state_group_transaction(group, (store,), writable=False))
    if requires_archive:
        if active_state_scope() is not None:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY,
                          safe_details={"reason": "capture_archive_not_enlisted"})
        if await group.open_archive(create=False) is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record, requires_archive = await group.read_many((store,), capture)
        if requires_archive:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return None if record is None else await restore_capture_value(record, read_object=read_object)


async def publish_capture_record(store: StateStore, *, namespace: str, tenant_id: str,
                                 key: str, value: JsonValue, source_digest: str | None = None,
                                 share_bindings: bool = False) -> StoredRecord:
    record = capture_record(namespace, tenant_id, key, value, source_digest=source_digest)
    bindings = []
    if share_bindings:
        def collect(binding: Mapping[str, JsonValue]) -> JsonValue:
            bindings.append(binding)
            return binding
        map_capture_binding_slots(key, value, collect)
    try:
        if bindings:
            group = store.storage_group
            opened = group.opened_archive or await group.open_archive(create=True)
            if opened is None:
                raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
            archive = ContentArchive(opened[1], opened[0], namespace=namespace, tenant_id=tenant_id, object_store=None)

            async def publish(transactions: StateGroupTransaction) -> StoredRecord:
                prepared = PreparedArchive()
                projected = await prepare_capture_bindings(key, value, archive, prepared)
                projected_record = capture_record(namespace, tenant_id, key, projected,
                    source_digest=capture_record_digest(record))
                await archive.publish(transactions.transaction(archive.store), prepared)
                await transactions.transaction(store).insert_record(projected_record)
                return projected_record

            record = await group.mutate((store, archive.store), publish)
        else:
            await store.mutate(lambda transaction: transaction.insert_record(record))
    except AIError as error:
        if error.code is not ErrorCode.STORAGE_CONFLICT:
            raise
        existing = await read_capture_record(store, namespace=namespace, tenant_id=tenant_id, key=key)
        if existing is None or capture_record_digest(existing) != capture_record_digest(record):
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT) from error
        return existing
    return record


def capture_record_digest(record: StoredRecord) -> str:
    value = capture_record_value(record)
    digest = decode_envelope(record.data).value["payload"].get("source_digest")
    if digest is None:
        return canonical_sha256(value)
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return digest


def _map_capture_attachments(
    key: str, value: JsonValue, transform: Callable[[JsonValue, str, bool], JsonValue],
    reference: Callable[[JsonValue], JsonValue] = lambda value: value,
) -> JsonValue:
    def mapping(value: JsonValue, encoded: bool) -> dict[str, JsonValue] | None:
        if not isinstance(value, Mapping):
            return None
        if not encoded:
            return dict(value)
        pairs = value.get("$mapping")
        if not isinstance(pairs, list):
            return None
        return {name: item for name, item in pairs}

    def mapped(original: JsonValue, result: Mapping[str, JsonValue], encoded: bool) -> JsonValue:
        return {"$mapping": [[name, result[name]] for name, _item in original["$mapping"]]} if encoded else dict(result)

    def prompt(value: JsonValue, encoded: bool = False) -> JsonValue:
        fields = mapping(value, encoded)
        if fields is None:
            return value
        if fields.get("kind") == "stored-user-content-v1":
            stored = decode_domain(fields["value"], object) if encoded else fields["value"]
            user_input = decode_domain(stored, StoredUserInput)
            original = encode_domain(user_input.payload)
            rewritten = reference(original)
            if rewritten != original:
                payload = decode_domain(rewritten, StoredPayload)
                stored = encode_domain(replace(user_input, payload=payload))
                fields["value"] = encode_domain(stored) if encoded else stored
        elif fields.get("kind") == "pydantic-user-content-v1":
            fields["value"] = transform(fields["value"], "user-content", encoded)
        elif fields.get("kind") == "task-user-content-v1":
            items = []
            for item in fields["items"]:
                entry = mapping(item, encoded)
                if entry is not None and entry.get("kind") == "native":
                    entry["value"] = transform(entry["value"], "user-content", encoded)
                    item = mapped(item, entry, encoded)
                items.append(item)
            fields["items"] = items
        return mapped(value, fields, encoded)

    def context(value: JsonValue, encoded: bool = False) -> JsonValue:
        fields = mapping(value, encoded)
        if fields is None:
            return value
        fields["history"] = transform(fields["history"], "model-messages", encoded)
        return mapped(value, fields, encoded)

    def agent_input(value: JsonValue, *, encoded: bool = False, fallback: bool = False) -> JsonValue:
        fields = mapping(value, encoded)
        if fields is None or (not fallback and fields.get("kind") != "agent-task-input"):
            return value
        for name in ("prompt", "capture_files"):
            if name in fields:
                fields[name] = prompt(fields[name], encoded)
        if fields.get("capture_context") is not None:
            fields["capture_context"] = context(fields["capture_context"], encoded)
        return mapped(value, fields, encoded)

    def contract(value: JsonValue) -> JsonValue:
        if not isinstance(value, Mapping):
            return value
        fields = value.get("fields")
        if value.get("$dataclass") != "task_invocation_input_contract" or not isinstance(fields, Mapping):
            return value
        if "dependencies" in fields:
            fields = {**fields, "dependencies": reference(fields["dependencies"])}
        binding = mapping(fields.get("binding"), True)
        if binding is None or binding.get("type") != "agent":
            return {**value, "fields": fields}
        return {**value, "fields": {**fields,
            "input": agent_input(fields["input"], encoded=True),
            "original_input": agent_input(fields["original_input"], encoded=True, fallback=True)}}

    def node(value: JsonValue) -> JsonValue:
        if not isinstance(value, Mapping) or value.get("$dataclass") != "task_node":
            return value
        fields = value["fields"]
        if "input_capture" in fields:
            fields = {**fields, "input_capture": reference(fields["input_capture"])}
        if fields.get("input_semantics") != "agent-task-input-v1":
            return {**value, "fields": fields}
        result = {**fields, "input": agent_input(fields["input"], encoded=True)}
        if result.get("original_input") is not None:
            result["original_input"] = agent_input(result["original_input"], encoded=True, fallback=True)
        return {**value, "fields": result}

    def template(value: JsonValue) -> JsonValue:
        if not isinstance(value, Mapping) or value.get("$dataclass") != "task_graph_template":
            return value
        fields = value["fields"]
        nodes = fields["nodes"]
        return {**value, "fields": {**fields, "nodes": {"$tuple": [node(item) for item in nodes["$tuple"]]}}}

    if key.startswith("v1/input-capture/agent/"):
        payload = value["payload"]
        result = {**payload, "prompt": prompt(payload["prompt"]),
                  "original_input": agent_input(payload["original_input"], fallback=True)}
        if payload.get("input_view") is not None:
            result["input_view"] = agent_input(payload["input_view"], fallback=True)
        if payload.get("input_context") is not None:
            result["input_context"] = context(payload["input_context"])
        if payload.get("task_input") is not None:
            result["task_input"] = contract(payload["task_input"])
        return {**value, "payload": result}
    if key.startswith("v1/input-capture/task/"):
        return {**value, "payload": {**value["payload"], "contract": contract(value["payload"]["contract"])}}
    if key.startswith(("v1/input-capture/graph/", "v1/input-capture/template/")):
        return {**value, "payload": {**value["payload"], "template": template(value["payload"]["template"])}}
    if key.startswith("v1/input-capture/invocation/"):
        return {**value, "node": node(value["node"])}
    if key.startswith("v1/input-capture/declaration/"):
        return template(value)
    return value


async def _transform_capture_attachments(
    key: str, value: JsonValue, transform: Callable[[JsonValue, str, bool], Awaitable[JsonValue]],
) -> JsonValue:
    slots: list[tuple[JsonValue, str, bool]] = []

    def collect(content: JsonValue, format: str, encoded: bool) -> JsonValue:
        slots.append((content, format, encoded))
        return content

    _map_capture_attachments(key, value, collect)
    transformed = iter([await transform(*slot) for slot in slots])
    return _map_capture_attachments(key, value, lambda *_args: next(transformed))


async def prepare_capture_value(key: str, value: JsonValue, *,
                                write_object: Callable[[bytes], Awaitable[ObjectRef]]) -> JsonValue:
    async def transform(value: JsonValue, format: str, encoded: bool) -> JsonValue:
        content = decode_domain(value, object) if encoded else value
        payload = await store_attachment_payload(canonical_json_bytes(content), format=format, write_object=write_object)
        return value if payload is None else encode_domain(payload)
    return await _transform_capture_attachments(key, value, transform)


async def restore_capture_value(record: StoredRecord, *,
                                read_object: Callable[[ObjectRef], Awaitable[bytes]]) -> JsonValue:
    async def transform(value: JsonValue, format: str, encoded: bool) -> JsonValue:
        if not isinstance(value, Mapping) or value.get("$dataclass") != "stored_payload":
            return value
        payload = decode_domain(value, StoredPayload)
        if payload.encoding != "json-attachments" or payload.value["format"] != format:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = json.loads(await read_attachment_payload(payload, read_object=read_object))
        return encode_domain(content) if encoded else content
    value = capture_record_value(record)
    key = capture_record_key(record)
    if capture_binding_references(key, value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    value = await _transform_capture_attachments(key, value, transform)
    return value


def capture_record_scope(record: StoredRecord) -> tuple[str, str]:
    capture_record_value(record)
    payload = decode_envelope(record.data).value["payload"]
    namespace, tenant_id = payload["namespace"], payload["tenant_id"]
    if not isinstance(namespace, str) or not namespace or not isinstance(tenant_id, str) or not tenant_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return namespace, tenant_id


def map_capture_references(record: StoredRecord, transform: Callable[[JsonValue], JsonValue]) -> StoredRecord:
    """Transform only reference-bearing slots owned by the capture contract."""
    value = capture_record_value(record)

    def attachment(content: JsonValue, format: str, encoded: bool) -> JsonValue:
        if isinstance(content, Mapping) and content.get("$dataclass") == "stored_payload":
            payload = decode_domain(content, StoredPayload)
            if payload.encoding != "json-attachments" or payload.value["format"] != format:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return transform(content)
        return content

    value = _map_capture_attachments(capture_record_key(record), value, attachment, transform)
    value = map_capture_binding_slots(capture_record_key(record), value,
        lambda binding: transform(binding) if binding_reference(binding) is not None else binding)
    if record.kind == "task_graph_binding_capture" and isinstance(value, Mapping) and "declarations" in value:
        value = {**value, "declarations": transform(value["declarations"])}
    envelope = decode_envelope(record.data).value
    return replace(record, data=encode_envelope({**envelope, "payload": {**envelope["payload"], "value": value}}))


def capture_retention_value(record: StoredRecord) -> JsonValue:
    """Project references only from the capture owner's declared durable slots."""
    references: list[JsonValue] = []

    def collect(value: JsonValue) -> JsonValue:
        references.append(value)
        return value

    map_capture_references(record, collect)
    return references


def capture_state_closure(records: Sequence[StoredRecord], *,
                          values: Sequence[JsonValue] = (), excluded_keys: Sequence[str] = (),
                          namespace: str | None = None, tenant_id: str | None = None,
                          require_bindings: bool = False) -> frozenset[str]:
    """Retain public captures and the exact dependency closure of live owners.

    Excluded public handles remain retained when another owner references them.
    Tombstones revoke a handle independently of any remaining references.
    """
    captures = {capture_record_key(record): record for record in records if record.kind in CAPTURE_RECORD_KINDS}
    scopes = {capture_record_scope(record) for record in captures.values()}
    for record in captures.values():
        if record.kind == "input_capture":
            capture_namespace, capture_tenant = capture_record_scope(record)
            validate_capture_record(record, namespace=capture_namespace, tenant_id=capture_tenant)
    if namespace is not None and tenant_id is not None:
        scopes.add((namespace, tenant_id))
    expired = {capture_record_value(record)["capture_key"] for key, record in captures.items()
               if key.startswith("v1/input-capture/expired/")}
    excluded = set(excluded_keys)
    pending: list[tuple[str, str | None]] = []
    for key in captures:
        if key.startswith(tuple("v1/input-capture/" + kind + "/" for kind in ("agent", "task", "graph", "template", "expired"))) and key not in excluded:
            pending.append((key, None))
    owners: list[JsonValue] = list(values)
    for record in records:
        if record.kind in CAPTURE_RECORD_KINDS or record.kind in {"evaluation_cleanup", "archive_content"}:
            continue
        owners.append(dict(record.data))
        if record.kind == "execution":
            payload = decode_envelope(record.data).value["payload"]
            identity = payload["fields"]["execution_id"]
            for namespace, tenant_id in scopes:
                if record.key == record_key(namespace, tenant_id, "execution", "execution", (identity,)):
                    key = input_capture_key(namespace, tenant_id, "invocation", identity)
                    if key in captures:
                        pending.append((key, None))
        elif record.kind in {"task_admission", "task_submission_payload"}:
            from .codec import _decode_enveloped_domain
            from ..task import TaskGraphAdmission, TaskGraphSubmission
            admission = (_decode_enveloped_domain(record.data, TaskGraphAdmission) if record.kind == "task_admission"
                         else _decode_enveloped_domain(record.data, TaskGraphSubmission).admission)
            for namespace, tenant_id in scopes:
                if record.key != record_key(namespace, tenant_id, "task", record.kind, (admission.graph_id,)):
                    continue
                key = input_capture_key(namespace, tenant_id, "declaration", admission.graph_id)
                if key in captures:
                    pending.append((key, None))
                key = task_graph_binding_capture_key(namespace, tenant_id, admission.graph_id, admission.initial_request_digest)
                if key in captures:
                    pending.append((key, None))
                elif require_bindings:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    for value in owners:
        pending.extend(iter_input_capture_dependencies(value, namespace="", tenant_id="", validate_scope=False))
    retained = set()
    while pending:
        key, digest = pending.pop()
        if key in expired:
            key, digest = input_capture_expiry_key(key), None
        record = captures.get(key)
        if record is None or (digest is not None and capture_record_digest(record) != digest):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if key in retained:
            continue
        retained.add(key)
        value = capture_retention_value(record)
        pending.extend(iter_input_capture_dependencies(value, namespace="", tenant_id="", validate_scope=False))
    return frozenset(retained)


def task_binding_content_reference(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, Mapping) or "$task_graph_binding" not in value:
        return None
    if set(value) != {"$task_graph_binding", "archive_scope", "content_scope"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    scope, content_scope, identity = value["archive_scope"], value["content_scope"], value["$task_graph_binding"]
    validate_archive_group_id(scope)
    decode_archive_scope(content_scope)
    validate_archive_content_identity(identity)
    return scope, content_scope, identity


async def prepare_task_binding_value(store: StateStore, manifest: Mapping[str, JsonValue], *,
                                     namespace: str, tenant_id: str) -> JsonValue:
    if not manifest["tasks"] and not manifest["expanders"]:
        return dict(manifest)
    raw = canonical_json_bytes({"tasks": manifest["tasks"], "expanders": manifest["expanders"]})
    opened = store.storage_group.opened_archive or await store.storage_group.open_archive(create=True)
    if opened is None:
        raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
    scope, content_store = opened
    archive = ContentArchive(content_store, scope, namespace=namespace, tenant_id=tenant_id, object_store=None)
    prepared = PreparedArchive()
    identity, = await archive.prepare_content(((TASK_BINDING_CODEC, raw),), prepared)
    await content_store.mutate(lambda transaction: archive.publish(transaction, prepared))
    return {**{key: value for key, value in manifest.items() if key not in {"tasks", "expanders"}},
            "declarations": {"$task_graph_binding": identity, "archive_scope": scope, "content_scope": archive.content_scope}}


async def read_task_binding_value(store: StateStore, value: JsonValue, *,
                                  namespace: str, tenant_id: str) -> JsonValue:
    reference = task_binding_content_reference(value.get("declarations")) if isinstance(value, Mapping) else None
    if reference is None:
        return value
    opened = store.storage_group.opened_archive or await store.storage_group.open_archive()
    if opened is None or reference[:2] != (opened[0], archive_content_scope(namespace, tenant_id)):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    archive = ContentArchive(opened[1], opened[0], namespace=namespace, tenant_id=tenant_id, object_store=None)
    raw, = await archive.read_contents((reference[2],), codec=TASK_BINDING_CODEC)
    return _expand_task_binding_value(value, raw)


def _expand_task_binding_value(value: Mapping[str, JsonValue], raw: bytes) -> JsonValue:
    # A retained declaration is the sole source, not an override for inline data.
    if "tasks" in value or "expanders" in value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        declarations = json.loads(raw)
        if not isinstance(declarations, Mapping) or set(declarations) != {"tasks", "expanders"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {**{key: item for key, item in value.items() if key != "declarations"}, **declarations}
    except (UnicodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def expand_snapshot_capture(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord], *,
                            namespace: str, tenant_id: str, reserve: Callable[[int], None] | None = None) -> StoredRecord:
    if record.kind == "input_capture":
        key, value = capture_record_key(record), capture_record_value(record)
        references = capture_binding_references(key, value)
        if not references:
            return record
        counts = _capture_binding_counts(key, value)
        raws = {}
        for reference in references:
            content = contents.get(reference)
            if (reference[1] != archive_content_scope(namespace, tenant_id) or content is None
                    or content.codec != BINDING_CODEC or content.identity != reference[2]
                    or content.payload is not None or content.body is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if reserve is not None:
                reserve(content.raw_size * counts[reference])
            raws[reference] = decode_payload_bytes(content.body, codec=content.compression,
                raw_digest=content.raw_digest, raw_size=content.raw_size)
        return capture_record(namespace, tenant_id, key, expand_capture_bindings(key, value, raws),
            storage_revision=record.storage_revision, source_digest=capture_record_digest(record))
    if record.kind != "task_graph_binding_capture":
        return record
    value = capture_record_value(record)
    reference = task_binding_content_reference(value.get("declarations")) if isinstance(value, Mapping) else None
    if reference is None:
        return record
    content = contents.get(reference)
    if (reference[1] != archive_content_scope(namespace, tenant_id) or content is None
            or content.codec != TASK_BINDING_CODEC or content.identity != reference[2]
            or content.payload is not None or content.body is None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if reserve is not None:
        reserve(content.raw_size)
    encoded = content.body
    raw = decode_payload_bytes(encoded, codec=content.compression,
                               raw_digest=content.raw_digest, raw_size=content.raw_size)
    return capture_record(namespace, tenant_id, capture_record_key(record), _expand_task_binding_value(value, raw),
                          storage_revision=record.storage_revision)


def _capture_binding_counts(key: str, value: JsonValue) -> Counter[tuple[str, str, str]]:
    counts = Counter()
    def count(binding: Mapping[str, JsonValue]) -> JsonValue:
        reference = binding_reference(binding)
        if reference is not None:
            counts[reference] += 1
        return binding
    map_capture_binding_slots(key, value, count)
    return counts


def snapshot_capture_size(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> int:
    if record.kind == "input_capture":
        total = 0
        for reference, count in _capture_binding_counts(capture_record_key(record), capture_record_value(record)).items():
            content = contents.get(reference)
            if (content is None or content.codec != BINDING_CODEC or content.identity != reference[2]
                    or content.payload is not None or content.body is None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            total += content.raw_size * count
        return total
    if record.kind != "task_graph_binding_capture":
        return 0
    value = capture_record_value(record)
    reference = task_binding_content_reference(value.get("declarations")) if isinstance(value, Mapping) else None
    if reference is None:
        return 0
    content = contents.get(reference)
    if content is None or content.codec != TASK_BINDING_CODEC or content.payload is not None or content.body is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content.raw_size


def validate_capture_record(record: StoredRecord, *, namespace: str, tenant_id: str) -> None:
    value = capture_record_value(record)
    try:
        expected = capture_record(namespace, tenant_id, capture_record_key(record), value, storage_revision=record.storage_revision,
                                  source_digest=decode_envelope(record.data).value["payload"].get("source_digest"))
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    capture_record_digest(record)
    if record != expected or capture_record_scope(record) != (namespace, tenant_id):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if record.kind == "input_capture":
        validate_input_capture_payload(capture_record_key(record), value, namespace=namespace, tenant_id=tenant_id)
    else:
        validate_task_binding_payload(capture_record_key(record), value, namespace=namespace, tenant_id=tenant_id)


def validate_task_binding_payload(key: str, value: JsonValue, *, namespace: str, tenant_id: str) -> None:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = value.get("format_version")
    if type(version) is not int or version < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if (set(value) != TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS
            or value.get("kind") != "task-graph-binding-capture"
            or value.get("namespace") != namespace or value.get("tenant_id") != tenant_id
            or not isinstance(value.get("graph_id"), str) or not isinstance(value.get("request_digest"), str)
            or task_graph_binding_capture_key(namespace, tenant_id, value["graph_id"], value["request_digest"]) != key):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    read_task_graph_binding_capture_declarations(value["tasks"], value["expanders"])


def input_capture_key(namespace: str, tenant_id: str, kind: str, identity: str) -> str:
    digest = canonical_sha256({"namespace": namespace, "tenant_id": tenant_id, "kind": kind, "identity": identity})
    return f"v1/input-capture/{kind}/{digest}"


def input_capture_expiry_key(key: str) -> str:
    return "v1/input-capture/expired/" + canonical_sha256(key)


def input_capture_state_dependency(
    reference: AgentInputCaptureRef | TaskInvocationInputRef | TaskGraphCaptureRef | TaskGraphTemplateRef,
) -> tuple[str, str]:
    kind = next(kind for cls, kind in _CAPTURE_TYPES.values() if isinstance(reference, cls))
    return input_capture_key(reference.namespace, reference.tenant_id, kind, reference.capture_id), reference.digest


def iter_input_capture_dependencies(value: JsonValue, *, namespace: str, tenant_id: str,
                                    validate_scope: bool = True) -> Iterator[tuple[str, str]]:
    if isinstance(value, list):
        for item in value:
            yield from iter_input_capture_dependencies(item, namespace=namespace, tenant_id=tenant_id, validate_scope=validate_scope)
    elif isinstance(value, Mapping):
        wire_id = value.get("$dataclass")
        if isinstance(wire_id, str) and wire_id in {"stored_payload", "runtime_payload_ref", "object_ref"}:
            return
        if isinstance(wire_id, str) and wire_id in _CAPTURE_TYPES:
            cls, _kind = _CAPTURE_TYPES[wire_id]
            reference = decode_domain(value, cls)
            if validate_scope and (reference.namespace != namespace or reference.tenant_id != tenant_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            yield input_capture_state_dependency(reference)
            return
        if wire_id == "task_dependency_capture":
            captured = decode_domain(value, TaskDependencyCapture)
            if captured.body_digest is not None:
                if validate_scope and (captured.source_ref.namespace != namespace or captured.source_ref.tenant_id != tenant_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                yield input_capture_key(captured.source_ref.namespace, captured.source_ref.tenant_id, "result", captured.body_digest), captured.body_digest
            return
        if wire_id == "evaluation_record":
            # Reservations own cleanup, not content that must already exist.
            fields = value.get("fields")
            if not isinstance(fields, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for reference in decode_domain(fields.get("owned_input_captures"), tuple[TaskInvocationInputRef, ...]):
                if validate_scope and (reference.namespace != namespace or reference.tenant_id != tenant_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            value = {name: item for name, item in fields.items() if name != "owned_input_captures"}
        for item in value.values():
            yield from iter_input_capture_dependencies(item, namespace=namespace, tenant_id=tenant_id, validate_scope=validate_scope)


def validate_input_capture_payload(key: str, value: JsonValue, *, namespace: str, tenant_id: str) -> None:
    if not isinstance(key, str) or not key.startswith("v1/input-capture/") or len(key.split("/")) != 4:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    kind = key.split("/")[2]
    if kind in {"agent", "task", "graph", "template"}:
        if not isinstance(value, Mapping) or value.get("namespace") != namespace or value.get("tenant_id") != tenant_id or value.get("kind") != kind or value.get("version") != 1:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if kind == "agent":
            version = value.get("request_identity_version", 1)
            if type(version) is not int or version not in (1, 2):
                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    elif kind == "expired":
        if (not isinstance(value, Mapping) or not {"namespace", "tenant_id", "capture_key", "expired_at"}.issubset(value)
                or set(value) - {"namespace", "tenant_id", "capture_key", "expired_at", "principal", "cleanup_objects"}
                or value.get("namespace") != namespace or value.get("tenant_id") != tenant_id
                or not isinstance(value.get("capture_key"), str)
                or not value["capture_key"].startswith("v1/input-capture/")
                or value["capture_key"].startswith("v1/input-capture/expired/")
                or input_capture_expiry_key(value["capture_key"]) != key):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            if datetime.fromisoformat(value["expired_at"]).tzinfo is None:
                raise ValueError("capture expiry must be timezone-aware")
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    elif kind not in {"result", "invocation", "declaration"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


from ._runtime_identity import task_graph_binding_capture_key
from ._task_graph_binding_capture import (
    TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION,
    TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS,
    read_task_graph_binding_capture_declarations,
    task_declaration_identity,
    task_expander_declaration_identity,
)

__all__ = ['CAPTURE_RECORD_KINDS', 'TASK_BINDING_CODEC', 'capture_record', 'capture_record_key', 'capture_record_value', 'capture_record_digest', 'prepare_capture_value', 'restore_capture_value', 'capture_record_scope', 'capture_retention_value', 'map_capture_references', 'capture_state_closure', 'read_capture_record', 'read_capture_value', 'publish_capture_record', 'task_binding_content_reference', 'prepare_task_binding_value', 'read_task_binding_value', 'expand_snapshot_capture', 'snapshot_capture_size', 'validate_capture_record', 'validate_task_binding_payload', 'TASK_GRAPH_BINDING_CAPTURE_FORMAT_VERSION', 'TASK_GRAPH_BINDING_CAPTURE_MANIFEST_KEYS', 'input_capture_expiry_key', 'input_capture_key', 'input_capture_state_dependency', 'iter_input_capture_dependencies', 'read_task_graph_binding_capture_declarations', 'task_declaration_identity', 'task_expander_declaration_identity', 'task_graph_binding_capture_key', 'validate_input_capture_payload']
