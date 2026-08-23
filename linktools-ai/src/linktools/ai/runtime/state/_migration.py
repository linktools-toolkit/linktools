#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-time normalization of reconstructable pre-composition Runtime V1 state.

Normal Runtime decoding remains exact.  This module recognizes only frozen,
known V1 shapes written before AgentDefinition/AgentBinding identity was split,
reconstructs their current exact binding through AgentCompiler, and rewrites
those records to the single current V1 shape before normal Runtime reads begin.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, get_type_hints

from pydantic import BaseModel

from ...agent import AgentBinding, AgentCatalog, AgentCompiler, AgentBindingSnapshot
from ...capability import RuntimeCapability
from ...core import JsonValue
from ...errors import AIError, ErrorCode
from ...spec import AgentSpec, AgentUsageLimits
from ._codec import (
    _CURRENT_CODEC,
    _decode_domain,
    _decode_enveloped_domain,
    encode_domain,
    encode_envelope,
    parse_envelope,
    wire_type_id,
)
from ._contracts import (
    ContextProjection,
    ExecutionRecord,
    RecoveryAdmissionRecord,
    RecoveryExecutionInput,
    SessionRecord,
)
from ._plan import RuntimeDomain
from ._repositories import (
    ExecutionRepositoryImpl,
    RecoveryCheckpointRepositoryImpl,
    SessionRepositoryImpl,
    _domain_data,
)
from ._root import RuntimeState
from ._store import RecordQuery, StateStore, StoredRecord

_PAGE_SIZE = 128
_CURRENT_BINDING_FIELDS = frozenset(
    {
        "version",
        "agent_spec",
        "agent_digest",
        "output_type_module",
        "output_type_qualname",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
    }
)
_LEGACY_BINDING_FIELDS = _CURRENT_BINDING_FIELDS - {"agent_digest"}
_LEGACY_AGENT_SPEC_FIELDS = frozenset(
    {
        "id",
        "revision",
        "model",
        "system_prompt",
        "instructions",
        "allow_tools",
        "metadata",
        "usage_limits",
    }
)
_LEGACY_SESSION_FIELDS = frozenset(
    {
        "session_id",
        "tenant_id",
        "owner_principal_id",
        "binding_digest",
        "status",
        "revision",
        "resource_generation",
        "cwd",
        "metadata",
        "created_at",
        "updated_at",
        "closed_at",
        "active_execution_id",
        "continuation",
        "history_quality",
        "history_id",
    }
)
_CURRENT_SESSION_FIELDS = (_LEGACY_SESSION_FIELDS - {"binding_digest"}) | {"agent_digest"}
_LEGACY_PROJECTION_FIELDS = frozenset({"binding_digest", "items", "digest"})
_CURRENT_PROJECTION_FIELDS = frozenset({"agent_digest", "items", "digest"})


async def migrate_v1_agent_identity_state(
    state: RuntimeState,
    catalog: AgentCatalog,
    compiler: AgentCompiler,
    *,
    tenant_id: str,
) -> int:
    """Normalize reconstructable legacy V1 Agent identity records in place.

    The operation is idempotent.  Current records are never rewritten.  Known
    legacy records are accepted only when their exact old shape is present and
    an exact current AgentBinding can be reconstructed.  Unknown/partial shapes
    continue to fail closed.
    """
    if not state.ready:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    sessions = state.conversation.sessions
    executions = state.execution.executions
    recovery = state.recovery.checkpoints
    if not isinstance(sessions, SessionRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not isinstance(executions, ExecutionRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    if not isinstance(recovery, RecoveryCheckpointRepositoryImpl):
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    legacy_bindings: dict[str, AgentBinding] = {}
    execution_records = await _records(executions, "execution")
    recovery_records = await _records(recovery, "recovery_admission")

    for record in execution_records:
        _collect_execution_binding(record, compiler, catalog, legacy_bindings)
    for record in recovery_records:
        _collect_recovery_binding(record, compiler, catalog, legacy_bindings)

    migrated = 0
    for record in await _records(sessions, "session"):
        current = _migrate_session_record(record, sessions, catalog, legacy_bindings)
        if current is not None:
            await _replace_data(sessions.state_store, record, _domain_data(current))
            migrated += 1

    for domain in (
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.RECOVERY,
    ):
        archive = state.steps.read_store(domain)
        history = archive.transcript_repository
        records = await _records(history, "context_projection")
        for record in records:
            data = _migrate_projection_data(record, legacy_bindings)
            if data is not None:
                await _replace_data(history._store, record, data)
                migrated += 1

    for record in execution_records:
        data = _migrate_execution_data(record, compiler, catalog, legacy_bindings)
        if data is not None:
            await _replace_data(executions.state_store, record, data)
            migrated += 1
    for record in recovery_records:
        data = _migrate_recovery_data(record, compiler, catalog, legacy_bindings)
        if data is not None:
            await _replace_data(recovery.state_store, record, data)
            migrated += 1
    return migrated


async def _records(repository: object, kind: str) -> tuple[StoredRecord, ...]:
    store = repository._store
    partition = repository._partition(kind)
    values: list[StoredRecord] = []
    after_sort_key: str | None = None
    after_key_digest: bytes | None = None
    while True:
        page = await store.read(
            lambda transaction, sort_key=after_sort_key, key_digest=after_key_digest: transaction.list_records(
                RecordQuery(
                    partition_digest=partition,
                    kind=kind,
                    after_sort_key=sort_key,
                    after_key_digest=key_digest,
                    limit=_PAGE_SIZE,
                )
            )
        )
        if not page:
            return tuple(values)
        values.extend(page)
        if len(page) < _PAGE_SIZE:
            return tuple(values)
        last = page[-1]
        after_sort_key = last.sort_key
        after_key_digest = last.key_digest


async def _replace_data(
    store: StateStore,
    current: StoredRecord,
    data: Mapping[str, JsonValue],
) -> None:
    candidate = replace(
        current,
        data=dict(data),
        storage_version=current.storage_version + 1,
    )
    replaced = await store.mutate(
        lambda transaction: transaction.replace_record(
            candidate,
            expected_storage_version=current.storage_version,
        )
    )
    if not replaced:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _domain_fields(
    data: Mapping[str, JsonValue],
    *,
    type_name: str,
    wire_id: str,
) -> Mapping[str, object]:
    envelope = parse_envelope(data)
    if envelope.version != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    value = envelope.value
    if set(value) != {"type", "payload"} or value.get("type") != type_name:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(payload) != {"$dataclass", "fields"} or payload.get("$dataclass") != wire_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return fields


def _collect_execution_binding(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    mappings: dict[str, AgentBinding],
) -> None:
    fields = _domain_fields(
        record.data,
        type_name="execution_record",
        wire_id="execution_record",
    )
    binding = fields.get("binding")
    if _is_current_binding(binding):
        return
    if binding is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={"record": "execution", "reason": "missing_exact_binding"},
        )
    migrated = _migrate_legacy_binding(binding, compiler)
    _remember_binding(mappings, migrated[0], migrated[1], catalog)


def _collect_recovery_binding(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    mappings: dict[str, AgentBinding],
) -> None:
    fields = _domain_fields(
        record.data,
        type_name="recovery_admission",
        wire_id="recovery_admission",
    )
    raw_input = fields.get("input")
    input_fields = _nested_dataclass_fields(raw_input, "recovery_execution_input")
    binding = input_fields.get("binding")
    if _is_current_binding(binding):
        return
    if binding is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={"record": "recovery_admission", "reason": "missing_exact_binding"},
        )
    migrated = _migrate_legacy_binding(binding, compiler)
    _remember_binding(mappings, migrated[0], migrated[1], catalog)


def _remember_binding(
    mappings: dict[str, AgentBinding],
    legacy_digest: str,
    binding: AgentBinding,
    catalog: AgentCatalog,
) -> None:
    previous = mappings.get(legacy_digest)
    if previous is not None and previous.snapshot != binding.snapshot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    mappings[legacy_digest] = binding
    catalog.register_definition(binding.definition)
    catalog.register_binding(binding)


def _migrate_execution_data(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    mappings: dict[str, AgentBinding],
) -> Mapping[str, JsonValue] | None:
    fields = _domain_fields(
        record.data,
        type_name="execution_record",
        wire_id="execution_record",
    )
    binding = fields.get("binding")
    if _is_current_binding(binding):
        _decode_enveloped_domain(record.data, ExecutionRecord)
        return None
    if binding is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    _remember_binding(mappings, legacy_digest, migrated, catalog)
    if _decode_domain(fields.get("binding_digest"), str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    current_fields = dict(fields)
    current_fields["binding_digest"] = encode_domain(migrated.digest)
    current_fields["binding"] = migrated.snapshot.to_payload()
    data = _rebuild_data("execution_record", current_fields)
    value = _decode_enveloped_domain(data, ExecutionRecord)
    return _domain_data(value)


def _migrate_recovery_data(
    record: StoredRecord,
    compiler: AgentCompiler,
    catalog: AgentCatalog,
    mappings: dict[str, AgentBinding],
) -> Mapping[str, JsonValue] | None:
    fields = _domain_fields(
        record.data,
        type_name="recovery_admission",
        wire_id="recovery_admission",
    )
    raw_input = fields.get("input")
    input_fields = _nested_dataclass_fields(raw_input, "recovery_execution_input")
    binding = input_fields.get("binding")
    if _is_current_binding(binding):
        _decode_enveloped_domain(record.data, RecoveryAdmissionRecord)
        return None
    if binding is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    legacy_digest, migrated = _migrate_legacy_binding(binding, compiler)
    _remember_binding(mappings, legacy_digest, migrated, catalog)
    if _decode_domain(input_fields.get("binding_digest"), str, _CURRENT_CODEC) != legacy_digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    next_input_fields = dict(input_fields)
    next_input_fields["binding_digest"] = encode_domain(migrated.digest)
    next_input_fields["binding"] = migrated.snapshot.to_payload()
    next_input = {"$dataclass": "recovery_execution_input", "fields": next_input_fields}
    next_fields = dict(fields)
    next_fields["input"] = next_input
    data = _rebuild_data("recovery_admission", next_fields)
    value = _decode_enveloped_domain(data, RecoveryAdmissionRecord)
    return _domain_data(value)


def _migrate_session_record(
    record: StoredRecord,
    repository: SessionRepositoryImpl,
    catalog: AgentCatalog,
    mappings: Mapping[str, AgentBinding],
) -> SessionRecord | None:
    fields = _domain_fields(
        record.data,
        type_name="session_record",
        wire_id="session_record",
    )
    keys = frozenset(fields)
    if keys == _CURRENT_SESSION_FIELDS:
        _decode_enveloped_domain(record.data, SessionRecord)
        return None
    if keys != _LEGACY_SESSION_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    hints = get_type_hints(SessionRecord)
    decoded: dict[str, object] = {}
    for name in _CURRENT_SESSION_FIELDS - {"agent_digest"}:
        decoded[name] = _decode_domain(fields[name], hints[name], _CURRENT_CODEC)
    legacy_digest = _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC)
    if not isinstance(legacy_digest, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    migrated = mappings.get(legacy_digest)
    if migrated is not None:
        agent_digest = migrated.definition.digest
    else:
        if decoded.get("continuation") is not None or decoded.get("active_execution_id") is not None:
            raise AIError(
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                safe_details={"record": "session", "reason": "binding_evidence_unavailable"},
            )
        metadata = decoded.get("metadata")
        if not isinstance(metadata, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        agent_id = metadata.get("linktools.ai.agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise AIError(
                ErrorCode.AGENT_DEFINITION_UNAVAILABLE,
                safe_details={"session_id": decoded.get("session_id")},
            )
        agent_digest = catalog.root_definition(agent_id).digest
    decoded["agent_digest"] = agent_digest
    try:
        value = SessionRecord(**decoded)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if value.tenant_id != repository._tenant_id:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
    return value


def _migrate_projection_data(
    record: StoredRecord,
    mappings: Mapping[str, AgentBinding],
) -> Mapping[str, JsonValue] | None:
    fields = _domain_fields(
        record.data,
        type_name="context_projection",
        wire_id="context_projection",
    )
    keys = frozenset(fields)
    if keys == _CURRENT_PROJECTION_FIELDS:
        _decode_enveloped_domain(record.data, ContextProjection)
        return None
    if keys != _LEGACY_PROJECTION_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest = _decode_domain(fields["binding_digest"], str, _CURRENT_CODEC)
    binding = mappings.get(legacy_digest)
    if binding is None:
        raise AIError(
            ErrorCode.STORAGE_VERSION_UNSUPPORTED,
            safe_details={"record": "context_projection", "reason": "binding_evidence_unavailable"},
        )
    current_fields = {
        "agent_digest": encode_domain(binding.definition.digest),
        "items": fields["items"],
        "digest": fields["digest"],
    }
    data = _rebuild_data("context_projection", current_fields)
    value = _decode_enveloped_domain(data, ContextProjection)
    return encode_envelope(
        {"type": wire_type_id(value), "payload": encode_domain(value)}
    )


def _migrate_legacy_binding(
    value: object,
    compiler: AgentCompiler,
) -> tuple[str, AgentBinding]:
    if not isinstance(value, Mapping) or frozenset(value) != _LEGACY_BINDING_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value.get("version") != 1 or isinstance(value.get("version"), bool):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    legacy_digest = _digest(value.get("binding_digest"))
    spec = _legacy_agent_spec(value.get("agent_spec"))
    descriptors = value.get("local_runtime_capability_descriptors")
    if not isinstance(descriptors, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        capabilities = tuple(
            RuntimeCapability.restore(_json_mapping(descriptor))
            for descriptor in descriptors
        )
        definition = compiler.compile(spec, capabilities=capabilities)
        output_type = _output_type(
            _string(value.get("output_type_module")),
            _string(value.get("output_type_qualname")),
        )
        binding = compiler.bind(definition, output=output_type)
    except AIError as error:
        if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
            raise
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
    snapshot = binding.snapshot
    if (
        snapshot.output_type_module != value.get("output_type_module")
        or snapshot.output_type_qualname != value.get("output_type_qualname")
        or snapshot.output_schema_id != value.get("output_schema_id")
        or snapshot.output_schema_revision != value.get("output_schema_revision")
        or snapshot.output_schema_fingerprint != value.get("output_schema_fingerprint")
    ):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return legacy_digest, binding


def _legacy_agent_spec(value: object) -> AgentSpec:
    if not isinstance(value, Mapping) or frozenset(value) != _LEGACY_AGENT_SPEC_FIELDS:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    metadata = value.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    instructions = value.get("instructions")
    allow_tools = value.get("allow_tools")
    if not isinstance(instructions, list) or not isinstance(allow_tools, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    limits = _usage_limits(value.get("usage_limits"))
    try:
        return AgentSpec(
            id=_string(value.get("id")),
            revision=_positive_int(value.get("revision")),
            model=_string(value.get("model")),
            system_prompt=_string(value.get("system_prompt"), allow_empty=True),
            instructions=tuple(_string(item, allow_empty=True) for item in instructions),
            allow_tools=tuple(_string(item) for item in allow_tools),
            usage_limits=limits,
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _usage_limits(value: object) -> AgentUsageLimits | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    names = {
        "model_requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    if set(value) != names:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    kwargs: dict[str, int | None] = {}
    for name in names:
        item = value[name]
        if item is not None and (
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kwargs[name] = item
    try:
        return AgentUsageLimits(**kwargs)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _output_type(module_name: str, qualname: str) -> type[BaseModel]:
    if module_name == "__main__" or "<locals>" in qualname:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    try:
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except Exception as error:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
    if not isinstance(target, type) or not issubclass(target, BaseModel):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return target


def _nested_dataclass_fields(value: object, wire_id: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(value) != {"$dataclass", "fields"} or value.get("$dataclass") != wire_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    fields = value.get("fields")
    if not isinstance(fields, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return fields


def _rebuild_data(wire_id: str, fields: Mapping[str, object]) -> Mapping[str, JsonValue]:
    return encode_envelope(
        {
            "type": wire_id,
            "payload": {"$dataclass": wire_id, "fields": dict(fields)},
        }
    )


def _is_current_binding(value: object) -> bool:
    if not isinstance(value, Mapping) or frozenset(value) != _CURRENT_BINDING_FIELDS:
        return False
    AgentBindingSnapshot.from_payload(value)
    return True


def _json_mapping(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value  # type: ignore[return-value]


def _string(value: object, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _digest(value: object) -> str:
    text = _string(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return text


__all__ = ["migrate_v1_agent_identity_state"]
