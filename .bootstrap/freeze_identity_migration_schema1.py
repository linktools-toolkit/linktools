#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_migration.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "_POST_COMPOSITION_EXECUTION_V1_FIELDS" in text:
        print("identity migration schema boundary already frozen")
        return

    text = replace_once(
        text,
        "from dataclasses import fields as dataclass_fields, replace\nfrom typing import get_type_hints\n",
        "from dataclasses import replace\n",
        "remove current dataclass reflection imports",
    )
    text = replace_once(
        text,
        "    _EXECUTION_V1_FIELDS,\n    _RECOVERY_EXECUTION_V1_FIELDS,\n    _dataclass_persistence_contract,\n",
        "",
        "remove current codec private shape imports",
    )

    old = '''_CURRENT_SESSION_FIELDS = frozenset(\n    field.name for field in dataclass_fields(SessionRecord)\n)\n_LEGACY_SESSION_FIELDS = (\n    _CURRENT_SESSION_FIELDS - {"agent_digest"}\n) | {"binding_digest"}\n_CURRENT_PROJECTION_FIELDS = frozenset(\n    field.name for field in dataclass_fields(ContextProjection)\n)\n_LEGACY_PROJECTION_FIELDS = (\n    _CURRENT_PROJECTION_FIELDS - {"agent_digest"}\n) | {"binding_digest"}\n_CURRENT_RECOVERY_INPUT_FIELDS = _RECOVERY_EXECUTION_V1_FIELDS\n_LEGACY_RECOVERY_INPUT_FIELDS = _CURRENT_RECOVERY_INPUT_FIELDS | {"agent_id"}\n'''
    new = '''# Frozen shapes owned by this one-time semantic migration.  They describe\n# the exact unversioned V1 wire before/after the Agent identity split and must\n# never be derived from the current Runtime dataclasses or codec implementation.\n_POST_COMPOSITION_EXECUTION_V1_FIELDS = frozenset(\n    {\n        "execution_id",\n        "tenant_id",\n        "session_id",\n        "binding_digest",\n        "parent_execution_id",\n        "root_execution_id",\n        "source_execution_id",\n        "base_execution_id",\n        "lineage_kind",\n        "status",\n        "revision",\n        "event_sequence",\n        "agent_run_sequence",\n        "error_code",\n        "safe_error_details",\n        "created_at",\n        "updated_at",\n        "planning",\n        "thinking",\n        "binding",\n        "memory_scope",\n        "conversation_step_run_id",\n        "result",\n    }\n)\n_POST_COMPOSITION_SESSION_V1_FIELDS = frozenset(\n    {\n        "session_id",\n        "tenant_id",\n        "owner_principal_id",\n        "agent_digest",\n        "status",\n        "revision",\n        "resource_generation",\n        "cwd",\n        "metadata",\n        "created_at",\n        "updated_at",\n        "closed_at",\n        "active_execution_id",\n        "continuation",\n        "history_quality",\n        "history_id",\n    }\n)\n_PRE_COMPOSITION_SESSION_V1_FIELDS = (\n    _POST_COMPOSITION_SESSION_V1_FIELDS - {"agent_digest"}\n) | {"binding_digest"}\n_POST_COMPOSITION_PROJECTION_V1_FIELDS = frozenset(\n    {"agent_digest", "items", "digest"}\n)\n_PRE_COMPOSITION_PROJECTION_V1_FIELDS = (\n    _POST_COMPOSITION_PROJECTION_V1_FIELDS - {"agent_digest"}\n) | {"binding_digest"}\n_POST_COMPOSITION_RECOVERY_ADMISSION_V1_FIELDS = frozenset(\n    {"execution_id", "tenant_id", "input", "created_at"}\n)\n_POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS = frozenset(\n    {\n        "user_prompt",\n        "principal_id",\n        "principal_kind",\n        "session_id",\n        "memory_scope",\n        "binding_digest",\n        "lineage_kind",\n        "parent_execution_id",\n        "root_execution_id",\n        "source_execution_id",\n        "base_execution_id",\n        "conversation_step_run_id",\n        "idempotency",\n        "planning",\n        "thinking",\n        "binding",\n    }\n)\n_PRE_COMPOSITION_RECOVERY_INPUT_V1_FIELDS = (\n    _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS | {"agent_id"}\n)\n'''
    text = replace_once(text, old, new, "freeze historical identity shapes")

    text = text.replace(
        "frozenset(fields) != _EXECUTION_V1_FIELDS",
        "frozenset(fields) != _POST_COMPOSITION_EXECUTION_V1_FIELDS",
    )
    text = text.replace(
        "keys == _CURRENT_RECOVERY_INPUT_FIELDS",
        "keys == _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS",
    )
    text = text.replace(
        "keys != _LEGACY_RECOVERY_INPUT_FIELDS",
        "keys != _PRE_COMPOSITION_RECOVERY_INPUT_V1_FIELDS",
    )
    text = text.replace(
        "frozenset(next_input_fields) != _CURRENT_RECOVERY_INPUT_FIELDS",
        "frozenset(next_input_fields) != _POST_COMPOSITION_RECOVERY_INPUT_V1_FIELDS",
    )

    old = '''    fields = _domain_fields(\n        record.data,\n        type_name="recovery_admission",\n        wire_id="recovery_admission",\n    )\n    input_fields = _nested_dataclass_fields(\n'''
    new = '''    fields = _domain_fields(\n        record.data,\n        type_name="recovery_admission",\n        wire_id="recovery_admission",\n    )\n    if frozenset(fields) != _POST_COMPOSITION_RECOVERY_ADMISSION_V1_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    input_fields = _nested_dataclass_fields(\n'''
    if text.count(old) != 2:
        raise RuntimeError(
            f"recovery outer frozen shape: expected two matches, found {text.count(old)}"
        )
    text = text.replace(old, new)

    old = '''    current_fields = dict(fields)\n    current_fields["binding_digest"] = encode_domain(migrated.digest)\n    current_fields["binding"] = migrated.snapshot.to_payload()\n    data = _rebuild_data("execution_record", current_fields)\n    value = _decode_enveloped_domain(data, ExecutionRecord)\n'''
    new = '''    current_fields = dict(fields)\n    current_fields["binding_digest"] = encode_domain(migrated.digest)\n    current_fields["binding"] = migrated.snapshot.to_payload()\n    data = _schema1_data("execution_record", current_fields)\n    value = _decode_enveloped_domain(data, ExecutionRecord)\n'''
    text = replace_once(text, old, new, "execution semantic migration emits schema1")

    old = '''    next_fields = dict(fields)\n    next_fields["input"] = {\n        "$dataclass": "recovery_execution_input",\n        "fields": next_input_fields,\n    }\n    data = _rebuild_data("recovery_admission", next_fields)\n    value = _decode_enveloped_domain(data, RecoveryAdmissionRecord)\n'''
    new = '''    next_fields = dict(fields)\n    next_fields["input"] = _schema1_dataclass(\n        "recovery_execution_input", next_input_fields\n    )\n    data = _schema1_data("recovery_admission", next_fields)\n    value = _decode_enveloped_domain(data, RecoveryAdmissionRecord)\n'''
    text = replace_once(text, old, new, "recovery semantic migration emits schema1")

    start = text.index("def _migrate_session_record(")
    end = text.index("\n\ndef _migrate_projection_data(", start)
    session = '''def _migrate_session_record(\n    record: StoredRecord,\n    repository: SessionRepositoryImpl,\n    agents: Mapping[str, str],\n) -> SessionRecord | None:\n    current = _decode_current_record(record.data, SessionRecord)\n    if isinstance(current, SessionRecord):\n        if current.tenant_id != repository._tenant_id:\n            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)\n        return None\n    fields = _domain_fields(\n        record.data,\n        type_name="session_record",\n        wire_id="session_record",\n    )\n    keys = frozenset(fields)\n    if keys == _POST_COMPOSITION_SESSION_V1_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    if keys != _PRE_COMPOSITION_SESSION_V1_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    legacy_digest = _decode_domain(\n        fields["binding_digest"], str, _CURRENT_CODEC\n    )\n    agent_digest = agents.get(legacy_digest)\n    if agent_digest is None:\n        session_id = _decode_domain(fields["session_id"], str, _CURRENT_CODEC)\n        raise AIError(\n            ErrorCode.STORAGE_VERSION_UNSUPPORTED,\n            safe_details={\n                "record": "session",\n                "reason": "binding_evidence_unavailable",\n                "session_id": session_id,\n            },\n        )\n    current_fields = {\n        key: value\n        for key, value in fields.items()\n        if key != "binding_digest"\n    }\n    current_fields["agent_digest"] = encode_domain(agent_digest)\n    if frozenset(current_fields) != _POST_COMPOSITION_SESSION_V1_FIELDS:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    value = _decode_enveloped_domain(\n        _schema1_data("session_record", current_fields),\n        SessionRecord,\n    )\n    if value.tenant_id != repository._tenant_id:\n        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)\n    return value\n'''
    text = text[:start] + session + text[end:]

    text = text.replace(
        "keys == _CURRENT_PROJECTION_FIELDS",
        "keys == _POST_COMPOSITION_PROJECTION_V1_FIELDS",
    )
    text = text.replace(
        "keys != _LEGACY_PROJECTION_FIELDS",
        "keys != _PRE_COMPOSITION_PROJECTION_V1_FIELDS",
    )
    text = replace_once(
        text,
        '''    data = _rebuild_data("context_projection", current_fields)\n    value = _decode_enveloped_domain(data, ContextProjection)\n''',
        '''    data = _schema1_data("context_projection", current_fields)\n    value = _decode_enveloped_domain(data, ContextProjection)\n''',
        "projection semantic migration emits schema1",
    )

    old = '''def _rebuild_data(\n    wire_id: str,\n    fields: Mapping[str, object],\n) -> Mapping[str, JsonValue]:\n    contract = _dataclass_persistence_contract(wire_id, _CURRENT_CODEC)\n    return encode_envelope(\n        {\n            "type": wire_id,\n            "payload": {\n                "$dataclass": wire_id,\n                "schema": contract.current_revision,\n                "fields": dict(fields),\n            },\n        }\n    )\n'''
    new = '''def _schema1_dataclass(\n    wire_id: str,\n    fields: Mapping[str, object],\n) -> dict[str, JsonValue]:\n    return {\n        "$dataclass": wire_id,\n        "schema": 1,\n        "fields": dict(fields),\n    }  # type: ignore[return-value]\n\n\ndef _schema1_data(\n    wire_id: str,\n    fields: Mapping[str, object],\n) -> Mapping[str, JsonValue]:\n    return encode_envelope(\n        {\n            "type": wire_id,\n            "payload": _schema1_dataclass(wire_id, fields),\n        }\n    )\n'''
    text = replace_once(text, old, new, "freeze semantic migration output revision")

    forbidden = (
        "    _EXECUTION_V1_FIELDS,\n",
        "    _RECOVERY_EXECUTION_V1_FIELDS,\n",
        "_CURRENT_SESSION_FIELDS",
        "_LEGACY_SESSION_FIELDS",
        "_CURRENT_PROJECTION_FIELDS",
        "_LEGACY_PROJECTION_FIELDS",
        "_CURRENT_RECOVERY_INPUT_FIELDS",
        "_LEGACY_RECOVERY_INPUT_FIELDS",
        "_rebuild_data(",
        "field.name for field in dataclass_fields(",
        "get_type_hints(",
        "_dataclass_persistence_contract(",
    )
    stale = [name for name in forbidden if name in text]
    if stale:
        raise RuntimeError(f"stale migration coupling remains: {stale}")

    PATH.write_text(text, encoding="utf-8")
    print("identity migration owns frozen V1 shapes and emits schema1")


if __name__ == "__main__":
    main()
