from pathlib import Path
from dataclasses import is_dataclass
import re

from linktools.ai.runtime.state import _codec as codec_module

path = Path('linktools-ai/src/linktools/ai/runtime/state/_codec.py')
text = path.read_text(encoding='utf-8')

current_fingerprints: dict[str, str] = {}
for wire_id, target in codec_module._V1_WIRE_TYPES:
    if not is_dataclass(target):
        continue
    try:
        current_fingerprints[wire_id] = codec_module._dataclass_schema_fingerprint(
            target, codec_module._V1_CODEC
        )
    except (TypeError, ValueError):
        continue

start = text.index('_EXECUTION_V1_LEGACY_FIELDS = frozenset(')
end = text.index('\n\ndef _encode_v1_task_node(', start)
replacement = '''_EXECUTION_V1_FIELDS = frozenset(\n    {\n        "execution_id", "tenant_id", "session_id", "binding_digest",\n        "parent_execution_id", "root_execution_id", "source_execution_id",\n        "base_execution_id", "lineage_kind", "status", "revision",\n        "event_sequence", "agent_run_sequence", "error_code",\n        "safe_error_details", "created_at", "updated_at", "planning",\n        "thinking", "binding", "memory_scope", "conversation_step_run_id",\n        "result",\n    }\n)\n_RECOVERY_EXECUTION_V1_FIELDS = frozenset(\n    {\n        "user_prompt", "principal_id", "principal_kind", "session_id",\n        "memory_scope", "binding_digest", "lineage_kind",\n        "parent_execution_id", "root_execution_id", "source_execution_id",\n        "base_execution_id", "conversation_step_run_id", "idempotency",\n        "planning", "thinking", "binding",\n    }\n)\n\n\ndef _encode_v1_exact_binding_record(\n    value: object,\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n) -> Mapping[str, JsonValue]:\n    actual_fields = frozenset(field.name for field in fields(value))\n    if actual_fields != expected_fields:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    planning = attrgetter("planning")(value)\n    thinking = attrgetter("thinking")(value)\n    binding = attrgetter("binding")(value)\n    if (\n        not isinstance(planning, bool)\n        or not isinstance(thinking, bool)\n        or not isinstance(binding, AgentBindingSnapshot)\n    ):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    return {\n        name: (\n            binding.to_payload()\n            if name == "binding"\n            else _encode_domain(attrgetter(name)(value), codec)\n        )\n        for name in expected_fields\n    }\n\n\ndef _decode_v1_exact_binding_record_fields(\n    raw_fields: Mapping[str, object],\n    target: type[object],\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n) -> dict[str, object]:\n    _require_exact_keys(raw_fields, expected_fields)\n    hints = get_type_hints(target)\n    return {\n        name: (\n            AgentBindingSnapshot.from_payload(raw_fields[name])\n            if name == "binding"\n            else _decode_domain(raw_fields[name], hints[name], codec)\n        )\n        for name in expected_fields\n    }\n\n\ndef _encode_v1_execution_record(\n    value: object,\n    codec: "_VersionCodec",\n) -> Mapping[str, JsonValue]:\n    if not isinstance(value, ExecutionRecord):\n        raise TypeError("V1 execution_record encoder received the wrong type")\n    return _encode_v1_exact_binding_record(value, _EXECUTION_V1_FIELDS, codec)\n\n\ndef _decode_v1_execution_record(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n) -> ExecutionRecord:\n    decoded = _decode_v1_exact_binding_record_fields(\n        raw_fields, ExecutionRecord, _EXECUTION_V1_FIELDS, codec\n    )\n    try:\n        return ExecutionRecord(**decoded)\n    except (TypeError, ValueError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n\n\ndef _encode_v1_recovery_execution_input(\n    value: object,\n    codec: "_VersionCodec",\n) -> Mapping[str, JsonValue]:\n    if not isinstance(value, RecoveryExecutionInput):\n        raise TypeError("V1 recovery_execution_input encoder received the wrong type")\n    return _encode_v1_exact_binding_record(\n        value, _RECOVERY_EXECUTION_V1_FIELDS, codec\n    )\n\n\ndef _decode_v1_recovery_execution_input(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n) -> RecoveryExecutionInput:\n    decoded = _decode_v1_exact_binding_record_fields(\n        raw_fields, RecoveryExecutionInput, _RECOVERY_EXECUTION_V1_FIELDS, codec\n    )\n    try:\n        return RecoveryExecutionInput(**decoded)\n    except (TypeError, ValueError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n'''
text = text[:start] + replacement + text[end:]

changed: list[tuple[str, str, str]] = []
for wire_id, fingerprint in current_fingerprints.items():
    pattern = rf'("{re.escape(wire_id)}"\s*:\s*")([0-9a-f]{{64}})(")'
    match = re.search(pattern, text)
    if match is None or match.group(2) == fingerprint:
        continue
    old = match.group(2)
    text = re.sub(pattern, rf'\g<1>{fingerprint}\g<3>', text, count=1)
    changed.append((wire_id, old, fingerprint))

for forbidden in (
    '_EXECUTION_V1_LEGACY_FIELDS', '_EXECUTION_V1_CURRENT_FIELDS',
    '_RECOVERY_EXECUTION_V1_LEGACY_FIELDS', '_RECOVERY_EXECUTION_V1_CURRENT_FIELDS',
):
    if forbidden in text:
        raise RuntimeError(f'legacy V1 residue remains: {forbidden}')
path.write_text(text, encoding='utf-8')
print('runtime state canonical V1 binding codec applied')
for wire_id, old, new in changed:
    print(f'fingerprint {wire_id}: {old} -> {new}')

path = Path('linktools-ai/src/linktools/ai/workspace/_factory.py')
text = path.read_text(encoding='utf-8')
old = '    bindings: Sequence[AssetTypeBinding[object]] = (),\n'
new = '    bindings: "Sequence[AssetTypeBinding[object]]" = (),\n'
if text.count(old) != 1:
    raise RuntimeError(f'workspace annotation mismatch: {text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('workspace heterogeneous binding annotation retained')
