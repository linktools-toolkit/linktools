#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_codec.py"
REPOSITORIES = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_repositories.py"
HISTORY = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_history.py"
MIGRATION = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_migration.py"
TEST = ROOT / "tests/ai/test_runtime_persistence_compatibility.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def patch_codec() -> None:
    text = CODEC.read_text(encoding="utf-8")

    text = replace_once(
        text,
        '''DataclassEncoder = Callable[\n    [object, "_VersionCodec"],\n    Mapping[str, JsonValue],\n]\nDataclassDecoder = Callable[\n    [Mapping[str, object], "_VersionCodec"],\n    object,\n]\n''',
        '''DataclassEncoder = Callable[\n    [object, "_VersionCodec", bool],\n    Mapping[str, JsonValue],\n]\nDataclassDecoder = Callable[\n    [Mapping[str, object], "_VersionCodec", bool],\n    object,\n]\n''',
        "typed custom codec signatures",
    )

    text = replace_once(
        text,
        '''def _encode_v1_exact_binding_record(\n    value: object,\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n) -> Mapping[str, JsonValue]:\n''',
        '''def _encode_v1_exact_binding_record(\n    value: object,\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n    persisted: bool,\n) -> Mapping[str, JsonValue]:\n''',
        "exact binding encoder signature",
    )
    text = replace_once(
        text,
        '''            binding.to_payload()\n            if name == "binding"\n            else _encode_domain(attrgetter(name)(value), codec)\n''',
        '''            binding.to_payload()\n            if name == "binding"\n            else _encode_domain(\n                attrgetter(name)(value), codec, persisted=persisted\n            )\n''',
        "exact binding nested encode",
    )
    text = replace_once(
        text,
        '''def _decode_v1_exact_binding_record_fields(\n    raw_fields: Mapping[str, object],\n    target: type[object],\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n) -> dict[str, object]:\n''',
        '''def _decode_v1_exact_binding_record_fields(\n    raw_fields: Mapping[str, object],\n    target: type[object],\n    expected_fields: frozenset[str],\n    codec: "_VersionCodec",\n    persisted: bool,\n) -> dict[str, object]:\n''',
        "exact binding decoder signature",
    )
    text = replace_once(
        text,
        '''            AgentBindingSnapshot.from_payload(raw_fields[name])\n            if name == "binding"\n            else _decode_domain(raw_fields[name], hints[name], codec)\n''',
        '''            AgentBindingSnapshot.from_payload(raw_fields[name])\n            if name == "binding"\n            else _decode_domain(\n                raw_fields[name], hints[name], codec, persisted=persisted\n            )\n''',
        "exact binding nested decode",
    )

    for name in ("execution_record", "recovery_execution_input"):
        text = text.replace(
            f'''def _encode_v1_{name}(\n    value: object,\n    codec: "_VersionCodec",\n)''',
            f'''def _encode_v1_{name}(\n    value: object,\n    codec: "_VersionCodec",\n    persisted: bool,\n)''',
            1,
        )
        text = text.replace(
            f'''def _decode_v1_{name}(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n)''',
            f'''def _decode_v1_{name}(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n    persisted: bool,\n)''',
            1,
        )
    text = replace_once(
        text,
        '''    return _encode_v1_exact_binding_record(value, _EXECUTION_V1_FIELDS, codec)\n''',
        '''    return _encode_v1_exact_binding_record(\n        value, _EXECUTION_V1_FIELDS, codec, persisted\n    )\n''',
        "execution encoder propagation",
    )
    text = replace_once(
        text,
        '''        raw_fields, ExecutionRecord, _EXECUTION_V1_FIELDS, codec\n''',
        '''        raw_fields, ExecutionRecord, _EXECUTION_V1_FIELDS, codec, persisted\n''',
        "execution decoder propagation",
    )
    text = replace_once(
        text,
        '''        value, _RECOVERY_EXECUTION_V1_FIELDS, codec\n''',
        '''        value, _RECOVERY_EXECUTION_V1_FIELDS, codec, persisted\n''',
        "recovery encoder propagation",
    )
    text = replace_once(
        text,
        '''        raw_fields, RecoveryExecutionInput, _RECOVERY_EXECUTION_V1_FIELDS, codec\n''',
        '''        raw_fields, RecoveryExecutionInput, _RECOVERY_EXECUTION_V1_FIELDS, codec, persisted\n''',
        "recovery decoder propagation",
    )

    text = replace_once(
        text,
        '''def _encode_v1_task_node(\n    value: object,\n    codec: "_VersionCodec",\n) -> Mapping[str, JsonValue]:\n''',
        '''def _encode_v1_task_node(\n    value: object,\n    codec: "_VersionCodec",\n    persisted: bool,\n) -> Mapping[str, JsonValue]:\n''',
        "task encoder signature",
    )
    text = replace_once(
        text,
        '''        "node_id": _encode_domain(value.node_id, codec),\n        "dependencies": _encode_domain(value.dependencies, codec),\n        "input": _encode_domain(value.input, codec),\n        "budget_cost": _encode_domain(value.budget_cost, codec),\n''',
        '''        "node_id": _encode_domain(value.node_id, codec, persisted=persisted),\n        "dependencies": _encode_domain(\n            value.dependencies, codec, persisted=persisted\n        ),\n        "input": _encode_domain(value.input, codec, persisted=persisted),\n        "budget_cost": _encode_domain(\n            value.budget_cost, codec, persisted=persisted\n        ),\n''',
        "task nested encode",
    )
    text = replace_once(
        text,
        '''def _decode_v1_task_node(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n) -> TaskNode:\n''',
        '''def _decode_v1_task_node(\n    raw_fields: Mapping[str, object],\n    codec: "_VersionCodec",\n    persisted: bool,\n) -> TaskNode:\n''',
        "task decoder signature",
    )
    text = text.replace(
        '''_decode_domain(raw_fields["node_id"], str, codec)''',
        '''_decode_domain(raw_fields["node_id"], str, codec, persisted=persisted)''',
        1,
    )
    text = text.replace(
        '''                tuple[str, ...],\n                codec,\n''',
        '''                tuple[str, ...],\n                codec,\n                persisted=persisted,\n''',
        1,
    )
    text = text.replace(
        '''            Any,\n            codec,\n''',
        '''            Any,\n            codec,\n            persisted=persisted,\n''',
        1,
    )
    text = text.replace(
        '''_decode_domain(raw_fields["budget_cost"], int, codec)''',
        '''_decode_domain(\n                raw_fields["budget_cost"], int, codec, persisted=persisted\n            )''',
        1,
    )

    start = text.index("def _encode_persisted_value(")
    end = text.index("def _runtime_persistence_manifest()", start)
    replacement = '''def _apply_persisted_upgrades(\n    raw_fields: Mapping[str, object],\n    revision: int,\n    contract: _DataclassPersistenceContract,\n    codec: _VersionCodec,\n) -> Mapping[str, object]:\n    fields_value: Mapping[str, object] = dict(raw_fields)\n    current = revision\n    while current < contract.current_revision:\n        upgrade = contract.upgrades.get(current)\n        if upgrade is None:\n            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n        fields_value = dict(upgrade(fields_value, codec))\n        current += 1\n    return fields_value\n\n\n'''
    text = text[:start] + replacement + text[end:]

    old = '''def encode_envelope(\n    value: Mapping[str, JsonValue],\n    *,\n    version: int = CURRENT_DATA_VERSION,\n) -> dict[str, JsonValue]:\n    if version != CURRENT_DATA_VERSION:\n        raise ValueError("only the frozen current data version may be written")\n    if not isinstance(value, Mapping):\n        raise TypeError("canonical data value must be a mapping")\n    codec = _VERSION_CODECS.get(version)\n    if codec is None:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    persisted = _encode_persisted_value(dict(value), codec)\n    if not isinstance(persisted, Mapping):\n        raise TypeError("canonical data value must remain a mapping")\n    return {"v": version, "value": dict(persisted)}\n'''
    new = '''def encode_envelope(\n    value: Mapping[str, JsonValue],\n    *,\n    version: int = CURRENT_DATA_VERSION,\n) -> dict[str, JsonValue]:\n    if version != CURRENT_DATA_VERSION:\n        raise ValueError("only the frozen current data version may be written")\n    if not isinstance(value, Mapping):\n        raise TypeError("canonical data value must be a mapping")\n    return {"v": version, "value": dict(value)}\n'''
    text = replace_once(text, old, new, "restore opaque envelope")

    text = replace_once(
        text,
        '''def _encode_domain(value: object, codec: _VersionCodec) -> JsonValue:\n''',
        '''def _encode_domain(\n    value: object,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> JsonValue:\n''',
        "typed encode mode",
    )
    text = text.replace(
        '''            encoded_fields = encoder(value, codec)''',
        '''            encoded_fields = encoder(value, codec, persisted)''',
        1,
    )
    text = text.replace(
        '''                field.name: _encode_domain(attrgetter(field.name)(value), codec)\n''',
        '''                field.name: _encode_domain(\n                    attrgetter(field.name)(value), codec, persisted=persisted\n                )\n''',
        1,
    )
    old = '''        wire: dict[str, JsonValue] = {\n            "$dataclass": wire_id,\n            "fields": dict(encoded_fields),\n        }\n        try:\n            _decode_dataclass(wire, type(value), codec)\n'''
    new = '''        wire: dict[str, JsonValue] = {\n            "$dataclass": wire_id,\n            "fields": dict(encoded_fields),\n        }\n        if persisted:\n            contract = _dataclass_persistence_contract(wire_id, codec)\n            wire = {\n                "$dataclass": wire_id,\n                "schema": contract.current_revision,\n                "fields": dict(encoded_fields),\n            }\n        try:\n            _decode_dataclass(wire, type(value), codec, persisted=persisted)\n'''
    text = replace_once(text, old, new, "dataclass persisted wrapper")

    text = text.replace(
        '''            encoded_key = _encode_domain(key, codec)\n            encoded_item = _encode_domain(item, codec)\n''',
        '''            encoded_key = _encode_domain(key, codec, persisted=persisted)\n            encoded_item = _encode_domain(item, codec, persisted=persisted)\n''',
        1,
    )
    text = text.replace(
        '''return {"$tuple": [_encode_domain(item, codec) for item in value]}''',
        '''return {\n            "$tuple": [\n                _encode_domain(item, codec, persisted=persisted) for item in value\n            ]\n        }''',
        1,
    )
    text = text.replace(
        '''return [_encode_domain(item, codec) for item in value]''',
        '''return [_encode_domain(item, codec, persisted=persisted) for item in value]''',
        1,
    )
    text = text.replace(
        '''            for encoded_item in (_encode_domain(item, codec),)\n''',
        '''            for encoded_item in (\n                _encode_domain(item, codec, persisted=persisted),\n            )\n''',
        1,
    )

    old = '''def encode_domain(value: DomainT) -> JsonValue:\n    """Encode one domain value into the shared canonical JSON representation."""\n    return _encode_domain(value, _CURRENT_CODEC)\n\n\ndef decode_domain(value: JsonValue, target: type[DomainT]) -> DomainT:\n'''
    new = '''def encode_domain(value: DomainT) -> JsonValue:\n    """Encode one domain value into the shared canonical JSON representation."""\n    return _encode_domain(value, _CURRENT_CODEC)\n\n\ndef _encode_persisted_domain(value: DomainT) -> JsonValue:\n    """Encode one domain value for a Runtime persistence envelope."""\n    return _encode_domain(value, _CURRENT_CODEC, persisted=True)\n\n\ndef decode_domain(value: JsonValue, target: type[DomainT]) -> DomainT:\n'''
    text = replace_once(text, old, new, "persisted encode entry")

    text = text.replace(
        text,
        '''    normalized = _normalize_persisted_value(payload, codec)\n    return _decode_domain(normalized, target, codec)\n''',
        '''    return _decode_domain(payload, target, codec, persisted=True)\n''',
        1,
    )
    text = replace_once(
        text,
        '''    normalized = _normalize_persisted_value(payload, codec)\n    yield from _iter_runtime_object_refs(normalized, default_domain, codec)\n''',
        '''    yield from _iter_runtime_object_refs(payload, default_domain, codec)\n''',
        "opaque object traversal envelope",
    )
    text = text.replace(
        '''_decode_domain(value, RuntimePayloadRef, codec)''',
        '''_decode_domain(value, RuntimePayloadRef, codec, persisted=True)''',
        1,
    )
    text = text.replace(
        '''_decode_domain(value, StoredPayload, codec)''',
        '''_decode_domain(value, StoredPayload, codec, persisted=True)''',
        1,
    )
    text = text.replace(
        '''_decode_domain(value, ObjectRef, codec)''',
        '''_decode_domain(value, ObjectRef, codec, persisted=True)''',
        1,
    )

    text = replace_once(
        text,
        '''def _decode_domain(\n    value: object,\n    target: object,\n    codec: _VersionCodec,\n) -> object:\n    if target is Any or target is object:\n        return _decode_any(value, codec)\n''',
        '''def _decode_domain(\n    value: object,\n    target: object,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> object:\n    if target is Any or target is object:\n        return _decode_any(value, codec, persisted=persisted)\n''',
        "typed decode mode",
    )

    # Propagate persisted mode through recursive typed decoding.
    text = text.replace(
        '''return _decode_domain(value, candidate, codec)''',
        '''return _decode_domain(value, candidate, codec, persisted=persisted)''',
    )
    text = text.replace(
        '''return [_decode_domain(item, item_type, codec) for item in value]''',
        '''return [\n            _decode_domain(item, item_type, codec, persisted=persisted)\n            for item in value\n        ]''',
    )
    text = text.replace(
        '''_decode_domain(item, arguments[0], codec)''',
        '''_decode_domain(item, arguments[0], codec, persisted=persisted)''',
    )
    text = text.replace(
        '''_decode_domain(item, item_type, codec)\n            for item, item_type in zip(items, arguments, strict=True)''',
        '''_decode_domain(item, item_type, codec, persisted=persisted)\n            for item, item_type in zip(items, arguments, strict=True)''',
    )
    text = text.replace(
        '''return _decode_frozenset_items(value, item_type, codec)''',
        '''return _decode_frozenset_items(\n            value, item_type, codec, persisted=persisted\n        )''',
    )
    text = text.replace(
        '''return _decode_mapping_items(value, key_type, item_type, codec)''',
        '''return _decode_mapping_items(\n            value, key_type, item_type, codec, persisted=persisted\n        )''',
    )
    text = text.replace(
        '''return _decode_dataclass(value, target, codec)''',
        '''return _decode_dataclass(value, target, codec, persisted=persisted)''',
        1,
    )
    text = text.replace(
        '''return _decode_any(value, codec)''',
        '''return _decode_any(value, codec, persisted=persisted)''',
        1,
    )

    # Replace the dataclass decoder wholesale.
    start = text.index("def _decode_dataclass(")
    end = text.index("def _decode_any(", start)
    decoder = '''def _decode_dataclass(\n    value: object,\n    target: type,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> object:\n    if not isinstance(value, Mapping):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    keys = frozenset(value)\n    if persisted:\n        if keys == frozenset({"$dataclass", "fields"}):\n            explicit_revision: object | None = None\n        elif keys == frozenset({"$dataclass", "schema", "fields"}):\n            explicit_revision = value.get("schema")\n        else:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    else:\n        _require_exact_keys(value, frozenset({"$dataclass", "fields"}))\n        explicit_revision = None\n\n    wire_id = value.get("$dataclass")\n    if not isinstance(wire_id, str):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    expected_target = codec.domain_types.get(wire_id)\n    if expected_target is None:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    if expected_target is not target:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    raw_fields = value.get("fields")\n    if not isinstance(raw_fields, Mapping):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n\n    if persisted:\n        contract = _dataclass_persistence_contract(wire_id, codec)\n        if explicit_revision is None:\n            revision = contract.legacy_revision\n            if revision is None:\n                raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n        else:\n            if (\n                isinstance(explicit_revision, bool)\n                or not isinstance(explicit_revision, int)\n                or explicit_revision < 1\n            ):\n                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n            revision = explicit_revision\n        if revision not in contract.fingerprints:\n            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n        raw_fields = _apply_persisted_upgrades(\n            raw_fields, revision, contract, codec\n        )\n\n    decoder = codec.dataclass_decoders.get(wire_id)\n    if decoder is not None:\n        return decoder(raw_fields, codec, persisted)\n    expected_fingerprint = codec.schema_fingerprints.get(wire_id)\n    if expected_fingerprint is None:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    try:\n        actual_fingerprint = _dataclass_schema_fingerprint(target, codec)\n    except TypeError as error:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error\n    if actual_fingerprint != expected_fingerprint:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    hints = get_type_hints(target)\n    declared_fields = tuple(fields(target))\n    if set(raw_fields.keys()) != {field.name for field in declared_fields}:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    decoded_fields: dict[str, object] = {}\n    for field in declared_fields:\n        decoded_fields[field.name] = _decode_domain(\n            raw_fields[field.name],\n            hints.get(field.name, Any),\n            codec,\n            persisted=persisted,\n        )\n    kwargs: dict[str, object] = {}\n    post_init_fields: dict[str, object] = {}\n    for field in declared_fields:\n        if field.init:\n            kwargs[field.name] = decoded_fields[field.name]\n        else:\n            post_init_fields[field.name] = decoded_fields[field.name]\n    try:\n        result = target(**kwargs)\n    except (TypeError, ValueError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n    for field_name, expected in post_init_fields.items():\n        try:\n            actual = attrgetter(field_name)(result)\n        except AttributeError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n        if actual != expected:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    return result\n\n\n'''
    text = text[:start] + decoder + text[end:]

    text = replace_once(
        text,
        '''def _decode_any(value: object, codec: _VersionCodec) -> object:\n''',
        '''def _decode_any(\n    value: object,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> object:\n''',
        "any decode signature",
    )
    text = text.replace(
        '''return _decode_dataclass(value, target, codec)''',
        '''return _decode_dataclass(value, target, codec, persisted=persisted)''',
        1,
    )
    text = text.replace(
        '''return _decode_mapping_items(value, Any, Any, codec)''',
        '''return _decode_mapping_items(\n                value, Any, Any, codec, persisted=persisted\n            )''',
        1,
    )
    text = text.replace(
        '''_decode_any(item, codec)\n                for item in _unwrap_tagged_list(value, "$tuple")''',
        '''_decode_any(item, codec, persisted=persisted)\n                for item in _unwrap_tagged_list(value, "$tuple")''',
        1,
    )
    text = text.replace(
        '''return _decode_frozenset_items(value, Any, codec)''',
        '''return _decode_frozenset_items(\n            value, Any, codec, persisted=persisted\n        )''',
        1,
    )
    text = text.replace(
        '''return [_decode_any(item, codec) for item in value]''',
        '''return [\n            _decode_any(item, codec, persisted=persisted) for item in value\n        ]''',
        1,
    )

    text = replace_once(
        text,
        '''def _decode_frozenset_items(\n    value: object,\n    item_type: object,\n    codec: _VersionCodec,\n) -> frozenset[object]:\n''',
        '''def _decode_frozenset_items(\n    value: object,\n    item_type: object,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> frozenset[object]:\n''',
        "frozenset decode signature",
    )
    text = text.replace(
        '''decoded_item = _decode_domain(item, item_type, codec)''',
        '''decoded_item = _decode_domain(\n            item, item_type, codec, persisted=persisted\n        )''',
        1,
    )
    text = replace_once(
        text,
        '''def _decode_mapping_items(\n    value: object,\n    key_type: object,\n    item_type: object,\n    codec: _VersionCodec,\n) -> dict[object, object]:\n''',
        '''def _decode_mapping_items(\n    value: object,\n    key_type: object,\n    item_type: object,\n    codec: _VersionCodec,\n    *,\n    persisted: bool = False,\n) -> dict[object, object]:\n''',
        "mapping decode signature",
    )
    text = text.replace(
        '''decoded_key = _decode_domain(pair[0], key_type, codec)''',
        '''decoded_key = _decode_domain(\n            pair[0], key_type, codec, persisted=persisted\n        )''',
        1,
    )
    text = text.replace(
        '''result[decoded_key] = _decode_domain(pair[1], item_type, codec)''',
        '''result[decoded_key] = _decode_domain(\n            pair[1], item_type, codec, persisted=persisted\n        )''',
        1,
    )

    text = replace_once(
        text,
        '''            "payload": encode_domain(value),\n''',
        '''            "payload": _encode_persisted_domain(value),\n''',
        "step persisted encode",
    )
    text = replace_once(
        text,
        '''    normalized = _normalize_persisted_value(payload, codec)\n    return _decode_domain(normalized, target, codec)\n''',
        '''    return _decode_domain(payload, target, codec, persisted=True)\n''',
        "step typed decode",
    )

    CODEC.write_text(text, encoding="utf-8")


def patch_repositories() -> None:
    text = REPOSITORIES.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    _decode_enveloped_domain,\n    encode_domain,\n    encode_envelope,\n''',
        '''    _decode_enveloped_domain,\n    _encode_persisted_domain,\n    encode_envelope,\n''',
        "repository codec import",
    )
    text = replace_once(
        text,
        '''    payload = encode_domain(value)\n''',
        '''    payload = _encode_persisted_domain(value)\n''',
        "repository persisted encode",
    )
    REPOSITORIES.write_text(text, encoding="utf-8")


def patch_history() -> None:
    text = HISTORY.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''from ._codec import (\n    _decode_enveloped_domain,\n    encode_domain,\n    encode_envelope,\n)\n''',
        '''from ._codec import (\n    _decode_enveloped_domain,\n    _encode_persisted_domain,\n    encode_domain,\n    encode_envelope,\n)\n''',
        "history codec import",
    )
    text = text.replace('''"payload": encode_domain(''', '''"payload": _encode_persisted_domain(''')
    HISTORY.write_text(text, encoding="utf-8")


def patch_migration() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    _decode_domain,\n    _decode_enveloped_domain,\n    encode_domain,\n''',
        '''    _dataclass_persistence_contract,\n    _decode_domain,\n    _decode_enveloped_domain,\n    _encode_persisted_domain,\n    encode_domain,\n''',
        "migration codec imports",
    )
    old = '''    return encode_envelope(\n        {\n            "type": wire_id,\n            "payload": {"$dataclass": wire_id, "fields": dict(fields)},\n        }\n    )\n'''
    new = '''    contract = _dataclass_persistence_contract(wire_id, _CURRENT_CODEC)\n    return encode_envelope(\n        {\n            "type": wire_id,\n            "payload": {\n                "$dataclass": wire_id,\n                "schema": contract.current_revision,\n                "fields": dict(fields),\n            },\n        }\n    )\n'''
    text = replace_once(text, old, new, "migration current rebuild")
    text = replace_once(
        text,
        '''        {"type": wire_type_id(value), "payload": encode_domain(value)}\n''',
        '''        {\n            "type": wire_type_id(value),\n            "payload": _encode_persisted_domain(value),\n        }\n''',
        "projection current persisted encode",
    )
    MIGRATION.write_text(text, encoding="utf-8")


def patch_test() -> None:
    text = TEST.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    _decode_step_envelope,\n    _encode_step_envelope,\n''',
        '''    _decode_step_envelope,\n    _encode_persisted_domain,\n    _encode_step_envelope,\n''',
        "test persisted import",
    )
    text = text.replace(
        '''{"type": wire_type_id(cursor), "payload": encode_domain(cursor)}''',
        '''{\n            "type": wire_type_id(cursor),\n            "payload": _encode_persisted_domain(cursor),\n        }''',
        1,
    )
    text = text.replace(
        '''            "payload": encode_domain(ConversationCursor("run", None, 0)),''',
        '''            "payload": _encode_persisted_domain(\n                ConversationCursor("run", None, 0)\n            ),''',
        1,
    )
    TEST.write_text(text, encoding="utf-8")


def main() -> None:
    patch_codec()
    patch_repositories()
    patch_history()
    patch_migration()
    patch_test()
    print("typed Runtime persistence codec boundary applied")


if __name__ == "__main__":
    main()
