#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_codec.py"
MANIFEST = ROOT / "linktools-ai/scripts/build/matrix/runtime-persistence-v1.json"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = CODEC.read_text(encoding="utf-8")
    if "_TASK_NODE_PERSISTENCE_DESCRIPTOR" in text:
        print("persistence exception gates already closed")
        return

    marker = '''_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(\n    {"task_node": _decode_v1_task_node}\n)\n\n\n'''
    contracts = '''_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(\n    {"task_node": _decode_v1_task_node}\n)\n\n_TASK_NODE_WIRE_FIELDS = frozenset(\n    {"node_id", "dependencies", "input", "budget_cost"}\n)\n_IDEMPOTENCY_TERMINAL_UPDATE_FIELDS = frozenset(\n    {\n        "scope",\n        "idempotency_key_digest",\n        "expected_status",\n        "next_status",\n        "request_digest",\n        "result_digest",\n        "error_code",\n    }\n)\n_OPERATION_TERMINAL_UPDATE_FIELDS = frozenset(\n    {\n        "operation_id",\n        "expected_status",\n        "next_status",\n        "result_ref",\n        "result_digest",\n        "error_code",\n    }\n)\n_TASK_NODE_PERSISTENCE_DESCRIPTOR: JsonValue = {\n    "fields": [\n        {"name": "node_id", "type": "str"},\n        {"name": "dependencies", "type": {"tuple_var": "str"}},\n        {\n            "name": "input",\n            "type": {"mapping": ["str", "json_value"]},\n        },\n        {"name": "budget_cost", "type": "int"},\n    ]\n}\n_RUNTIME_INLINE_DATACLASSES = frozenset(\n    {IdempotencyTerminalUpdate, OperationTerminalUpdate}\n)\n\n\n'''
    text = replace_once(text, marker, contracts, "custom persistence contracts")

    text = replace_once(
        text,
        '''    _require_exact_keys(\n        raw_fields,\n        frozenset(\n            {\n                "node_id",\n                "dependencies",\n                "input",\n                "budget_cost",\n            }\n        ),\n    )\n''',
        '''    _require_exact_keys(raw_fields, _TASK_NODE_WIRE_FIELDS)\n''',
        "task node decoder field contract",
    )
    text = replace_once(
        text,
        '''        _require_exact_keys(value, frozenset({\n            "scope", "idempotency_key_digest", "expected_status", "next_status",\n            "request_digest", "result_digest", "error_code",\n        }))\n''',
        '''        _require_exact_keys(value, _IDEMPOTENCY_TERMINAL_UPDATE_FIELDS)\n''',
        "idempotency terminal decoder field contract",
    )
    text = replace_once(
        text,
        '''        _require_exact_keys(value, frozenset({\n            "operation_id", "expected_status", "next_status", "result_ref",\n            "result_digest", "error_code",\n        }))\n''',
        '''        _require_exact_keys(value, _OPERATION_TERMINAL_UPDATE_FIELDS)\n''',
        "operation terminal decoder field contract",
    )

    old = '''def _dataclass_schema_descriptor(\n    target: type[object],\n    codec: _VersionCodec,\n) -> JsonValue:\n    if not is_dataclass(target):\n        raise TypeError(f"schema target is not a dataclass: {target!r}")\n    if target not in codec.wire_ids:\n        raise TypeError(f"schema target is not a V1 dataclass: {target!r}")\n    try:\n        hints = get_type_hints(target)\n    except (NameError, TypeError) as error:\n        raise TypeError(f"schema annotations are unresolved: {target!r}") from error\n    descriptors: list[JsonValue] = []\n    for field in fields(target):\n        annotation = hints.get(field.name)\n        if annotation is None:\n            raise TypeError(f"schema annotation is missing: {target!r}.{field.name}")\n        descriptors.append(\n            {\n                "name": field.name,\n                "init": field.init,\n                "type": _schema_type_descriptor(annotation, codec),\n            }\n        )\n    return {"fields": descriptors}\n'''
    new = '''def _inline_dataclass_schema_descriptor(\n    target: type[object],\n    codec: _VersionCodec,\n) -> JsonValue:\n    if not is_dataclass(target):\n        raise TypeError(f"schema target is not a dataclass: {target!r}")\n    try:\n        hints = get_type_hints(target)\n    except (NameError, TypeError) as error:\n        raise TypeError(f"schema annotations are unresolved: {target!r}") from error\n    descriptors: list[JsonValue] = []\n    for field in fields(target):\n        annotation = hints.get(field.name)\n        if annotation is None:\n            raise TypeError(f"schema annotation is missing: {target!r}.{field.name}")\n        descriptors.append(\n            {\n                "name": field.name,\n                "init": field.init,\n                "type": _schema_type_descriptor(annotation, codec),\n            }\n        )\n    return {"fields": descriptors}\n\n\ndef _dataclass_schema_descriptor(\n    target: type[object],\n    codec: _VersionCodec,\n) -> JsonValue:\n    if target is TaskNode:\n        return _TASK_NODE_PERSISTENCE_DESCRIPTOR\n    if target not in codec.wire_ids:\n        raise TypeError(f"schema target is not a V1 dataclass: {target!r}")\n    return _inline_dataclass_schema_descriptor(target, codec)\n'''
    text = replace_once(text, old, new, "schema descriptor ownership")

    needle = '''    external_descriptor = codec.external_schema_types.get(annotation)\n    if external_descriptor is not None:\n        return {"external": external_descriptor}\n'''
    replacement = '''    if annotation in _RUNTIME_INLINE_DATACLASSES:\n        return {\n            "inline_dataclass": _inline_dataclass_schema_descriptor(\n                annotation, codec\n            )\n        }\n    external_descriptor = codec.external_schema_types.get(annotation)\n    if external_descriptor is not None:\n        return {"external": external_descriptor}\n'''
    text = replace_once(text, needle, replacement, "runtime inline dataclass descriptor")

    old_validation = '''    task_node_fields = tuple(field.name for field in fields(TaskNode))\n    if task_node_fields != ("node_id", "dependencies", "budget_cost", "_input"):\n        raise RuntimeError("GA v1 task_node source contract changed")\n    for wire_id, target in _V1_WIRE_TYPES:\n        if wire_id in custom_dataclasses or not is_dataclass(target):\n            continue\n        actual = _dataclass_schema_fingerprint(target, _V1_CODEC)\n'''
    new_validation = '''    task_node_fields = tuple(field.name for field in fields(TaskNode))\n    if task_node_fields != ("node_id", "dependencies", "budget_cost", "_input"):\n        raise RuntimeError("GA v1 task_node source contract changed")\n    task_node_wire = _encode_v1_task_node(TaskNode("persistence-contract"), _V1_CODEC, False)\n    if frozenset(task_node_wire) != _TASK_NODE_WIRE_FIELDS:\n        raise RuntimeError("GA v1 task_node writer contract changed")\n    inline_contracts = (\n        (\n            IdempotencyTerminalUpdate,\n            (\n                "scope",\n                "idempotency_key_digest",\n                "expected_status",\n                "next_status",\n                "request_digest",\n                "result_digest",\n                "error_code",\n            ),\n            _IDEMPOTENCY_TERMINAL_UPDATE_FIELDS,\n            IdempotencyTerminalUpdate(\n                "scope",\n                "a" * 64,\n                IdempotencyStatus.STARTED,\n                IdempotencyStatus.COMPLETED,\n                "b" * 64,\n                None,\n                None,\n            ),\n        ),\n        (\n            OperationTerminalUpdate,\n            (\n                "operation_id",\n                "expected_status",\n                "next_status",\n                "result_ref",\n                "result_digest",\n                "error_code",\n            ),\n            _OPERATION_TERMINAL_UPDATE_FIELDS,\n            OperationTerminalUpdate(\n                "operation",\n                OperationStatus.RUNNING,\n                OperationStatus.SUCCEEDED,\n                None,\n                None,\n                None,\n            ),\n        ),\n    )\n    for target, expected_fields, wire_fields, sample in inline_contracts:\n        if tuple(field.name for field in fields(target)) != expected_fields:\n            raise RuntimeError(\n                f"GA v1 inline dataclass source contract changed: {target.__name__}"\n            )\n        encoded = _encode_external(sample, _V1_CODEC)\n        if not isinstance(encoded, Mapping) or frozenset(encoded) != wire_fields:\n            raise RuntimeError(\n                f"GA v1 inline dataclass writer contract changed: {target.__name__}"\n            )\n    for wire_id, target in _V1_WIRE_TYPES:\n        if not is_dataclass(target):\n            continue\n        actual = _dataclass_schema_fingerprint(target, _V1_CODEC)\n'''
    text = replace_once(text, old_validation, new_validation, "exception contract validation")

    # Temporarily disable module-level validation so the two mechanically
    # changed descriptor fingerprints can be calculated from the patched code.
    text = replace_once(
        text,
        "\n_validate_v1_codec_definition()\n",
        "\n# _validate_v1_codec_definition()  # bootstrap fingerprint calculation\n",
        "temporary validation disable",
    )
    CODEC.write_text(text, encoding="utf-8")

    sys.path.insert(0, str(ROOT / "linktools-ai/src"))
    codec = importlib.import_module("linktools.ai.runtime.state._codec")
    computed = {
        wire_id: codec._dataclass_schema_fingerprint(
            codec._V1_DOMAIN_TYPES[wire_id], codec._V1_CODEC
        )
        for wire_id in ("task_node", "execution_terminal_commit")
    }

    final = CODEC.read_text(encoding="utf-8")
    for wire_id, fingerprint in computed.items():
        pattern = rf'(\s+"{re.escape(wire_id)}":\s+")[0-9a-f]{{64}}(",)'
        final, count = re.subn(
            pattern,
            lambda match, fp=fingerprint: match.group(1) + fp + match.group(2),
            final,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"failed to repin {wire_id}")
    final = replace_once(
        final,
        "\n# _validate_v1_codec_definition()  # bootstrap fingerprint calculation\n",
        "\n_validate_v1_codec_definition()\n",
        "restore validation",
    )
    CODEC.write_text(final, encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for wire_id, fingerprint in computed.items():
        manifest["dataclasses"][wire_id]["revisions"]["1"] = fingerprint
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"repinned": computed}, sort_keys=True))


if __name__ == "__main__":
    main()
