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
MARKER = "GA v1 current dataclass fingerprint changed without a schema revision"


def main() -> None:
    sys.path.insert(0, str(ROOT / "linktools-ai/src"))
    codec = importlib.import_module("linktools.ai.runtime.state._codec")
    computed = {
        wire_id: codec._dataclass_schema_fingerprint(
            codec._V1_DOMAIN_TYPES[wire_id], codec._V1_CODEC
        )
        for wire_id in ("execution_record", "recovery_execution_input")
    }

    text = CODEC.read_text(encoding="utf-8")
    for wire_id, fingerprint in computed.items():
        pattern = rf'(\s+"{re.escape(wire_id)}":\s+")[0-9a-f]{{64}}(",)'
        text, count = re.subn(
            pattern,
            lambda match, fp=fingerprint: match.group(1) + fp + match.group(2),
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"failed to pin current fingerprint: {wire_id}")

    if MARKER not in text:
        needle = '''    if not set(_V1_DATACLASS_DECODERS).issubset(dataclass_wire_ids):\n        raise RuntimeError("GA v1 dataclass decoder manifest contains an unknown type")\n    if enum_value_ids != set(enum_wire_ids):\n'''
        replacement = '''    if not set(_V1_DATACLASS_DECODERS).issubset(dataclass_wire_ids):\n        raise RuntimeError("GA v1 dataclass decoder manifest contains an unknown type")\n    task_node_fields = tuple(field.name for field in fields(TaskNode))\n    if task_node_fields != ("node_id", "dependencies", "budget_cost", "_input"):\n        raise RuntimeError("GA v1 task_node source contract changed")\n    for wire_id, target in _V1_WIRE_TYPES:\n        if wire_id in custom_dataclasses or not is_dataclass(target):\n            continue\n        actual = _dataclass_schema_fingerprint(target, _V1_CODEC)\n        contract = _V1_DATACLASS_PERSISTENCE[wire_id]\n        if actual != contract.fingerprints[contract.current_revision]:\n            raise RuntimeError(\n                f"GA v1 current dataclass fingerprint changed without a "\n                f"schema revision: {wire_id}"\n            )\n    if enum_value_ids != set(enum_wire_ids):\n'''
        count = text.count(needle)
        if count != 1:
            raise RuntimeError(
                f"mechanical fingerprint gate: expected one insertion point, found {count}"
            )
        text = text.replace(needle, replacement, 1)

    CODEC.write_text(text, encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for wire_id, fingerprint in computed.items():
        manifest["dataclasses"][wire_id]["revisions"]["1"] = fingerprint
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"generic_binding_fingerprints": computed}, sort_keys=True))


if __name__ == "__main__":
    main()
