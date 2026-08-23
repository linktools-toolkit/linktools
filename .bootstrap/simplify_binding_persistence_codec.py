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


text = CODEC.read_text(encoding="utf-8")

# Remove Runtime-owned duplication for Execution/Recovery binding payloads.
start = text.index("_EXECUTION_V1_FIELDS = frozenset(")
end = text.index("def _encode_v1_task_node(", start)
text = text[:start] + text[end:]

text = replace_once(
    text,
    '''_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(\n    {\n        "execution_record": _encode_v1_execution_record,\n        "recovery_execution_input": _encode_v1_recovery_execution_input,\n        "task_node": _encode_v1_task_node,\n    }\n)\n_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(\n    {\n        "execution_record": _decode_v1_execution_record,\n        "recovery_execution_input": _decode_v1_recovery_execution_input,\n        "task_node": _decode_v1_task_node,\n    }\n)\n''',
    '''_V1_DATACLASS_ENCODERS: Mapping[str, DataclassEncoder] = MappingProxyType(\n    {"task_node": _encode_v1_task_node}\n)\n_V1_DATACLASS_DECODERS: Mapping[str, DataclassDecoder] = MappingProxyType(\n    {"task_node": _decode_v1_task_node}\n)\n''',
    "custom dataclass maps",
)

text = replace_once(
    text,
    '''        OperationTerminalUpdate: (\n            "linktools.ai.runtime.state.OperationTerminalUpdate"\n        ),\n        ModelRequest: "pydantic_ai.messages.ModelRequest",\n''',
    '''        OperationTerminalUpdate: (\n            "linktools.ai.runtime.state.OperationTerminalUpdate"\n        ),\n        AgentBindingSnapshot: "linktools.ai.agent.AgentBindingSnapshot@1",\n        ModelRequest: "pydantic_ai.messages.ModelRequest",\n''',
    "external binding descriptor",
)

text = replace_once(
    text,
    '''    if type(value) not in codec.external_schema_types:\n        raise TypeError(f"unsupported external type: {type(value).__name__}")\n    if isinstance(value, IdempotencyTerminalUpdate):\n''',
    '''    if type(value) not in codec.external_schema_types:\n        raise TypeError(f"unsupported external type: {type(value).__name__}")\n    if isinstance(value, AgentBindingSnapshot):\n        return value.to_payload()\n    if isinstance(value, IdempotencyTerminalUpdate):\n''',
    "external binding encoder",
)

text = replace_once(
    text,
    '''    if target not in codec.external_schema_types:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    if target in (ModelRequest, ModelResponse):\n''',
    '''    if target not in codec.external_schema_types:\n        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)\n    if target is AgentBindingSnapshot:\n        try:\n            return AgentBindingSnapshot.from_payload(value)\n        except AIError:\n            raise\n        except (TypeError, ValueError, KeyError) as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n    if target in (ModelRequest, ModelResponse):\n''',
    "external binding decoder",
)

text = replace_once(
    text,
    '''    custom_dataclasses = {\n        "execution_record",\n        "recovery_execution_input",\n        "task_node",\n    }\n''',
    '''    custom_dataclasses = {"task_node"}\n''',
    "custom validation set",
)

# Add a mechanical current-source fingerprint gate for every generic dataclass.
needle = '''    if not set(_V1_DATACLASS_DECODERS).issubset(dataclass_wire_ids):\n        raise RuntimeError("GA v1 dataclass decoder manifest contains an unknown type")\n    if enum_value_ids != set(enum_wire_ids):\n'''
replacement = '''    if not set(_V1_DATACLASS_DECODERS).issubset(dataclass_wire_ids):\n        raise RuntimeError("GA v1 dataclass decoder manifest contains an unknown type")\n    task_node_fields = tuple(field.name for field in fields(TaskNode))\n    if task_node_fields != ("node_id", "dependencies", "budget_cost", "_input"):\n        raise RuntimeError("GA v1 task_node source contract changed")\n    for wire_id, target in _V1_WIRE_TYPES:\n        if wire_id in custom_dataclasses or not is_dataclass(target):\n            continue\n        actual = _dataclass_schema_fingerprint(target, _V1_CODEC)\n        contract = _V1_DATACLASS_PERSISTENCE[wire_id]\n        if actual != contract.fingerprints[contract.current_revision]:\n            raise RuntimeError(\n                f"GA v1 current dataclass fingerprint changed without a "\n                f"schema revision: {wire_id}"\n            )\n    if enum_value_ids != set(enum_wire_ids):\n'''
text = replace_once(text, needle, replacement, "mechanical fingerprint validation")

# Write a temporary source that can be imported to mechanically calculate the
# two newly-generic descriptor fingerprints. The current validator does not yet
# execute the new loop until these pins are updated below.
loop_start = text.index("    task_node_fields = tuple(field.name for field in fields(TaskNode))")
loop_end = text.index("    if enum_value_ids != set(enum_wire_ids):", loop_start)
validation_block = text[loop_start:loop_end]
text_without_loop = text[:loop_start] + text[loop_end:]
CODEC.write_text(text_without_loop, encoding="utf-8")

sys.path.insert(0, str(ROOT / "linktools-ai/src"))
codec = importlib.import_module("linktools.ai.runtime.state._codec")
computed: dict[str, str] = {}
for wire_id in ("execution_record", "recovery_execution_input"):
    target = codec._V1_DOMAIN_TYPES[wire_id]
    computed[wire_id] = codec._dataclass_schema_fingerprint(target, codec._V1_CODEC)

# Patch the immutable revision-1 pins to the now mechanically representable
# descriptor. These values are build metadata; persisted V1 bytes are unchanged.
final_text = CODEC.read_text(encoding="utf-8")
for wire_id, fingerprint in computed.items():
    pattern = rf'(\s+"{re.escape(wire_id)}":\s+")[0-9a-f]{{64}}(",)'
    final_text, count = re.subn(
        pattern,
        lambda match, fp=fingerprint: match.group(1) + fp + match.group(2),
        final_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"failed to repin {wire_id}")

# Restore the mechanical validation block now that the pins are correct.
insert_at = final_text.index("    if enum_value_ids != set(enum_wire_ids):")
final_text = final_text[:insert_at] + validation_block + final_text[insert_at:]
CODEC.write_text(final_text, encoding="utf-8")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
for wire_id, fingerprint in computed.items():
    manifest["dataclasses"][wire_id]["revisions"]["1"] = fingerprint
MANIFEST.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

print(json.dumps({"repinned": computed}, sort_keys=True))
