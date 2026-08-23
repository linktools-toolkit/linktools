#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC = ROOT / "linktools-ai/src/linktools/ai/runtime/state/_codec.py"
PERSISTENCE = ROOT / "linktools-ai/scripts/build/persistence.py"
MANIFEST = ROOT / "linktools-ai/scripts/build/matrix/runtime-persistence-v1.json"
FIXTURE = ROOT / "linktools-ai/scripts/build/matrix/runtime_agent_binding_snapshot_v1.json"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def fixture_payload() -> dict[str, object]:
    return {
        "version": 1,
        "agent_spec": {
            "id": "runtime-persistence-v1",
            "revision": 1,
            "model": "default",
            "system_prompt": "",
            "instructions": [],
            "allow_tools": ["*"],
            "usage_limits": None,
        },
        "agent_digest": "a" * 64,
        "output_type_module": "linktools.ai.runtime.fixture",
        "output_type_qualname": "FixtureOutput",
        "output_schema_id": "runtime.persistence.fixture",
        "output_schema_revision": 1,
        "output_schema_fingerprint": "b" * 64,
        "local_runtime_capability_descriptors": [],
        "binding_digest": "c" * 64,
    }


def patch_codec() -> None:
    text = CODEC.read_text(encoding="utf-8")
    if "runtime_agent_binding_snapshot_v1.json" in text:
        return
    text = replace_once(
        text,
        '            "agent_binding_snapshot": {"owner": "agent", "version": 1},\n',
        '''            "agent_binding_snapshot": {\n                "owner": "agent",\n                "version": 1,\n                "fixture": "runtime_agent_binding_snapshot_v1.json",\n            },\n''',
        "binding snapshot external manifest",
    )
    CODEC.write_text(text, encoding="utf-8")


def patch_manifest() -> None:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = value["external"]["agent_binding_snapshot"]
    contract["fixture"] = "runtime_agent_binding_snapshot_v1.json"
    MANIFEST.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def patch_persistence() -> None:
    text = PERSISTENCE.read_text(encoding="utf-8")
    if "def _agent_binding_snapshot_fixture_value(" in text:
        return
    text = replace_once(
        text,
        "from linktools.ai.core import JsonValue, canonical_json_bytes\n",
        '''from linktools.ai.agent import AgentBindingSnapshot\nfrom linktools.ai.core import JsonValue, canonical_json_bytes\n''',
        "binding fixture agent import",
    )
    text = replace_once(
        text,
        "from linktools.ai.runtime.state._codec import _runtime_persistence_manifest\n",
        '''from linktools.ai.runtime.state._codec import _runtime_persistence_manifest\nfrom linktools.ai.spec import AgentSpec\n''',
        "binding fixture spec import",
    )

    marker = "\ndef _model_message_fixture_values() -> tuple[ModelRequest, ...]:\n"
    helper = '''\ndef _agent_binding_snapshot_fixture_value() -> AgentBindingSnapshot:\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("runtime-persistence-v1"),\n        agent_digest="a" * 64,\n        output_type_module="linktools.ai.runtime.fixture",\n        output_type_qualname="FixtureOutput",\n        output_schema_id="runtime.persistence.fixture",\n        output_schema_revision=1,\n        output_schema_fingerprint="b" * 64,\n        local_runtime_capability_descriptors=(),\n        binding_digest="c" * 64,\n    )\n\n\ndef _load_external_fixture(path: Path) -> object:\n    try:\n        return json.loads(path.read_text(encoding="utf-8"))\n    except (OSError, json.JSONDecodeError) as error:\n        raise ValueError(f"invalid external persistence fixture: {path}: {error}") from error\n\n'''
    if marker not in text:
        raise RuntimeError("model fixture helper insertion point not found")
    text = text.replace(marker, helper + marker, 1)

    start = text.index("def validate_external_fixtures(")
    end = text.index("\n\ndef _git(", start)
    replacement = '''def validate_external_fixtures(\n    manifest: Mapping[str, JsonValue],\n    matrix_dir: str | Path,\n) -> tuple[str, ...]:\n    external = manifest.get("external")\n    if not isinstance(external, Mapping):\n        return ("external manifest is invalid",)\n    root = Path(matrix_dir)\n\n    binding_contract = external.get("agent_binding_snapshot")\n    if not isinstance(binding_contract, Mapping):\n        return ("agent_binding_snapshot external contract is invalid",)\n    binding_fixture = binding_contract.get("fixture")\n    if not isinstance(binding_fixture, str) or not binding_fixture:\n        return ("agent_binding_snapshot fixture is not declared",)\n    binding_path = root / binding_fixture\n    if not binding_path.is_file():\n        return (f"missing external persistence fixture: {binding_path}",)\n    try:\n        binding_value = _load_external_fixture(binding_path)\n    except ValueError as error:\n        return (str(error),)\n    expected_binding = _agent_binding_snapshot_fixture_value()\n    if binding_value != expected_binding.to_payload():\n        return ("agent_binding_snapshot writer drifted from the frozen V1 fixture",)\n    try:\n        decoded_binding = AgentBindingSnapshot.from_payload(binding_value)\n    except Exception as error:\n        return (f"agent_binding_snapshot V1 fixture is no longer readable: {error}",)\n    if decoded_binding != expected_binding:\n        return ("agent_binding_snapshot V1 fixture decodes to different semantics",)\n\n    model_contract = external.get("model_message")\n    if not isinstance(model_contract, Mapping):\n        return ("model_message external contract is invalid",)\n    model_fixture = model_contract.get("fixture")\n    if not isinstance(model_fixture, str) or not model_fixture:\n        return ("model_message fixture is not declared",)\n    model_path = root / model_fixture\n    if not model_path.is_file():\n        return (f"missing external persistence fixture: {model_path}",)\n    try:\n        model_value = _load_external_fixture(model_path)\n    except ValueError as error:\n        return (str(error),)\n    expected_model = json.loads(\n        encode_model_messages(_model_message_fixture_values()).decode("utf-8")\n    )\n    if model_value != expected_model:\n        return ("model_message writer drifted from the frozen V1 fixture",)\n    try:\n        decoded_model = decode_model_messages(canonical_json_bytes(model_value))\n    except Exception as error:\n        return (f"model_message V1 fixture is no longer readable: {error}",)\n    if decoded_model != _model_message_fixture_values():\n        return ("model_message V1 fixture decodes to different semantics",)\n    return ()\n'''
    text = text[:start] + replacement + text[end:]
    PERSISTENCE.write_text(text, encoding="utf-8")


def main() -> None:
    FIXTURE.write_text(
        json.dumps(fixture_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    patch_codec()
    patch_manifest()
    patch_persistence()
    print("agent binding snapshot V1 fixture and gate installed")


if __name__ == "__main__":
    main()
