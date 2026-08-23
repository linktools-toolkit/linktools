#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart

ROOT = Path(__file__).resolve().parents[1]
PERSISTENCE = ROOT / "linktools-ai/scripts/build/persistence.py"
MATRIX = ROOT / "linktools-ai/scripts/build/matrix"
FIXTURE = MATRIX / "runtime_model_messages_v1.json"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def fixture_messages() -> tuple[ModelRequest, ...]:
    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (
        ModelRequest(
            parts=[
                UserPromptPart(
                    content="runtime-persistence-v1",
                    timestamp=fixed,
                )
            ]
        ),
    )


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "linktools-ai/src"))
    from linktools.ai.runtime._message import encode_model_messages

    fixture = json.loads(encode_model_messages(fixture_messages()).decode("utf-8"))
    if not FIXTURE.exists():
        FIXTURE.write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    text = PERSISTENCE.read_text(encoding="utf-8")
    if "def validate_external_fixtures(" in text:
        print("external persistence fixture gate already present")
        return

    text = replace_once(
        text,
        "import json\nfrom collections.abc import Mapping\nfrom pathlib import Path\n",
        "import json\nfrom collections.abc import Mapping\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n",
        "fixture imports",
    )
    text = replace_once(
        text,
        "from linktools.ai.core import JsonValue\nfrom linktools.ai.runtime.state._codec import _runtime_persistence_manifest\n",
        '''from linktools.ai.core import JsonValue, canonical_json_bytes\nfrom linktools.ai.runtime._message import (\n    decode_model_messages,\n    encode_model_messages,\n)\nfrom linktools.ai.runtime.state._codec import _runtime_persistence_manifest\nfrom pydantic_ai.messages import ModelRequest, UserPromptPart\n''',
        "fixture codec imports",
    )

    marker = "\ndef _default_manifest() -> Path:\n"
    addition = '''\ndef _model_message_fixture_values() -> tuple[ModelRequest, ...]:\n    fixed = datetime(2026, 1, 1, tzinfo=timezone.utc)\n    return (\n        ModelRequest(\n            parts=[\n                UserPromptPart(\n                    content="runtime-persistence-v1",\n                    timestamp=fixed,\n                )\n            ]\n        ),\n    )\n\n\ndef validate_external_fixtures(\n    manifest: Mapping[str, JsonValue],\n    matrix_dir: str | Path,\n) -> tuple[str, ...]:\n    external = manifest.get("external")\n    if not isinstance(external, Mapping):\n        return ("external manifest is invalid",)\n    model_contract = external.get("model_message")\n    if not isinstance(model_contract, Mapping):\n        return ("model_message external contract is invalid",)\n    fixture_name = model_contract.get("fixture")\n    if not isinstance(fixture_name, str) or not fixture_name:\n        return ("model_message fixture is not declared",)\n    path = Path(matrix_dir) / fixture_name\n    if not path.is_file():\n        return (f"missing external persistence fixture: {path}",)\n    try:\n        value = json.loads(path.read_text(encoding="utf-8"))\n    except (OSError, json.JSONDecodeError) as error:\n        return (f"invalid external persistence fixture: {path}: {error}",)\n    expected = json.loads(\n        encode_model_messages(_model_message_fixture_values()).decode("utf-8")\n    )\n    if value != expected:\n        return ("model_message writer drifted from the frozen V1 fixture",)\n    try:\n        decoded = decode_model_messages(canonical_json_bytes(value))\n    except Exception as error:\n        return (f"model_message V1 fixture is no longer readable: {error}",)\n    if decoded != _model_message_fixture_values():\n        return ("model_message V1 fixture decodes to different semantics",)\n    return ()\n'''
    if marker not in text:
        raise RuntimeError("fixture validation insertion point not found")
    text = text.replace(marker, addition + marker, 1)

    text = replace_once(
        text,
        '''    errors = list(validate_runtime_persistence_manifest(candidate))\n    if args.baseline is not None:\n''',
        '''    errors = list(validate_runtime_persistence_manifest(candidate))\n    errors.extend(\n        validate_external_fixtures(candidate, args.manifest.parent)\n    )\n    if args.baseline is not None:\n''',
        "fixture gate main",
    )
    text = replace_once(
        text,
        '''    "load_manifest",\n    "validate_append_only",\n    "validate_runtime_persistence_manifest",\n''',
        '''    "load_manifest",\n    "validate_append_only",\n    "validate_external_fixtures",\n    "validate_runtime_persistence_manifest",\n''',
        "fixture gate exports",
    )
    PERSISTENCE.write_text(text, encoding="utf-8")
    print("external model_message V1 fixture and gate installed")


if __name__ == "__main__":
    main()
