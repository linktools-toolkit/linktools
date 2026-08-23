#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    ROOT / "tests/ai/test_runtime_state_identity_migration.py",
    ROOT / "tests/ai/test_runtime_state_identity_migration_sqlite.py",
)

for path in FILES:
    text = path.read_text(encoding="utf-8")
    old = '''    payload = dict(value["payload"])\n    fields = dict(payload["fields"])\n'''
    replacement = '''    payload = dict(value["payload"])\n    payload.pop("schema", None)\n    fields = dict(payload["fields"])\n'''
    # Every _domain_data-based historical root must drop the current schema tag.
    text = text.replace(old, replacement)

    old = '''    raw_input = dict(fields["input"])\n    input_fields = dict(raw_input["fields"])\n'''
    if old in text:
        text = text.replace(
            old,
            '''    raw_input = dict(fields["input"])\n    raw_input.pop("schema", None)\n    input_fields = dict(raw_input["fields"])\n''',
        )
    path.write_text(text, encoding="utf-8")

print("legacy identity fixtures normalized to authentic unversioned V1")
