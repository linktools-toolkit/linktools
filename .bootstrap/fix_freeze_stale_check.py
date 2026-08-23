#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / ".bootstrap/freeze_identity_migration_schema1.py"


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    old = '''    forbidden = (\n        "_EXECUTION_V1_FIELDS",\n        "_RECOVERY_EXECUTION_V1_FIELDS",\n        "_CURRENT_SESSION_FIELDS",\n        "_LEGACY_SESSION_FIELDS",\n        "_CURRENT_PROJECTION_FIELDS",\n        "_LEGACY_PROJECTION_FIELDS",\n        "_CURRENT_RECOVERY_INPUT_FIELDS",\n        "_LEGACY_RECOVERY_INPUT_FIELDS",\n        "_rebuild_data(",\n        "dataclass_fields",\n        "get_type_hints",\n        "_dataclass_persistence_contract",\n    )\n    stale = [name for name in forbidden if name in text]\n'''
    new = '''    forbidden = (\n        "    _EXECUTION_V1_FIELDS,\\n",\n        "    _RECOVERY_EXECUTION_V1_FIELDS,\\n",\n        "_CURRENT_SESSION_FIELDS",\n        "_LEGACY_SESSION_FIELDS",\n        "_CURRENT_PROJECTION_FIELDS",\n        "_LEGACY_PROJECTION_FIELDS",\n        "_CURRENT_RECOVERY_INPUT_FIELDS",\n        "_LEGACY_RECOVERY_INPUT_FIELDS",\n        "_rebuild_data(",\n        "dataclass_fields(",\n        "get_type_hints(",\n        "_dataclass_persistence_contract(",\n    )\n    stale = [name for name in forbidden if name in text]\n'''
    count = text.count(old)
    if count == 0 and '"dataclass_fields("' in text:
        print("migration bootstrap stale-symbol check already precise")
        return
    if count != 1:
        raise RuntimeError(f"stale-symbol check: expected one match, found {count}")
    PATH.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("migration bootstrap stale-symbol check made exact")


if __name__ == "__main__":
    main()
