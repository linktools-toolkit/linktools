#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time validation for the Runtime persistence compatibility contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from linktools.ai.core import JsonValue
from linktools.ai.runtime.state._codec import _runtime_persistence_manifest


def load_manifest(path: str | Path) -> dict[str, JsonValue]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime persistence manifest must be an object")
    return cast("dict[str, JsonValue]", value)


def validate_runtime_persistence_manifest(
    manifest: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    expected = _runtime_persistence_manifest()
    return () if dict(manifest) == expected else (
        "runtime persistence manifest does not match the current codec contract",
    )


def validate_append_only(
    baseline: Mapping[str, JsonValue],
    candidate: Mapping[str, JsonValue],
) -> tuple[str, ...]:
    errors: list[str] = []
    if baseline.get("wire_version") != candidate.get("wire_version"):
        errors.append("wire_version changed")

    baseline_dataclasses = baseline.get("dataclasses")
    candidate_dataclasses = candidate.get("dataclasses")
    if not isinstance(baseline_dataclasses, Mapping) or not isinstance(
        candidate_dataclasses, Mapping
    ):
        return tuple((*errors, "dataclasses manifest is invalid"))
    for wire_id, raw_baseline in baseline_dataclasses.items():
        raw_candidate = candidate_dataclasses.get(wire_id)
        if not isinstance(raw_baseline, Mapping) or not isinstance(
            raw_candidate, Mapping
        ):
            errors.append(f"historical dataclass removed: {wire_id}")
            continue
        if raw_baseline.get("legacy_revision") != raw_candidate.get(
            "legacy_revision"
        ):
            errors.append(f"legacy revision changed: {wire_id}")
        baseline_revisions = raw_baseline.get("revisions")
        candidate_revisions = raw_candidate.get("revisions")
        if not isinstance(baseline_revisions, Mapping) or not isinstance(
            candidate_revisions, Mapping
        ):
            errors.append(f"revision manifest is invalid: {wire_id}")
            continue
        for revision, fingerprint in baseline_revisions.items():
            if candidate_revisions.get(revision) != fingerprint:
                errors.append(
                    f"historical revision changed: {wire_id}@{revision}"
                )

    baseline_enums = baseline.get("enums")
    candidate_enums = candidate.get("enums")
    if not isinstance(baseline_enums, Mapping) or not isinstance(
        candidate_enums, Mapping
    ):
        errors.append("enum manifest is invalid")
    else:
        for wire_id, raw_values in baseline_enums.items():
            candidate_values = candidate_enums.get(wire_id)
            if not isinstance(raw_values, list) or not isinstance(
                candidate_values, list
            ):
                errors.append(f"historical enum removed: {wire_id}")
                continue
            if not set(raw_values).issubset(candidate_values):
                errors.append(f"historical enum values removed: {wire_id}")

    baseline_external = baseline.get("external")
    candidate_external = candidate.get("external")
    if not isinstance(baseline_external, Mapping) or not isinstance(
        candidate_external, Mapping
    ):
        errors.append("external manifest is invalid")
    else:
        for name, value in baseline_external.items():
            if candidate_external.get(name) != value:
                errors.append(f"historical external contract changed: {name}")
    return tuple(errors)


def _default_manifest() -> Path:
    return Path(__file__).with_name("matrix") / "runtime-persistence-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    candidate = load_manifest(args.manifest)
    errors = list(validate_runtime_persistence_manifest(candidate))
    if args.baseline is not None:
        errors.extend(validate_append_only(load_manifest(args.baseline), candidate))
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_manifest",
    "validate_append_only",
    "validate_runtime_persistence_manifest",
]
