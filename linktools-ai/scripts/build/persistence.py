#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-time validation for the Runtime persistence compatibility contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from linktools.ai.core import JsonValue, canonical_json_bytes
from linktools.ai.runtime._message import (
    decode_model_messages,
    encode_model_messages,
)
from linktools.ai.runtime.state._codec import _runtime_persistence_manifest
from pydantic_ai.messages import ModelRequest, UserPromptPart


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


def _model_message_fixture_values() -> tuple[ModelRequest, ...]:
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


def validate_external_fixtures(
    manifest: Mapping[str, JsonValue],
    matrix_dir: str | Path,
) -> tuple[str, ...]:
    external = manifest.get("external")
    if not isinstance(external, Mapping):
        return ("external manifest is invalid",)
    model_contract = external.get("model_message")
    if not isinstance(model_contract, Mapping):
        return ("model_message external contract is invalid",)
    fixture_name = model_contract.get("fixture")
    if not isinstance(fixture_name, str) or not fixture_name:
        return ("model_message fixture is not declared",)
    path = Path(matrix_dir) / fixture_name
    if not path.is_file():
        return (f"missing external persistence fixture: {path}",)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return (f"invalid external persistence fixture: {path}: {error}",)
    expected = json.loads(
        encode_model_messages(_model_message_fixture_values()).decode("utf-8")
    )
    if value != expected:
        return ("model_message writer drifted from the frozen V1 fixture",)
    try:
        decoded = decode_model_messages(canonical_json_bytes(value))
    except Exception as error:
        return (f"model_message V1 fixture is no longer readable: {error}",)
    if decoded != _model_message_fixture_values():
        return ("model_message V1 fixture decodes to different semantics",)
    return ()


def _git(
    repository: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def load_git_baseline(
    manifest: str | Path,
    *,
    base_ref: str | None = None,
) -> dict[str, JsonValue] | None:
    """Load the manifest at the immutable branch/release comparison point.

    Feature branches compare against their merge-base with ``origin/master``.
    A master checkout compares against its first parent.  Repositories without
    Git history, including source distributions, simply skip this extra gate.
    """
    path = Path(manifest).resolve()
    repository = Path(__file__).resolve().parents[3]
    if not (repository / ".git").exists():
        return None
    try:
        relative = path.relative_to(repository.resolve()).as_posix()
    except ValueError:
        return None

    ref = base_ref or os.environ.get(
        "LINKTOOLS_PERSISTENCE_BASE_REF", "origin/master"
    )
    verified = _git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if verified.returncode != 0:
        return None
    head = _git(repository, "rev-parse", "HEAD")
    base = _git(repository, "merge-base", "HEAD", ref)
    if head.returncode != 0 or base.returncode != 0:
        return None
    head_sha = head.stdout.strip()
    base_sha = base.stdout.strip()
    if not head_sha or not base_sha:
        return None
    if head_sha == base_sha:
        parent = _git(repository, "rev-parse", "HEAD^")
        if parent.returncode != 0 or not parent.stdout.strip():
            return None
        base_sha = parent.stdout.strip()

    historical = _git(repository, "show", f"{base_sha}:{relative}")
    if historical.returncode != 0:
        return None
    try:
        value = json.loads(historical.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"historical runtime persistence manifest is invalid at {base_sha}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"historical runtime persistence manifest is not an object at {base_sha}"
        )
    return cast("dict[str, JsonValue]", value)


def _default_manifest() -> Path:
    return Path(__file__).with_name("matrix") / "runtime-persistence-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=_default_manifest())
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--baseline-ref")
    args = parser.parse_args()
    candidate = load_manifest(args.manifest)
    errors = list(validate_runtime_persistence_manifest(candidate))
    errors.extend(validate_external_fixtures(candidate, args.manifest.parent))
    if args.baseline is not None:
        baseline = load_manifest(args.baseline)
    else:
        try:
            baseline = load_git_baseline(
                args.manifest,
                base_ref=args.baseline_ref,
            )
        except ValueError as error:
            errors.append(str(error))
            baseline = None
    if baseline is not None:
        errors.extend(validate_append_only(baseline, candidate))
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "load_git_baseline",
    "load_manifest",
    "validate_append_only",
    "validate_external_fixtures",
    "validate_runtime_persistence_manifest",
]
