#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Traceability matrix loading."""

import json
from pathlib import Path


def load_matrix(path: "str | Path") -> "dict[str, object]":
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_matrix(
    matrix: "dict[str, object]",
    source_root: "str | Path",
    test_root: "str | Path",
) -> "tuple[str, ...]":
    """Check that every requirement points at real implementation and tests."""
    source = Path(source_root)
    package_root = source.parents[2]
    _tests = Path(test_root)
    errors: "list[str]" = []
    requirements = matrix.get("requirements", {})
    if not isinstance(requirements, dict) or not requirements:
        return ("matrix has no requirements",)
    for requirement_id, value in requirements.items():
        if not isinstance(value, dict):
            errors.append(f"{requirement_id}: invalid requirement entry")
            continue
        implementations = value.get("implementation", ())
        linked_tests = value.get("tests", ())
        for implementation in implementations if isinstance(implementations, list) else ():
            if not (source / implementation).exists() and not (package_root / implementation).exists():
                errors.append(f"{requirement_id}: missing implementation {implementation}")
        for test_id in linked_tests if isinstance(linked_tests, list) else ():
            if not isinstance(test_id, str) or not test_id.startswith("MT-"):
                errors.append(f"{requirement_id}: invalid test id {test_id}")
    return tuple(errors)


__all__ = ["load_matrix", "validate_matrix"]
