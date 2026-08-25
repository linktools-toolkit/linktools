#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-gate tests for exact Runtime persistence fixtures."""

import json
import sys
from pathlib import Path

import pytest
from scripts.check.ai import persistence

_MATRIX = (
    Path(__file__).resolve().parents[2]
    / "scripts/check/ai/matrix"
)


def test_checked_in_exact_persistence_fixtures_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["persistence"])
    assert persistence.main() == 0


@pytest.mark.parametrize(
    ("validator", "fixture"),
    [
        (persistence.validate_agent_binding_fixture, "runtime_agent_binding_snapshot_v1.json"),
        (persistence.validate_custom_wire_fixture, "runtime_custom_wire_v1.json"),
        (persistence.validate_model_message_fixture, "runtime_model_messages_v1.json"),
    ],
)
def test_exact_fixture_gate_rejects_missing_fixture(
    tmp_path: Path,
    validator,
    fixture: str,
) -> None:
    errors = validator(tmp_path)
    assert errors == (f"missing exact persistence fixture: {tmp_path / fixture}",)


@pytest.mark.parametrize(
    ("validator", "fixture"),
    [
        (persistence.validate_agent_binding_fixture, "runtime_agent_binding_snapshot_v1.json"),
        (persistence.validate_custom_wire_fixture, "runtime_custom_wire_v1.json"),
        (persistence.validate_model_message_fixture, "runtime_model_messages_v1.json"),
    ],
)
def test_exact_fixture_gate_rejects_malformed_fixture(
    tmp_path: Path,
    validator,
    fixture: str,
) -> None:
    (tmp_path / fixture).write_text("{", encoding="utf-8")
    errors = validator(tmp_path)
    assert errors == (
        (
            f"invalid persistence fixture: {tmp_path / fixture}: "
            "Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)"
        ),
    )


def test_custom_fixture_gate_rejects_shape_drift(tmp_path: Path) -> None:
    value = json.loads(
        (_MATRIX / "runtime_custom_wire_v1.json").read_text(encoding="utf-8")
    )
    value["task_node@1"]["fields"]["extra"] = None
    (tmp_path / "runtime_custom_wire_v1.json").write_text(
        json.dumps(value),
        encoding="utf-8",
    )
    assert persistence.validate_custom_wire_fixture(tmp_path) == (
        "Custom wire writer drifted from its fixture",
    )
