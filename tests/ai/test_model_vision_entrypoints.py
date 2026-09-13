#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision configuration reaches the existing ModelBinding semantic."""

from pathlib import Path

import pytest

from linktools.ai.runtime._factory import _build_default_models
from linktools.commands.ai.run import _openai_vision_default
from linktools.cli import CommandError
from linktools.ai.workspace import Workspace


def test_default_runtime_reads_openai_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/vision")
    monkeypatch.setenv("OPENAI_VISION", "true")

    binding = _build_default_models(
        Workspace.load(tmp_path, workspace_id="workspace")
    ).snapshot().resolve("default")

    assert binding.vision is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    (("true", True), ("false", False), ("1", True), ("0", False)),
)
def test_cli_vision_env_uses_strict_boolean_cast(
    raw: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_VISION", raw)
    assert _openai_vision_default() is expected


def test_cli_vision_env_rejects_invalid_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_VISION", "sometimes")
    with pytest.raises(CommandError, match="OPENAI_VISION must be a boolean"):
        _openai_vision_default()
