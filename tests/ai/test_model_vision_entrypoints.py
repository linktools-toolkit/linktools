#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision configuration reaches the existing ModelBinding semantic."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._factory import _build_default_models
from linktools.ai.workspace import Workspace
from linktools.cli import CommandError
from linktools.commands.ai.run import _openai_vision_default


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


def test_default_runtime_rejects_invalid_openai_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/vision")
    monkeypatch.setenv("OPENAI_VISION", "sometimes")

    with pytest.raises(AIError) as raised:
        _build_default_models(Workspace.load(tmp_path, workspace_id="workspace"))

    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"provider": "openai", "field": "vision"}


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
