#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default Runtime and command composition share workspace input declarations."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.commands.ai._common import _local_models
from linktools.ai.workspace import Workspace


def test_default_runtime_reads_workspace_inputs_and_environment_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_mime_types:\n  - image/png\n")
    binding = _local_models(Workspace.load(tmp_path)).capture().resolve("default")
    assert binding.model_identity == "openai:testing/inputs"
    assert binding.contract["input_mime_types"] == ["image/png"]



def test_default_runtime_rejects_invalid_legacy_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    monkeypatch.setenv("OPENAI_VISION", "sometimes")
    with pytest.raises(AIError, match="input_mime_types") as raised:
        _local_models(Workspace.load(tmp_path))
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"reason": "legacy_vision_configuration"}


def test_workspace_configuration_rejects_removed_input_declarations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_capabilities:\n  - media_type: image/png\n    status: supported\n")
    with pytest.raises(AIError) as raised:
        _local_models(Workspace.load(tmp_path))
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"reason": "model_input_configuration_invalid"}
