#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default Runtime and command composition share workspace input declarations."""
from linktools.ai.model import create_model

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.commands.ai._common import _local_model
from linktools.ai.workspace import Workspace


def test_default_runtime_reads_workspace_inputs_and_environment_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_types:\n  - audio\n  - document\n  - mime:audio/*\n  - mime:application/pdf\n")
    binding = _local_model(Workspace.load(tmp_path))
    assert binding.model_identity == "openai:testing/inputs"


    authored = create_model(protocol='openai',
        model="testing/inputs",
        input_types=["audio", "document", "mime:audio/*", "mime:application/pdf"],
    )
    assert binding.contract == authored.contract
    assert "audio/*" in binding.contract["input_mime_types"]
    assert "application/pdf" in binding.contract["input_mime_types"]


def test_default_runtime_ignores_unrelated_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    monkeypatch.setenv("OPENAI_VISION", "sometimes")
    binding = _local_model(Workspace.load(tmp_path))
    assert binding.contract["input_mime_types"] == []


def test_workspace_configuration_rejects_invalid_input_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_types: image/png\n")
    with pytest.raises(AIError) as raised:
        _local_model(Workspace.load(tmp_path))
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"reason": "model_input_configuration_invalid"}
