#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default Runtime and command composition share workspace input declarations."""
from linktools.ai.model import create_model_binding

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.commands.ai._common import _ModelOptions
from linktools.ai.workspace import Workspace


def test_default_runtime_reads_workspace_inputs_and_environment_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    config = tmp_path / ".linktools" / "config.yaml"
    config.parent.mkdir()
    config.write_text("input_types:\n  - audio\n  - document\n  - mime:audio/*\n  - mime:application/pdf\n")
    binding = _ModelOptions.load(Workspace.load(tmp_path)).create_model_binding()
    assert binding.model_identity == "openai:testing/inputs"


    authored = create_model_binding(api='openai',
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
    monkeypatch.setenv("VISION", "true")
    monkeypatch.setenv("LINKTOOLS_VISION", "true")
    binding = _ModelOptions.load(Workspace.load(tmp_path)).create_model_binding()
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
        _ModelOptions.load(Workspace.load(tmp_path)).create_model_binding()
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"reason": "model_input_configuration_invalid"}


@pytest.mark.parametrize(("api", "override"), [
    ("openai", None), ("openai", "true"), ("openai", "false"),
    ("anthropic", "1"), ("anthropic", "0"),
])
def test_local_vision_changes_only_image_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api: str, override: str | None,
) -> None:
    monkeypatch.setenv("LINKTOOLS_MODEL_API", api)
    monkeypatch.setenv(f"{api.upper()}_MODEL", "testing/inputs")
    if override is None:
        monkeypatch.delenv("LINKTOOLS_MODEL_VISION", raising=False)
    else:
        monkeypatch.setenv("LINKTOOLS_MODEL_VISION", override)
    values = ["mime:application/pdf", "mime:audio/*", "mime:IMAGE/PNG"]
    if override in {"0", "false"}:
        values += ["image", "mime:image/*"]
    workspace = Workspace(tmp_path, {"input_types": values})

    expected = ["mime:application/pdf", "mime:audio/*"]
    if override in {"1", "true"}:
        expected.append("image")
    elif override is None:
        expected.append("mime:image/png")
    binding = _ModelOptions.load(workspace).create_model_binding()
    authored = create_model_binding(api=api, model="testing/inputs", input_types=expected)
    assert binding.contract == authored.contract
    assert "vision" not in binding.contract


def test_local_vision_rejects_invalid_boolean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/inputs")
    monkeypatch.setenv("LINKTOOLS_MODEL_VISION", "sometimes")
    with pytest.raises(AIError) as raised:
        _ModelOptions.load(Workspace(tmp_path, {})).create_model_binding()
    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID


@pytest.mark.parametrize("process_env", [None, "0"])
def test_local_vision_reads_cli_project_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process_env: str | None,
) -> None:
    from linktools.commands.ai import _common
    from linktools.core import Environ

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MODEL_API", raising=False)
    monkeypatch.delenv("MODEL_VISION", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    if process_env is None:
        monkeypatch.delenv("LINKTOOLS_MODEL_VISION", raising=False)
    else:
        monkeypatch.setenv("LINKTOOLS_MODEL_API", "openai")
        monkeypatch.setenv("LINKTOOLS_MODEL_VISION", process_env)
    (tmp_path / ".linktools.json").write_text('''{
        "env": {
            "LINKTOOLS_MODEL_API": "anthropic",
            "ANTHROPIC_MODEL": "testing/profile-inputs",
            "LINKTOOLS_MODEL_VISION": "1"
        }
    }''', encoding="utf-8")
    monkeypatch.setattr(_common, "environ", Environ())

    binding = _ModelOptions.load(Workspace.load(tmp_path)).create_model_binding()
    assert binding.model_identity == "anthropic:testing/profile-inputs"
    assert "image/png" in binding.contract["input_mime_types"]
