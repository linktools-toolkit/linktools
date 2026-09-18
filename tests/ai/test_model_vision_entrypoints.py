#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vision configuration reaches the model binding semantic."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.commands.ai._common import _local_models
from linktools.ai.workspace import Workspace


def test_default_runtime_reads_openai_vision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "testing/vision")
    monkeypatch.setenv("OPENAI_VISION", "true")

    binding = _local_models(
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
        _local_models(Workspace.load(tmp_path, workspace_id="workspace"))

    assert raised.value.code is ErrorCode.MODEL_CONFIG_INVALID
    assert raised.value.safe_details == {"provider": "openai", "field": "vision"}
