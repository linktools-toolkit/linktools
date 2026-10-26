#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Capability authoring imports stay independent from Runtime initialization."""

import subprocess
import sys

import pytest

from linktools.ai.capability import (
    CheckpointCapability,
    TraceCapability,
)
from linktools.ai.persistence import PersistencePlan

from .test_explicit_runtime_features import _features


def test_capability_import_does_not_initialize_runtime_package() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from linktools.ai.capability import ExecutionCapability, "
            "CheckpointCapability, TraceCapability; "
            "assert 'linktools.ai.runtime' not in sys.modules",
        ],
        check=True,
    )


def test_runtime_has_no_duplicate_capability_exports() -> None:
    import linktools.ai.runtime as runtime

    for name in (
        "ExecutionCapability",
        "CheckpointCapability",
        "TraceCapability",
        "MetricsCapability",
    ):
        assert name not in runtime.__all__
        assert not hasattr(runtime, name)
    import linktools.ai.capability as capability
    assert not hasattr(capability, "MetricsCapability")
    assert not hasattr(capability, "ConversationCapability")


@pytest.mark.parametrize("route", ("checkpoint", "trace"))
def test_invalid_module_route_is_rejected_before_storage_initialization(
    route: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_initialize(*args, **kwargs):
        pytest.fail("invalid declarations must fail before storage I/O")

    def unexpected_model_capture(*args, **kwargs):
        pytest.fail("invalid declarations must fail before model resolution")

    monkeypatch.setattr(Persistence, "initialize", unexpected_initialize)
    features = _features(CheckpointCapability() if route == "checkpoint" else TraceCapability())
    features.model("default", unexpected_model_capture)
    with pytest.raises(ValueError, match="PersistencePlan contains an invalid route"):
        PersistencePlan(**{route: object()})


def test_invalid_session_route_is_rejected_by_root_storage_plan() -> None:
    with pytest.raises(ValueError, match="PersistencePlan contains an invalid route"):
        PersistencePlan(conversation=object())
