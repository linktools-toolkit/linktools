#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capability authoring imports stay independent from Runtime initialization."""

import subprocess
import sys

import pytest

from linktools.ai.capability import (
    CheckpointCapability,
    TraceCapability,
)
from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeStoragePlan
from linktools.ai.runtime.state import RuntimeState

from .test_explicit_runtime_features import _FeatureModels, _features


def test_capability_import_does_not_initialize_runtime_package() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from linktools.ai.capability import ConversationCapability, "
            "CheckpointCapability, TraceCapability; "
            "assert 'linktools.ai.runtime' not in sys.modules",
        ],
        check=True,
    )


def test_runtime_has_no_duplicate_capability_exports() -> None:
    import linktools.ai.runtime as runtime

    for name in (
        "ConversationCapability",
        "CheckpointCapability",
        "TraceCapability",
        "MetricsCapability",
    ):
        assert name not in runtime.__all__
        assert not hasattr(runtime, name)
    import linktools.ai.capability as capability
    assert not hasattr(capability, "MetricsCapability")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capability", [CheckpointCapability, TraceCapability]
)
async def test_invalid_module_route_is_rejected_before_storage_initialization(
    capability, monkeypatch
) -> None:
    async def unexpected_initialize(*args, **kwargs):
        pytest.fail("invalid declarations must fail before storage I/O")

    def unexpected_model_capture(*args, **kwargs):
        pytest.fail("invalid declarations must fail before model resolution")

    monkeypatch.setattr(RuntimeState, "initialize", unexpected_initialize)
    monkeypatch.setattr(_FeatureModels, "capture", unexpected_model_capture)
    features = _features(capability(storage=object()))
    with pytest.raises(TypeError, match="storage must be StorageRoute"):
        async with Runtime.open(
            "invalid-route",
            storage=RuntimeStorage.in_memory(),
            models=_FeatureModels(),
            features=features,
        ):
            pytest.fail("invalid declaration must not activate Runtime")


def test_invalid_session_route_is_rejected_by_root_storage_plan() -> None:
    with pytest.raises(ValueError, match="RuntimeStoragePlan contains an invalid route"):
        RuntimeStoragePlan(conversation=object())
