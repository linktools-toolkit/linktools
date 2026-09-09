#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression contracts for review-driven boundary cleanup."""

import inspect

import pytest
from pydantic_ai_harness.memory import MemoryOperation

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import _memory as runtime_memory
from linktools.ai.runtime._capabilities import (
    _RuntimeStepPersistence,
    compose_platform_capabilities,
)
from linktools.ai.runtime._harness_memory import HarnessMemoryStoreAdapter
from linktools.ai.runtime.state import StagingStepStore
import linktools.ai.runtime.state as runtime_state


def test_platform_composition_has_no_dead_conversation_id() -> None:
    assert "conversation_id" not in inspect.signature(compose_platform_capabilities).parameters


def test_runtime_step_persistence_requires_explicit_harness_adapter() -> None:
    with pytest.raises(TypeError):
        _RuntimeStepPersistence(store=StagingStepStore(), tool_operations=object())


def test_harness_memory_mutation_requires_outer_operation_identity() -> None:
    adapter = object.__new__(HarnessMemoryStoreAdapter)
    with pytest.raises(AIError) as raised:
        adapter._operation(MemoryOperation(id="operation", fingerprint="0" * 64))
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_runtime_memory_has_no_legacy_compatibility_helpers() -> None:
    assert not hasattr(runtime_memory, "normalize_memory_file")
    assert not hasattr(runtime_memory, "memory_operation_fingerprint")


def test_state_public_surface_does_not_expose_new_internal_step_contracts() -> None:
    assert "StagingStepStore" not in runtime_state.__all__
    assert "StepStore" not in runtime_state.__all__
    assert "ToolRepositoryImpl" not in runtime_state.__all__
