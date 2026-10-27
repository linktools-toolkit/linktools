#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve effective output contracts before invoking model loaders."""

import pytest

from linktools.ai.agent import AgentCompiler, AssistantTextOutput, bind_output
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ModelBinding, create_model
from linktools.ai.spec import AgentSpec


@pytest.mark.parametrize("invalid_output", ["explicit", "default"])
def test_invalid_output_is_rejected_before_loading_model(invalid_output: str) -> None:
    loaded: list[bool] = []

    def load() -> ModelBinding:
        loaded.append(True)
        raise AIError(ErrorCode.MODEL_CONFIG_INVALID)

    spec = AgentSpec("default", output_schema={"type": "string"} if invalid_output == "default" else None)
    compiler = AgentCompiler(models={"default": load}, candidates=(), agents={spec.id: spec})
    compiled = compiler.compile(spec)
    with pytest.raises(AIError) as failed:
        compiler.bind(compiled, output=object if invalid_output == "explicit" else None)
    assert failed.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    assert loaded == []


def test_valid_explicit_output_overrides_unusable_default_and_loads_model() -> None:
    loaded: list[bool] = []

    def load() -> ModelBinding:
        loaded.append(True)
        return create_model(api="openai", model="gpt-test")

    spec = AgentSpec("default", output_schema={"type": "string"})
    compiler = AgentCompiler(models={"default": load}, candidates=(), agents={spec.id: spec})
    binding = compiler.bind(compiler.compile(spec), output=AssistantTextOutput)
    assert binding.output_binding == bind_output(AssistantTextOutput)
    assert loaded
