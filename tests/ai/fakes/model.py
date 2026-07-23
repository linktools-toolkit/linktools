#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared test fakes for the model layer.

Every test that drives a Runtime needs a FunctionModel + ModelRegistry; this
module centralizes that setup so individual tests stop duplicating it."""

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from linktools.ai.model.registry import ModelRegistry
from linktools.ai.model.resolver import ModelResolver


def echo_model_fn(text: str = "hello"):
    """A model function that always returns a fixed text response."""

    def _fn(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=f'{{"response": {{"message": "{text}"}}}}')]
        )

    return _fn


def make_router(text: str = "hello") -> ModelResolver:
    """Build a ModelResolver with a single registered FunctionModel."""
    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(echo_model_fn(text)))
    return ModelResolver(registry=registry)
