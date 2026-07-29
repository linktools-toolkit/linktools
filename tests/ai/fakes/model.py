#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared test fakes for the model layer.

Every test that drives a Runtime needs a FunctionModel + ModelRegistry; this
module centralizes that setup so individual tests stop duplicating it."""

import asyncio

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


def make_raising_router(exc: "Exception | None" = None) -> ModelResolver:
    """A ModelResolver whose model raises immediately -- used to prove an
    unexpected (programming/config/protocol) error aborts the run with a
    persisted RunRecord.error instead of stranding it in RUNNING."""

    async def _fn(messages, info: AgentInfo) -> ModelResponse:
        raise exc if exc is not None else RuntimeError("boom")

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_fn))
    return ModelResolver(registry=registry)


def make_hanging_router(started: "asyncio.Event | None" = None) -> ModelResolver:
    """A ModelResolver whose model suspends forever until its driving
    asyncio.Task is cancelled -- used to prove a cancel() actually interrupts
    a live model call rather than only flipping a status flag nothing is
    awaiting on. ``started`` (if given) is set once the model function is
    entered, so a caller can wait for the run to actually reach the hang
    before cancelling it."""

    async def _fn(messages, info: AgentInfo) -> ModelResponse:
        if started is not None:
            started.set()
        await asyncio.Event().wait()  # never set: hangs until task.cancel()
        raise AssertionError("unreachable")  # pragma: no cover

    registry = ModelRegistry()
    registry.register("test-model", model=FunctionModel(_fn))
    return ModelResolver(registry=registry)
