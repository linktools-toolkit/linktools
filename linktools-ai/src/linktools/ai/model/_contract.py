#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical model binding and resolution contracts."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from pydantic_ai.models import Model

from ..core import JsonValue


@runtime_checkable
class ModelBinding(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def contract(self) -> Mapping[str, JsonValue]: ...

    def materialize(self) -> Model:
        """Build a model whose async context must be owned by the caller.

        Expected configuration failures raise AIError. SDK providers retain
        ownership rules for any externally supplied clients.
        """
        ...


@runtime_checkable
class _ThinkingControlSource(Protocol):
    def thinking_control(self) -> Mapping[str, JsonValue]: ...


def model_thinking_control(binding: ModelBinding) -> Mapping[str, JsonValue]:
    """Inspect declared control without materializing a model or resolving credentials.

    Custom bindings can provide thinking_control; otherwise control is unknown.
    This projection does not determine whether returned thinking is displayable.
    """
    if isinstance(binding, _ThinkingControlSource):
        return binding.thinking_control()
    return {"mode": "unknown", "enabled_by_default": None}


__all__ = ["ModelBinding", "model_thinking_control"]
