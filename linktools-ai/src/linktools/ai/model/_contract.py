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


__all__ = ["ModelBinding"]
