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
        ownership rules for any externally supplied clients. Native referenced
        input requires ModelInputValidator on the returned model or its public
        SDK wrapper chain; implementing this binding Protocol alone does not
        establish a contract for an arbitrary materialized model.
        """
        ...


@runtime_checkable
class ModelInputValidator(Protocol):
    """The actual materialized model's input permission and encoding contract.

    A custom model may implement this operation explicitly. Merely implementing
    ModelBinding or accepting SDK content types does not establish permission.
    Validation must not load references or mutate caller-owned input.
    """

    def validate_inputs(self, value: object) -> None: ...


__all__ = ["ModelBinding", "ModelInputValidator"]
