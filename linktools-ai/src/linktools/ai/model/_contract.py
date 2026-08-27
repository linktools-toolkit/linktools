#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logical model binding and resolution contracts."""

from collections.abc import Mapping
from typing import Protocol

from pydantic_ai.models import Model

from ..core import JsonValue


class ModelBinding(Protocol):
    @property
    def route_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model_identity(self) -> str: ...

    @property
    def semantic_payload(self) -> Mapping[str, JsonValue]: ...

    @property
    def fingerprint(self) -> str: ...

    def materialize(self) -> Model: ...


class ModelResolver(Protocol):
    def resolve(self, route_id: str) -> ModelBinding: ...

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: "str | None" = None,
    ) -> ModelBinding: ...


__all__ = ["ModelBinding", "ModelResolver"]
