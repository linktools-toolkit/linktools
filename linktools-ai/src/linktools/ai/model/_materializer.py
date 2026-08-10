#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model deployment materialization boundary."""

from typing import Protocol

from pydantic_ai.models import Model

from ._connection import ModelConnectionConfig
from ._registry import ModelRoute


class ModelMaterializer(Protocol):
    def materialize(self, route: ModelRoute, connection: "ModelConnectionConfig | None") -> Model: ...


__all__ = ["ModelMaterializer"]
