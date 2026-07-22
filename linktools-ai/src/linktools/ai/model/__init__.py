#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.model: model selection policy, the model registry, and the
router that resolves a policy to a live ModelBundle. Re-exports the public
surface so callers can ``from linktools.ai.model import ModelResolver`` etc.
without reaching into submodule paths."""

from .policy import ModelPolicy
from .registry import (
    ModelBundle,
    ModelClientUnavailable,
    ModelOutputError,
    ModelRegistry,
    ModelTurnLimitExceeded,
    RuntimeModelConfig,
    model_registry,
)
from .router import ModelGateway, ModelResolver, ModelRoutingError

__all__ = [
    "ModelPolicy",
    "ModelBundle",
    "ModelClientUnavailable",
    "ModelOutputError",
    "ModelRegistry",
    "ModelTurnLimitExceeded",
    "RuntimeModelConfig",
    "model_registry",
    "ModelResolver",
    "ModelGateway",
    "ModelRoutingError",
]
