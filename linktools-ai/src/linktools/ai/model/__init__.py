#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""linktools.ai.model: model selection policy, the model registry, and the
resolver that resolves a policy to a live ResolvedModel (a real pydantic-ai
Model + revision). Re-exports the public surface so callers can
``from linktools.ai.model import ModelResolver`` etc. without reaching into
submodule paths."""

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
from .resolver import (
    ModelResolver,
    ModelRetryConfigurationError,
    ModelRoutingError,
    ResolvedModel,
)

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
    "ResolvedModel",
    "ModelRoutingError",
    "ModelRetryConfigurationError",
]
