#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model registry and resolver contracts."""

from ._connection import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelCredentialProvider,
    StaticModelCredentialProvider,
)
from ._materializer import ModelMaterializer, OpenAIModelMaterializer
from ._registry import ModelRegistry, ModelRegistrySnapshot, ModelRoute
from ._resolver import ModelResolver, SnapshotModelResolver

__all__ = [
    "ModelConnectionConfig", "ModelConnectionRegistry", "ModelCredentialProvider", "ModelMaterializer",
    "ModelRegistry", "ModelRegistrySnapshot", "ModelResolver", "ModelRoute", "OpenAIModelMaterializer", "SnapshotModelResolver",
    "StaticModelCredentialProvider",
]
