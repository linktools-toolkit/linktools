#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model registry, resolver, and binding contracts."""

from pydantic_ai.messages import AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType

from ._contract import ModelBinding, ModelResolver
from ._client_request import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from ._errors import model_binding_error
from ._registry import ModelRegistry
from ._input import validate_model_input, validate_model_inputs

__all__ = [
    "AudioMediaType", "DocumentMediaType", "ImageMediaType", "VideoMediaType",
    "ModelBinding", "ModelRegistry", "ModelResolver", "model_binding_error",
    "ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse",
    "validate_model_input", "validate_model_inputs",
]
