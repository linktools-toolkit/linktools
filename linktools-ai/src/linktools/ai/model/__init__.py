#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model registry, resolver, and binding contracts."""

from ._contract import ModelBinding, ModelResolver
from ._client_request import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from ._errors import model_binding_error
from ._registry import ModelRegistry
from ._input import (
    ModelInputCapability, ModelInputDecision, ModelInputKind, ModelInputStatus, ModelInputTransport,
    UnknownModelInputs, decide_model_input, validate_model_input, validate_model_inputs,
)

__all__ = [
    "ModelBinding", "ModelRegistry", "ModelResolver", "model_binding_error",
    "ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse",
    "ModelInputCapability", "ModelInputDecision", "ModelInputKind", "ModelInputStatus", "ModelInputTransport",
    "UnknownModelInputs", "decide_model_input", "validate_model_input", "validate_model_inputs",
]
