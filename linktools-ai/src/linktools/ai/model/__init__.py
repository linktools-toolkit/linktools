#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model bindings, resolver, and binding contracts."""

from pydantic_ai.messages import AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType

from ._contract import ModelBinding, model_thinking_control
from ._client_request import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from ._errors import model_binding_error
from ._resolver import ModelResolver
from ._factory import create_model
from ._input import validate_model_input, validate_model_inputs

from ._user_input import (StoredUserInput, CanonicalUserInput, MaterializedUserContent, append_user_input_text, UserPromptInput, validate_user_content, validate_user_input, normalize_input_files)

__all__ = [
    "StoredUserInput",
    "CanonicalUserInput",
    "MaterializedUserContent",
    "append_user_input_text",
    "UserPromptInput",
    "validate_user_content",
    "validate_user_input",
    "normalize_input_files",

    "AudioMediaType", "DocumentMediaType", "ImageMediaType", "VideoMediaType",
    "model_thinking_control",
    "ModelBinding", "create_model", "ModelResolver", "model_binding_error",
    "ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse",
    "validate_model_input", "validate_model_inputs",
]
