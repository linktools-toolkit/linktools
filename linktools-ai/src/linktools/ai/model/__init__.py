#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model construction, custom binding contracts, and model input types."""

from pydantic_ai.messages import AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType

from ._contract import ModelBinding, ModelInputValidator, model_thinking_control
from ._client_request import ModelClientRequest, ModelClientRequestRecorder, ModelClientResponse
from ._errors import model_binding_error, model_request_error
from ._stream_attempt import ModelStreamAttempt
from ._factory import create_model
from ._fallback import FallbackBinding
from ._input import validate_model_input, validate_model_inputs
from ._referenced_content import ReferencedContent
from ._referenced_input import ReferencedInputModel
from ._response import project_model_attempts, project_model_response_attempts

from ._user_input import (StoredUserInput, CanonicalUserInput, MaterializedUserContent, append_user_input_text, UserPromptInput, validate_user_content, validate_user_input, normalize_input_files)

__all__ = [
    "project_model_attempts", "project_model_response_attempts",
    "ReferencedContent", "ReferencedInputModel", "ModelInputValidator", "model_request_error",
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
    "ModelBinding", "FallbackBinding", "create_model", "model_binding_error",
    "ModelClientRequest", "ModelClientRequestRecorder", "ModelClientResponse", "ModelStreamAttempt",
    "validate_model_input", "validate_model_inputs",
]
