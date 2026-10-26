#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-specific model construction without route or client ownership."""

from collections.abc import Mapping, Sequence

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._contract import ModelBinding


def create_model(
    *,
    protocol: str,
    model: str,
    input_types: Sequence[str] = (),
    model_settings: Mapping[str, JsonValue] | None = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    max_retries: int = 2,
    retry_delay: "int | float" = 1.0,
    retry_max_delay: "int | float" = 60.0,
    retry_backoff_factor: "int | float" = 2.0,
    context_window: "int | None" = None,
) -> ModelBinding:
    """Construct a binding without creating a client or resolving credentials.

    ``model_settings`` accepts JSON-valued SDK settings, including provider-specific
    settings and ``extra_body`` for native fields. Request limits and timeouts
    belong only to ``model_settings``. Transport retries belong to this factory;
    ``model_settings.max_retries`` may only be omitted or zero. Authentication
    stays in connection parameters;
    request-body values are durable configuration and must not contain secrets.
    """
    if protocol == "openai":
        from ._openai import _OpenAIModelBinding as binding_type
    elif protocol == "openai-responses":
        from ._openai import _OpenAIResponsesModelBinding as binding_type
    elif protocol == "anthropic":
        from ._anthropic import _AnthropicModelBinding as binding_type
    else:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Unsupported model protocol; expected openai, openai-responses or anthropic",
            retryable=False,
            safe_details={"reason": "unsupported_model_protocol"},
        )
    return binding_type(
        model=model,
        input_types=input_types,
        model_settings={} if model_settings is None else model_settings,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_max_delay=retry_max_delay,
        retry_backoff_factor=retry_backoff_factor,
        context_window=context_window,
    )


__all__ = ["create_model"]
