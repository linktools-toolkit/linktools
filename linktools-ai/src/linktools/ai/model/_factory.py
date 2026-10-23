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
    profile: Mapping[str, JsonValue] | None = None,
    base_url: "str | None" = None,
    api_key: "str | None" = None,
    timeout: "int | float | None" = None,
    max_retries: int = 2,
    retry_delay: "int | float" = 1.0,
    max_tokens: "int | None" = None,
) -> ModelBinding:
    """Construct a model binding without creating a client or resolving credentials."""
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
        profile=profile,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        max_tokens=max_tokens,
    )


__all__ = ["create_model"]
