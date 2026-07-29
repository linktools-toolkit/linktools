"""Strict decoding helpers for model configuration."""

from typing import Any

from ..errors import InvalidSpecError
from ..spec.parsing import StrictConfigReader
from .policy import ModelPolicy


def parse_model_policy(payload: dict[str, Any]) -> ModelPolicy:
    reader = StrictConfigReader(
        payload,
        allowed={
            "primary",
            "fallbacks",
            "request_retries",
            "timeout_seconds",
            "max_tokens",
            "budget",
        },
        context="model policy",
    )
    primary = reader.required_str("primary").strip()
    if not primary:
        raise InvalidSpecError("model policy primary must not be empty")
    return ModelPolicy(
        primary=primary,
        fallbacks=reader.string_tuple("fallbacks", default=()),
        request_retries=reader.non_negative_int("request_retries", default=0),
        timeout_seconds=reader.positive_number("timeout_seconds", default=30.0),
        max_tokens=reader.positive_int("max_tokens"),
        budget=reader.non_negative_decimal("budget"),
    )


__all__ = ["parse_model_policy"]
