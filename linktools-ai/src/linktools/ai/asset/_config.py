#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict configuration values used by asset loaders and codecs."""

import math
import os
from collections.abc import Collection, Mapping
from decimal import Decimal, InvalidOperation

from ..core import JsonValue
from ..core.errors import AssetError, ErrorCode


class StrictConfigReader:
    def __init__(
        self,
        values: Mapping[str, JsonValue],
        *,
        allowed: "Collection[str] | None" = None,
        context: str = "asset",
    ) -> None:
        self._values = values
        self._context = context
        if allowed is not None:
            unknown = sorted(set(values) - set(allowed))
            if unknown:
                raise AssetError(
                    f"{context}: unknown fields: {', '.join(unknown)}",
                    ErrorCode.ASSET_CONFIG_TYPE_INVALID,
                )

    @property
    def context(self) -> str:
        return self._context

    def _present(self, name: str) -> "tuple[bool, JsonValue]":
        if name not in self._values:
            return False, None
        value = self._values[name]
        if value is None:
            raise AssetError(
                f"{self._context}: {name} must not be null",
                ErrorCode.ASSET_CONFIG_TYPE_INVALID,
            )
        return True, value

    def str_or_bool(self, name: str, default: "str | bool | None" = None) -> "str | bool | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool):
            return value
        if not isinstance(value, str) or not value.strip():
            raise AssetError(
                f"{self._context}: {name} must be a non-empty string or boolean",
                ErrorCode.ASSET_CONFIG_TYPE_INVALID,
            )
        if value.startswith("${") and value.endswith("}"):
            environment_name = value[2:-1]
            if not environment_name or "${" in environment_name:
                raise AssetError(
                    f"{self._context}: {name} has an invalid environment reference",
                    ErrorCode.ASSET_CONFIG_TYPE_INVALID,
                )
            environment_value = os.environ.get(environment_name)
            if environment_value is None:
                raise AssetError(
                    f"{self._context}: environment variable {environment_name} is missing",
                    ErrorCode.ASSET_ENV_MISSING,
                )
            return environment_value
        if "${" in value or "}" in value:
            raise AssetError(
                f"{self._context}: {name} has an invalid environment reference",
                ErrorCode.ASSET_CONFIG_TYPE_INVALID,
            )
        return value.strip()

    def required_str(self, name: str) -> str:
        value = self.str_or_bool(name)
        if not isinstance(value, str):
            raise AssetError(f"{self._context}: {name} must be a string", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return value

    def optional_str(self, name: str) -> "str | None":
        value = self.str_or_bool(name)
        return value if isinstance(value, str) else None

    def bool(self, name: str, default: "bool | None" = None) -> "bool | None":
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, bool):
            raise AssetError(f"{self._context}: {name} must be a boolean", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return value

    def non_negative_int(self, name: str, default: "int | None" = None) -> "int | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AssetError(f"{self._context}: {name} must be a non-negative integer", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return value

    def positive_int(self, name: str, default: "int | None" = None) -> "int | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AssetError(f"{self._context}: {name} must be a positive integer", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return value

    def positive_number(self, name: str, default: "float | None" = None) -> "float | None":
        present, value = self._present(name)
        if not present:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise AssetError(f"{self._context}: {name} must be a positive number", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return float(value)

    def non_negative_decimal(self, name: str) -> "Decimal | None":
        present, value = self._present(name)
        if not present:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AssetError(f"{self._context}: {name} must be a number", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        try:
            decimal = Decimal(str(value))
        except (InvalidOperation, ValueError) as error:
            raise AssetError(f"{self._context}: {name} must be a valid number", ErrorCode.ASSET_CONFIG_TYPE_INVALID) from error
        if not decimal.is_finite() or decimal < 0:
            raise AssetError(f"{self._context}: {name} must be finite and non-negative", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return decimal

    def mapping(self, name: str) -> "dict[str, JsonValue] | None":
        present, value = self._present(name)
        if not present:
            return None
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise AssetError(f"{self._context}: {name} must be an object", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return value

    def string_tuple(self, name: str, *, default: "tuple[str, ...] | None" = None) -> "tuple[str, ...] | None":
        present, value = self._present(name)
        if not present:
            return default
        if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
            raise AssetError(f"{self._context}: {name} must be a list of non-empty strings", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
        return tuple(item.strip() for item in value)


def resolved_name(reader: StrictConfigReader, entity_id: str) -> str:
    value = reader.optional_str("name")
    if value is None:
        return entity_id
    if not value.strip():
        raise AssetError(f"{reader.context}: name must not be empty", ErrorCode.ASSET_CONFIG_TYPE_INVALID)
    return value.strip()


__all__ = ["StrictConfigReader", "resolved_name"]
