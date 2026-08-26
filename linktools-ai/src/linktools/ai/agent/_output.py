#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output bindings backed by one durable JSON-schema contract."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict
from pydantic_ai import StructuredDict
from pydantic_core import core_schema

from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode


class AssistantTextOutput(BaseModel):
    """Canonical Runtime representation of plain assistant text."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str


OutputMode = Literal["text", "structured"]


@dataclass(frozen=True, slots=True)
class OutputBinding:
    """Bind the model output mode to one self-contained durable JSON schema."""

    mode: OutputMode
    _schema_payload: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if self.mode not in {"text", "structured"}:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        schema = self.schema_definition
        _validate_schema_definition(schema)

    @classmethod
    def create(cls, mode: OutputMode, schema: Mapping[str, JsonValue]) -> "OutputBinding":
        normalized = _normalize_schema(schema)
        return cls(mode, canonical_json_bytes(normalized))

    @property
    def schema_definition(self) -> "dict[str, JsonValue]":
        try:
            value = json.loads(self._schema_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if not isinstance(value, dict):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return cast("dict[str, JsonValue]", value)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract": "output-v1",
                "mode": self.mode,
                "schema": self.schema_definition,
            }
        )

    @property
    def runtime_output_type(self) -> "type[object]":
        if self.mode == "text":
            return AssistantTextOutput
        return _durable_runtime_type(self.schema_definition)


def bind_output(output: "type[BaseModel] | None" = None) -> OutputBinding:
    """Create the exact durable output contract for a fresh execution."""
    if output is None or output is AssistantTextOutput:
        return OutputBinding.create("text", AssistantTextOutput.model_json_schema())
    if not isinstance(output, type) or not issubclass(output, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        schema = _normalize_schema(output.model_json_schema())
        _durable_runtime_type(schema)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    return OutputBinding.create("structured", schema)


def restore_output(mode: JsonValue, schema: JsonValue) -> OutputBinding:
    """Restore an output binding only from its historical v1 semantics."""
    if mode not in {"text", "structured"} or not isinstance(schema, Mapping):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    try:
        binding = OutputBinding.create(
            cast(OutputMode, mode),
            cast(Mapping[str, JsonValue], schema),
        )
        if binding.mode == "text" and binding.schema_definition != bind_output().schema_definition:
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        if binding.mode == "structured":
            _durable_runtime_type(binding.schema_definition)
        return binding
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error


def _durable_runtime_type(
    schema: Mapping[str, JsonValue],
) -> "type[object]":
    normalized = _normalize_schema(schema)
    validator = _schema_validator(normalized)
    try:
        structured_type = StructuredDict(cast(object, normalized))
    except Exception as error:
        raise AIError(
            ErrorCode.OUTPUT_CONTRACT_INVALID,
            safe_details={"reason": "output_schema_not_durable"},
        ) from error

    def validate(value: object) -> object:
        try:
            validator.validate(value)
        except JsonSchemaValidationError as error:
            raise ValueError("output does not match durable JSON schema") from error
        return value

    class DurableStructuredOutput(structured_type):  # type: ignore[misc, valid-type]
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler: Any,
        ) -> core_schema.CoreSchema:
            del cls, source_type, handler
            return core_schema.no_info_after_validator_function(
                validate,
                core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(),
                    values_schema=core_schema.any_schema(),
                ),
            )

    return cast("type[object]", DurableStructuredOutput)


def _schema_validator(
    schema: Mapping[str, JsonValue],
) -> Draft202012Validator:
    _validate_schema_definition(schema)
    return Draft202012Validator(dict(schema))


def _validate_schema_definition(schema: Mapping[str, JsonValue]) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error


def _normalize_schema(value: object) -> "dict[str, JsonValue]":
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        schema = json.loads(
            canonical_json_bytes(cast(JsonValue, dict(value))).decode("utf-8")
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(schema, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return cast("dict[str, JsonValue]", schema)


def _validate_fingerprint(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)


__all__ = [
    "AssistantTextOutput",
    "OutputBinding",
    "OutputMode",
    "bind_output",
    "restore_output",
]
