#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output bindings backed by durable JSON schemas."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import BaseModel, ConfigDict
from pydantic_ai import StructuredDict
from pydantic_core import core_schema

from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode


class AssistantTextOutput(BaseModel):
    """Canonical default output containing assistant response text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str


ASSISTANT_TEXT_OUTPUT_SCHEMA_ID = "assistant-text"
ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION = 1


@dataclass(frozen=True, slots=True)
class OutputBinding:
    """Bind a runtime output representation to one durable JSON schema."""

    runtime_output_type: "type[object]"
    schema_id: str
    schema_revision: int
    schema_fingerprint: str
    _schema_payload: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_output_type, type):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if (
            not isinstance(self.schema_revision, int)
            or isinstance(self.schema_revision, bool)
            or self.schema_revision < 1
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        _validate_fingerprint(self.schema_fingerprint)
        try:
            schema = json.loads(self._schema_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if not isinstance(schema, dict) or canonical_sha256(schema) != self.schema_fingerprint:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        _validate_schema_definition(schema)

    @classmethod
    def create(
        cls,
        schema_id: str,
        runtime_output_type: "type[object]",
        schema: Mapping[str, JsonValue],
        *,
        schema_revision: int = 1,
    ) -> "OutputBinding":
        normalized_schema = _normalize_schema(schema)
        return cls(
            runtime_output_type,
            schema_id,
            schema_revision,
            canonical_sha256(normalized_schema),
            canonical_json_bytes(normalized_schema),
        )

    @property
    def schema_definition(self) -> "dict[str, JsonValue]":
        value = json.loads(self._schema_payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return cast("dict[str, JsonValue]", value)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "schema_id": self.schema_id,
                "schema_revision": self.schema_revision,
                "schema_fingerprint": self.schema_fingerprint,
            }
        )

    @property
    def descriptor(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "schema_id": self.schema_id,
            "schema_revision": self.schema_revision,
            "schema_fingerprint": self.schema_fingerprint,
        }


def bind_output(
    output: "type[BaseModel] | None" = None,
    *,
    schema_id: "str | None" = None,
    schema_revision: int = 1,
) -> OutputBinding:
    """Bind one output class to a self-contained durable schema contract."""
    if not isinstance(schema_revision, int) or isinstance(schema_revision, bool):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if output is None:
        if (
            schema_id is not None
            or schema_revision != ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return OutputBinding.create(
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            AssistantTextOutput,
            AssistantTextOutput.model_json_schema(),
            schema_revision=ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
        )
    if not isinstance(output, type) or not issubclass(output, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if output is AssistantTextOutput:
        if schema_id not in (None, ASSISTANT_TEXT_OUTPUT_SCHEMA_ID):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if schema_revision != ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return bind_output()
    if schema_id is not None or schema_revision != 1:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        schema = _normalize_schema(output.model_json_schema())
        _durable_runtime_type(schema)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    fingerprint = canonical_sha256(schema)
    return OutputBinding(
        output,
        f"schema:{fingerprint}",
        1,
        fingerprint,
        canonical_json_bytes(schema),
    )


def restore_output(descriptor: Mapping[str, JsonValue]) -> OutputBinding:
    """Restore an output binding from its persisted schema definition."""
    required = {"version", "schema_id", "schema_revision", "schema_fingerprint"}
    if not required.issubset(descriptor) or descriptor.get("version") != 1:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    schema_id = descriptor.get("schema_id")
    revision = descriptor.get("schema_revision")
    schema_fingerprint = descriptor.get("schema_fingerprint")
    if (
        not isinstance(schema_id, str)
        or not schema_id.strip()
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 1
        or not isinstance(schema_fingerprint, str)
    ):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    schema_value = descriptor.get("schema_definition")
    if schema_id == ASSISTANT_TEXT_OUTPUT_SCHEMA_ID:
        builtin = bind_output()
        if (
            revision != builtin.schema_revision
            or schema_fingerprint != builtin.schema_fingerprint
        ):
            raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        if schema_value is not None:
            try:
                schema = _normalize_schema(cast(Mapping[str, JsonValue], schema_value))
            except (AIError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
            if schema != builtin.schema_definition:
                raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
        return builtin
    if (
        revision != 1
        or not schema_id.startswith("schema:")
        or schema_id != f"schema:{schema_fingerprint}"
        or not isinstance(schema_value, Mapping)
    ):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    try:
        schema = _normalize_schema(schema_value)
        if canonical_sha256(schema) != schema_fingerprint:
            raise ValueError("output schema fingerprint mismatch")
        return OutputBinding(
            _durable_runtime_type(schema),
            schema_id,
            revision,
            schema_fingerprint,
            canonical_json_bytes(schema),
        )
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
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_ID",
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION",
    "AssistantTextOutput",
    "OutputBinding",
    "bind_output",
    "restore_output",
]
