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
from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic_ai import StructuredDict
from pydantic_core import PydanticCustomError, core_schema

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode, ErrorDiagnostics
from ..spec import canonicalize_json_schema, canonicalize_pydantic_model_schema


class AssistantTextOutput(BaseModel):
    """Canonical Runtime representation of plain assistant text."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str


_TEXT_OUTPUT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


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
        if self.mode == "text" and schema != _TEXT_OUTPUT_SCHEMA:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)

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

    def validate_payload(self, value: JsonValue) -> None:
        """Validate one final JSON payload against this durable output contract."""
        try:
            _schema_validator(self.schema_definition).validate(value)
        except JsonSchemaValidationError as error:
            diagnostic = _OutputSchemaDiagnostic.from_schema_error(
                error, _declared_property_names(self.schema_definition)
            )
            raise diagnostic.to_error(ErrorCode.OUTPUT_VALIDATION_FAILED) from None

    @property
    def runtime_output_type(self) -> "type[object]":
        if self.mode == "text":
            return AssistantTextOutput
        return _durable_runtime_type(self.schema_definition)


def bind_output(output: "type[BaseModel] | None" = None) -> OutputBinding:
    """Create the exact durable output contract for a fresh execution."""
    if output is None or output is AssistantTextOutput:
        return OutputBinding.create("text", _TEXT_OUTPUT_SCHEMA)
    if not isinstance(output, type) or not issubclass(output, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        schema = canonicalize_pydantic_model_schema(output)
        _durable_runtime_type(schema)
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    return OutputBinding.create("structured", schema)


def restore_output(mode: JsonValue, schema: JsonValue) -> OutputBinding:
    """Restore an output binding only from its historical v1 semantics."""
    if mode not in {"text", "structured"} or not isinstance(schema, Mapping):
        raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE)
    try:
        binding = OutputBinding.create(
            cast(OutputMode, mode),
            cast(Mapping[str, JsonValue], schema),
        )
        if binding.mode == "structured":
            _durable_runtime_type(binding.schema_definition)
        return binding
    except AIError:
        raise
    except Exception as error:
        raise AIError(ErrorCode.AGENT_BINDING_UNAVAILABLE) from error


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

    declared_names = _declared_property_names(normalized)

    def validate(value: object, handler: core_schema.ValidatorFunctionWrapHandler) -> object:
        try:
            value = handler(value)
        except ValidationError:
            diagnostic = _OutputSchemaDiagnostic(
                "$", "type", "object with string property names", _json_type(value)
            )
            raise diagnostic.to_validation_error() from None
        try:
            validator.validate(value)
        except JsonSchemaValidationError as error:
            diagnostic = _OutputSchemaDiagnostic.from_schema_error(error, declared_names)
            raise diagnostic.to_validation_error() from None
        return value

    class DurableStructuredOutput(structured_type):  # type: ignore[misc, valid-type]
        @classmethod
        def __get_pydantic_core_schema__(
            cls,
            source_type: Any,
            handler: Any,
        ) -> core_schema.CoreSchema:
            del cls, source_type, handler
            return core_schema.no_info_wrap_validator_function(
                validate,
                core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(),
                    values_schema=core_schema.any_schema(),
                ),
            )

    return cast("type[object]", DurableStructuredOutput)


@dataclass(frozen=True, slots=True)
class _OutputSchemaDiagnostic:
    path: str
    rule: str
    expected_type: str | None = None
    actual_type: str | None = None

    @classmethod
    def from_schema_error(
        cls, error: JsonSchemaValidationError, declared_names: frozenset[str]
    ) -> "_OutputSchemaDiagnostic":
        path = list(error.absolute_path)
        if error.validator == "required" and isinstance(error.instance, Mapping):
            missing = next(
                (name for name in error.validator_value if name not in error.instance), None
            )
            if missing is not None:
                path.append(missing)
        segments = ["$"]
        for segment in path[:8]:
            if isinstance(segment, int):
                segments.append(f"[{segment}]")
            elif isinstance(segment, str) and segment in declared_names:
                segments.append(f"[{json.dumps(segment[:64], ensure_ascii=True)}]")
            else:
                segments.append('["<key>"]')
        if len(path) > 8:
            segments.append("[...]")
        rule = error.validator if error.validator in Draft202012Validator.VALIDATORS else "falseSchema"
        expected_type = None
        actual_type = None
        if rule == "type":
            types = error.validator_value
            expected_type = ", ".join(types) if isinstance(types, list) else types
            actual_type = _json_type(error.instance)
        location = "".join(segments)
        if len(location) > 512:
            location = location[:507] + "[...]"
        return cls(location, rule, expected_type, actual_type)

    def __str__(self) -> str:
        message = f"output does not match durable JSON schema at {self.path}: {self.rule}"
        if self.expected_type is not None:
            message += f" (expected {self.expected_type}, got {self.actual_type})"
        return message

    def to_validation_error(self) -> ValidationError:
        # hide_input alone is lost when an enclosing Pydantic validator rewraps
        # the failure. Omit the input itself so SDK retry payloads stay safe too.
        return ValidationError.from_exception_data(
            "DurableStructuredOutput",
            [{
                "type": PydanticCustomError(
                    "durable_output_schema", "{diagnostic}", {"diagnostic": self}
                ),
                "loc": (),
            }],
            hide_input=True,
        )

    def to_error(self, code: ErrorCode) -> AIError:
        details: dict[str, JsonValue] = {"path": self.path, "rule": self.rule}
        if self.expected_type is not None:
            details["expected_type"] = self.expected_type
            details["actual_type"] = self.actual_type
        return AIError(
            code,
            str(self),
            retryable=False,
            safe_details={"output_validation": details},
            diagnostics=ErrorDiagnostics.from_exception(self.to_validation_error()),
        )


def output_validation_error(error: BaseException, *, code: ErrorCode) -> "AIError | None":
    """Project durable schema failures without SDK input values or exception text."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValidationError):
            for detail in current.errors(include_input=False, include_url=False):
                diagnostic = detail.get("ctx", {}).get("diagnostic")
                if isinstance(diagnostic, _OutputSchemaDiagnostic):
                    return diagnostic.to_error(code)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return None


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "non-JSON value"


def _declared_property_names(schema: Mapping[str, JsonValue]) -> frozenset[str]:
    # Only schema-declared field names may appear in a diagnostic path; arbitrary
    # object keys (including pattern/additional properties) can contain secrets.
    names: set[str] = set()
    pending: list[object] = [schema]
    while pending:
        node = pending.pop()
        if not isinstance(node, Mapping):
            continue
        required = node.get("required")
        if isinstance(required, list):
            names.update(name for name in required if isinstance(name, str))
        for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
            children = node.get(keyword)
            if isinstance(children, Mapping):
                if keyword == "properties":
                    names.update(children)
                pending.extend(children.values())
        for keyword in (
            "items", "prefixItems", "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
            "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems", "propertyNames",
        ):
            child = node.get(keyword)
            pending.extend(child if isinstance(child, list) else [child])
    return frozenset(names)


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
    return canonicalize_output_schema_v1(value)


def canonicalize_output_schema_v1(
    value: object,
    output_type: "type[BaseModel] | None" = None,
) -> "dict[str, JsonValue]":
    """Canonicalize one output schema using the shared schema contract."""
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if output_type is not None:
        return canonicalize_pydantic_model_schema(output_type)
    return canonicalize_json_schema(cast(Mapping[str, JsonValue], value))


__all__ = [
    "AssistantTextOutput",
    "OutputBinding",
    "OutputMode",
    "bind_output",
    "output_validation_error",
    "restore_output",
]
