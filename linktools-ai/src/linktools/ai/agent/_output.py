#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output bindings backed by one durable JSON-schema contract."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, cast

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


_TEXT_OUTPUT_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


OutputMode = Literal["text", "structured"]
_LITERAL_JSON_KEYWORDS = frozenset({"const", "default", "enum", "examples"})
_SCHEMA_MAP_KEYWORDS = frozenset({"properties", "patternProperties", "dependentSchemas"})


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

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract": "output-v1",
                "mode": self.mode,
                "schema": self.schema_definition,
            }
        )

    def validate_payload(self, value: JsonValue) -> None:
        """Validate one final JSON payload against this durable output contract."""
        try:
            _schema_validator(self.schema_definition).validate(value)
        except JsonSchemaValidationError as error:
            raise AIError(
                ErrorCode.OUTPUT_VALIDATION_FAILED, retryable=False
            ) from error

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
        schema = canonicalize_output_schema_v1(output.model_json_schema(), output)
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
    return canonicalize_output_schema_v1(value)


def canonicalize_output_schema_v1(
    value: object,
    output_type: "type[BaseModel] | None" = None,
) -> "dict[str, JsonValue]":
    """Canonicalize only stable generated-name noise in a v1 output schema."""
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
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error

    definitions = schema.get("$defs", {})
    if definitions is not None and not isinstance(definitions, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    refs = _local_definition_refs(schema)
    if output_type is not None:
        title = schema.get("title")
        config = getattr(output_type, "model_config", {})
        explicit_title = isinstance(config, Mapping) and config.get("title") is not None
        if not explicit_title and title == output_type.__name__:
            schema.pop("title", None)

    reachable: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(definition_key: str) -> None:
        if definition_key in visiting:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if definition_key in visited:
            return
        if not isinstance(definitions, dict) or definition_key not in definitions:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        visiting.add(definition_key)
        reachable.append(definition_key)
        for child in refs.get(definition_key, ()):
            visit(child)
        visiting.remove(definition_key)
        visited.add(definition_key)

    for definition_key in refs.get("", ()):
        visit(definition_key)

    rewritten = _canonical_schema_node(schema, reachable, root=True)
    if reachable:
        rewritten["$defs"] = {
            f"d{index}": _canonical_schema_node(
                definitions[key],
                reachable,
                original_key=key,
            )
            for index, key in enumerate(reachable)
        }
    else:
        rewritten.pop("$defs", None)
    return cast("dict[str, JsonValue]", rewritten)


def _local_definition_refs(value: object) -> "dict[str, tuple[str, ...]]":
    references: dict[str, list[str]] = {"": []}

    def collect(node: object, owner: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key == "$defs":
                    continue
                if key in _LITERAL_JSON_KEYWORDS or key == "dependentRequired":
                    continue
                if key in _SCHEMA_MAP_KEYWORDS:
                    if not isinstance(child, Mapping):
                        continue
                    for nested_schema in child.values():
                        collect(nested_schema, owner)
                    continue
                if key == "$dynamicRef":
                    raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
                if key == "$ref":
                    if not isinstance(child, str) or not child.startswith("#/$defs/"):
                        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
                    target = _decode_definition_pointer(child)
                    references.setdefault(owner, []).append(target)
                    continue
                collect(child, owner)
        elif isinstance(node, list):
            for child in node:
                collect(child, owner)

    collect(value, "")
    definitions = value.get("$defs") if isinstance(value, Mapping) else None
    if isinstance(definitions, Mapping):
        for key, definition in definitions.items():
            if not isinstance(key, str):
                raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
            references.setdefault(key, [])
            collect(definition, key)
    return {key: tuple(values) for key, values in references.items()}


def _decode_definition_pointer(value: str) -> str:
    key = value.removeprefix("#/$defs/")
    if not key:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return key.replace("~1", "/").replace("~0", "~")


def _canonical_schema_node(
    value: object,
    reachable: "Sequence[str]",
    *,
    root: bool = False,
    original_key: "str | None" = None,
) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if key == "$defs":
                continue
            if key in _LITERAL_JSON_KEYWORDS:
                result[str(key)] = child
                continue
            if key == "$ref" and isinstance(child, str):
                target = _decode_definition_pointer(child)
                if target not in reachable:
                    raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
                child = f"#/$defs/d{reachable.index(target)}"
            if key == "title" and not root and original_key is not None and child == original_key:
                continue
            result[str(key)] = _canonical_schema_node(child, reachable)
        for key in ("required", "type"):
            current = result.get(key)
            if isinstance(current, list):
                result[key] = sorted(set(current), key=lambda item: str(item))
        dependent = result.get("dependentRequired")
        if isinstance(dependent, dict):
            result["dependentRequired"] = {
                name: sorted(set(items), key=lambda item: str(item))
                if isinstance(items, list)
                else items
                for name, items in dependent.items()
            }
        return result
    if isinstance(value, list):
        return [_canonical_schema_node(child, reachable) for child in value]
    return value


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
