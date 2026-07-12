#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON-Schema validation for tool arguments and schemas.

``jsonschema`` is a core dependency (see requirements.yml). A missing
installation is a broken environment, not a reason to skip validation -- the
default validator raises on construction so the failure surfaces at first use
rather than silently letting unvalidated arguments reach a tool handler.

The validator translates the underlying jsonschema exceptions into stable
error types (``ToolSchemaValidationError`` / ``ToolSchemaDefinitionError``) so
downstream never sees a bare ``jsonschema.ValidationError`` /
``SchemaError`` / ``ImportError``."""

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from ..errors import (
    RuntimeInitializationError,
    ToolSchemaDefinitionError,
    ToolSchemaValidationError,
)

if TYPE_CHECKING:
    pass


@runtime_checkable
class ToolSchemaValidator(Protocol):
    def validate_schema(self, schema: "Mapping[str, Any]") -> None: ...
    def validate_arguments(
        self,
        schema: "Mapping[str, Any]",
        arguments: "Mapping[str, Any]",
        *,
        tool_name: str = "",
    ) -> None: ...


class JsonSchemaToolValidator:
    """Validates tool parameters_json_schema (definition) and arguments against
    a schema. Fails closed: a missing ``jsonschema`` install raises on
    construction (it is a core dep), never silently skips validation."""

    def __init__(self) -> None:
        try:
            import jsonschema  # noqa: F401
        except ImportError as exc:  # pragma: no cover - core dep, env broken
            raise RuntimeInitializationError(
                "jsonschema is a core dependency but is not installed; the "
                "tool-schema validation gate cannot run -- refusing to fail open"
            ) from exc

    def validate_schema(self, schema: "Mapping[str, Any]") -> None:
        import jsonschema

        try:
            jsonschema.Draft7Validator.check_schema(_thaw(schema))
        except jsonschema.SchemaError as exc:
            raise ToolSchemaDefinitionError(
                f"malformed parameters_json_schema: {exc.message}"
            ) from exc

    def validate_arguments(
        self,
        schema: "Mapping[str, Any]",
        arguments: "Mapping[str, Any]",
        *,
        tool_name: str = "",
    ) -> None:
        import jsonschema

        try:
            jsonschema.validate(dict(arguments), _thaw(schema))
        except jsonschema.ValidationError as exc:
            raise ToolSchemaValidationError(
                f"tool {tool_name!r} arguments failed schema validation: {exc.message}"
            ) from exc


_DEFAULT: "JsonSchemaToolValidator | None" = None


def _thaw(value: Any) -> Any:
    """Convert immutable public snapshots back to JSON-native containers."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_thaw(item) for item in value]
    return value


def _default_validator() -> JsonSchemaToolValidator:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = JsonSchemaToolValidator()
    return _DEFAULT


def validate_arguments(
    arguments: "Mapping[str, Any]",
    parameters_json_schema: "Mapping[str, Any] | None",
    *,
    tool_name: str = "",
) -> None:
    """Validate ``arguments`` against ``parameters_json_schema``. No-op when no
    schema is available (the tool/adapter couldn't supply one). Raises
    ToolSchemaValidationError on a mismatch."""
    if not parameters_json_schema:
        return
    _default_validator().validate_arguments(
        parameters_json_schema, arguments, tool_name=tool_name
    )


def validate_schema(parameters_json_schema: "Mapping[str, Any] | None") -> None:
    """Validate that ``parameters_json_schema`` is itself a well-formed JSON
    schema (assembly-time check). No-op when None. Raises
    ToolSchemaDefinitionError on a malformed schema."""
    if not parameters_json_schema:
        return
    _default_validator().validate_schema(parameters_json_schema)
