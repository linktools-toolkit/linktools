#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output binding helpers."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from pydantic import BaseModel, ConfigDict

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
    value_type: "type[BaseModel]"
    schema_id: str
    schema_revision: int
    schema_fingerprint: str
    _schema_payload: bytes = field(repr=False, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.value_type, type) or not issubclass(self.value_type, BaseModel):
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

    @classmethod
    def create(
        cls,
        schema_id: str,
        value_type: "type[BaseModel]",
        *,
        schema_revision: int = 1,
    ) -> "OutputBinding":
        if not isinstance(schema_id, str) or not schema_id.strip():
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if not isinstance(value_type, type) or not issubclass(value_type, BaseModel):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if (
            not isinstance(schema_revision, int)
            or isinstance(schema_revision, bool)
            or schema_revision < 1
        ):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        try:
            schema = value_type.model_json_schema()
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if not isinstance(schema, dict):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return cls(
            value_type,
            schema_id,
            schema_revision,
            canonical_sha256(schema),
            canonical_json_bytes(schema),
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
    """Bind an output type to a stable durable schema identity."""
    if output is None:
        if schema_id is not None:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return OutputBinding.create(
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            AssistantTextOutput,
            schema_revision=ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
        )
    if not isinstance(output, type) or not issubclass(output, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if output is AssistantTextOutput:
        if schema_id not in (None, ASSISTANT_TEXT_OUTPUT_SCHEMA_ID):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if schema_revision != ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return OutputBinding.create(
            ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
            AssistantTextOutput,
            schema_revision=ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
        )
    if schema_id is None:
        try:
            schema = output.model_json_schema()
        except Exception as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if not isinstance(schema, dict):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        schema_fingerprint = canonical_sha256(schema)
        return OutputBinding(
            output,
            f"schema:{schema_fingerprint}",
            schema_revision,
            schema_fingerprint,
            canonical_json_bytes(schema),
        )
    if not isinstance(schema_id, str) or not schema_id.strip():
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if schema_id == ASSISTANT_TEXT_OUTPUT_SCHEMA_ID:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    return OutputBinding.create(
        schema_id,
        output,
        schema_revision=schema_revision,
    )


def restore_output(
    descriptor: Mapping[str, JsonValue],
    *,
    outputs: "Mapping[tuple[str, int], OutputBinding] | None" = None,
) -> OutputBinding:
    required = {"version", "schema_id", "schema_revision", "schema_fingerprint"}
    if set(descriptor) != required or descriptor.get("version") != 1:
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
    selected_outputs = outputs
    if selected_outputs is None:
        builtin = bind_output()
        selected_outputs = {
            (builtin.schema_id, builtin.schema_revision): builtin,
        }
    current = selected_outputs.get((schema_id, revision))
    if current is None or current.schema_fingerprint != schema_fingerprint:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return current


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
