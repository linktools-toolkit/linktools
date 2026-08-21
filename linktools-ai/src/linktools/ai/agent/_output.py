#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output binding helpers."""

import importlib
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
        if not isinstance(self.schema_revision, int) or isinstance(self.schema_revision, bool) or self.schema_revision < 1:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        _validate_fingerprint(self.schema_fingerprint)
        try:
            schema = json.loads(self._schema_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
        if not isinstance(schema, dict) or canonical_sha256(schema) != self.schema_fingerprint:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)

    @property
    def schema_definition(self) -> "dict[str, JsonValue]":
        value = json.loads(self._schema_payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        return cast("dict[str, JsonValue]", value)

    @property
    def fingerprint(self) -> str:
        return self.schema_fingerprint

    @property
    def descriptor(self) -> "dict[str, JsonValue]":
        return {
            "version": 1,
            "schema_id": self.schema_id,
            "schema_revision": self.schema_revision,
            "schema_fingerprint": self.schema_fingerprint,
            "module": self.value_type.__module__,
            "qualname": self.value_type.__qualname__,
        }


def bind_output(output: "type[BaseModel] | None" = None) -> OutputBinding:
    """Freeze one importable Python output model into an effective Agent definition."""
    selected = AssistantTextOutput if output is None else output
    if not isinstance(selected, type) or not issubclass(selected, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    _validate_importable_type(selected)
    try:
        schema = selected.model_json_schema()
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if not isinstance(schema, dict):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    if selected is AssistantTextOutput:
        schema_id = ASSISTANT_TEXT_OUTPUT_SCHEMA_ID
        schema_revision = ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION
    else:
        schema_id = f"{selected.__module__}.{selected.__qualname__}"
        schema_revision = 1
    return OutputBinding(
        selected,
        schema_id,
        schema_revision,
        canonical_sha256(schema),
        canonical_json_bytes(schema),
    )


def restore_output(descriptor: Mapping[str, JsonValue]) -> OutputBinding:
    fields = {"version", "schema_id", "schema_revision", "schema_fingerprint", "module", "qualname"}
    if set(descriptor) != fields or descriptor.get("version") != 1:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    module_name = descriptor.get("module")
    qualname = descriptor.get("qualname")
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    try:
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (AttributeError, ImportError, ModuleNotFoundError) as error:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
    if not isinstance(target, type) or not issubclass(target, BaseModel):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    try:
        restored = bind_output(target)
    except AIError as error:
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE) from error
    if restored.descriptor != dict(descriptor):
        raise AIError(ErrorCode.AGENT_DEFINITION_UNAVAILABLE)
    return restored


def _validate_importable_type(value: type[BaseModel]) -> None:
    module_name = value.__module__
    qualname = value.__qualname__
    if module_name == "__main__" or "<locals>" in qualname:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        target: object = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (AttributeError, ImportError, ModuleNotFoundError) as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    if target is not value:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)


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
