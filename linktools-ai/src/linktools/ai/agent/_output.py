#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent output binding helpers."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..core import canonical_sha256
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
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.value_type, type) or not issubclass(self.value_type, BaseModel):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        if len(self.fingerprint) != 64 or any(character not in "0123456789abcdef" for character in self.fingerprint):
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)


def bind_output(output: "type[BaseModel] | None" = None) -> OutputBinding:
    """Freeze the selected Python output model into the effective Agent definition."""
    selected = AssistantTextOutput if output is None else output
    if not isinstance(selected, type) or not issubclass(selected, BaseModel):
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    try:
        schema = selected.model_json_schema()
    except Exception as error:
        raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID) from error
    return OutputBinding(selected, canonical_sha256(schema))


__all__ = [
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_ID",
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION",
    "AssistantTextOutput",
    "OutputBinding",
    "bind_output",
]
