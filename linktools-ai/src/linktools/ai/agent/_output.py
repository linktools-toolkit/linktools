#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen output schema registry for executable Agent bindings."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode


class AssistantTextOutput(BaseModel):
    """Canonical structured output containing the assistant response text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str


ASSISTANT_TEXT_OUTPUT_SCHEMA_ID = "assistant-text"
ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION = 1


@dataclass(frozen=True, slots=True)
class OutputSchemaManifestEntry:
    schema_id: str
    revision: int
    value_type: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class OutputSchemaManifest:
    entries: "tuple[OutputSchemaManifestEntry, ...]"
    digest: str


class OutputTypeRegistry:
    def __init__(self) -> None:
        self._types: dict[tuple[str, int], type[BaseModel]] = {}
        self._fingerprints: dict[tuple[str, int], str] = {}
        self._manifest: OutputSchemaManifest | None = None

    def register(self, schema_id: str, revision: int, output_type: "type[BaseModel]") -> None:
        if self._manifest is not None:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_DRIFT, "output registry is frozen")
        if not schema_id.strip() or revision < 1:
            raise AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
        key = schema_id, revision
        fingerprint = canonical_sha256(output_type.model_json_schema())
        previous = self._fingerprints.get(key)
        previous_type = self._types.get(key)
        if previous is not None and (previous != fingerprint or previous_type is not output_type):
            raise AIError(ErrorCode.OUTPUT_SCHEMA_DRIFT)
        self._types[key] = output_type
        self._fingerprints[key] = fingerprint

    @property
    def frozen(self) -> bool:
        return self._manifest is not None

    def freeze(self) -> OutputSchemaManifest:
        entries = tuple(
            OutputSchemaManifestEntry(
                schema_id=key[0],
                revision=key[1],
                value_type=f"{value.__module__}.{value.__qualname__}",
                fingerprint=self._fingerprints[key],
            )
            for key, value in sorted(self._types.items())
        )
        digest = canonical_sha256(
            {
                "entries": [
                    {
                        "schema_id": entry.schema_id,
                        "revision": entry.revision,
                        "value_type": entry.value_type,
                        "fingerprint": entry.fingerprint,
                    }
                    for entry in entries
                ]
            }
        )
        self._manifest = OutputSchemaManifest(entries, digest)
        return self._manifest

    def resolve(self, schema_id: str, revision: int) -> "type[BaseModel]":
        try:
            return self._types[(schema_id, revision)]
        except KeyError as exc:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_UNKNOWN) from exc

    def fingerprint(self, schema_id: str, revision: int) -> str:
        try:
            return self._fingerprints[(schema_id, revision)]
        except KeyError as exc:
            raise AIError(ErrorCode.OUTPUT_SCHEMA_UNKNOWN) from exc

    def manifest(self) -> OutputSchemaManifest:
        if self._manifest is None:
            return self.freeze()
        return self._manifest


__all__ = [
    "ASSISTANT_TEXT_OUTPUT_SCHEMA_ID", "ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION", "AssistantTextOutput",
    "OutputSchemaManifest", "OutputSchemaManifestEntry", "OutputTypeRegistry",
]
