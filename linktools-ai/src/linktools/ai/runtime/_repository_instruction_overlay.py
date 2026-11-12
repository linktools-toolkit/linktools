#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint overlay representation independent of accepted instruction identity."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..core import ImmutableJsonMapping, JsonValue, PersistenceDomain, canonical_sha256
from ..errors import AIError, ErrorCode
from ..execution import RuntimePayloadRef
from ..spec import RepositoryInstructions
from ..storage import StoredPayload


@dataclass(frozen=True, slots=True)
class RepositoryInstructionOverlay:
    semantic_digest: str
    instructions: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if (not isinstance(self.semantic_digest, str) or len(self.semantic_digest) != 64
                or any(character not in "0123456789abcdef" for character in self.semantic_digest)
                or not isinstance(self.instructions, Mapping)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            object.__setattr__(self, "instructions", ImmutableJsonMapping(self.instructions))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    @classmethod
    def capture(
        cls, accepted: RepositoryInstructions, stored: Mapping[str, JsonValue],
    ) -> "RepositoryInstructionOverlay":
        return cls(canonical_sha256(accepted.to_payload()), stored)

    def verify(self, accepted: RepositoryInstructions) -> None:
        if canonical_sha256(accepted.to_payload()) != self.semantic_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def to_reference(self) -> RuntimePayloadRef:
        return RuntimePayloadRef(StoredPayload.inline_json({
            "kind": "repository-instruction-overlay", "version": 1,
            "semantic_digest": self.semantic_digest, "instructions": dict(self.instructions),
        }), PersistenceDomain.CHECKPOINT)

    @classmethod
    def from_reference(cls, reference: RuntimePayloadRef) -> "RepositoryInstructionOverlay":
        if (not isinstance(reference, RuntimePayloadRef)
                or reference.source_domain is not PersistenceDomain.CHECKPOINT
                or reference.payload.kind != "inline" or reference.payload.encoding != "json"):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        value = reference.payload.decode()
        if (not isinstance(value, Mapping)
                or set(value) != {"kind", "version", "semantic_digest", "instructions"}
                or value["kind"] != "repository-instruction-overlay"
                or type(value["version"]) is not int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value["version"] != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return cls(value["semantic_digest"], value["instructions"])


__all__ = ["RepositoryInstructionOverlay"]
