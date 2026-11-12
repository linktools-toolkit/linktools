#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted instruction payloads separate semantic identity from stored bytes."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..core import ImmutableJsonMapping, JsonValue, PersistenceDomain, canonical_sha256
from ..errors import AIError, ErrorCode
from ..execution import RuntimePayloadRef
from ..spec import RepositoryInstructions
from ..storage import StoredPayload


@dataclass(frozen=True, slots=True)
class RepositoryInstructionPayload:
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
    ) -> "RepositoryInstructionPayload":
        return cls(canonical_sha256(accepted.to_payload()), stored)

    def verify(self, accepted: RepositoryInstructions) -> None:
        if canonical_sha256(accepted.to_payload()) != self.semantic_digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def to_reference(self, source_domain: PersistenceDomain) -> RuntimePayloadRef:
        kind = self._kind(source_domain)
        return RuntimePayloadRef(StoredPayload.inline_json({
            "kind": kind, "version": 1,
            "semantic_digest": self.semantic_digest, "instructions": dict(self.instructions),
        }), source_domain)

    @classmethod
    def from_reference(
        cls, reference: RuntimePayloadRef, *, source_domain: PersistenceDomain | None = None,
    ) -> "RepositoryInstructionPayload":
        if (not isinstance(reference, RuntimePayloadRef)
                or source_domain is not None and reference.source_domain is not source_domain
                or reference.payload.kind != "inline" or reference.payload.encoding != "json"):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = cls._kind(reference.source_domain)
        value = reference.payload.decode()
        if (not isinstance(value, Mapping)
                or set(value) != {"kind", "version", "semantic_digest", "instructions"}
                or value["kind"] != kind
                or type(value["version"]) is not int):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if value["version"] != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return cls(value["semantic_digest"], value["instructions"])

    @classmethod
    def _kind(cls, source_domain: PersistenceDomain) -> str:
        if source_domain is PersistenceDomain.EXECUTION:
            return "repository-instruction-admission"
        if source_domain is PersistenceDomain.CHECKPOINT:
            return "repository-instruction-overlay"
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


__all__ = ["RepositoryInstructionPayload"]
