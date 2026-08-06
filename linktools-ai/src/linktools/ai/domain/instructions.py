#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trusted and untrusted instruction parts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes


class InstructionTrust(StrEnum):
    """Trust labels preserved through prompt assembly."""

    TRUSTED = "trusted"
    UNTRUSTED_CONTEXT = "untrusted_context"


class InstructionPart(BaseModel):
    """One bounded instruction source."""

    model_config = ConfigDict(frozen=True)

    source: str
    trust: InstructionTrust
    digest: str
    content: str


class RunInstructionSet(BaseModel):
    """Versioned bounded instruction collection."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=1, ge=1)
    parts: "tuple[InstructionPart, ...]"
    digest: str

    def verify(self, expected_digest: str) -> bool:
        """Return whether the stored and expected aggregate digests match."""
        content = tuple(part.model_dump(mode="json") for part in self.parts)
        actual = sha256_digest(canonical_json_bytes(content))
        return self.digest == expected_digest == actual
