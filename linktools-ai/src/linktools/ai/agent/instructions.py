#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure instruction assembly for fixed runs."""

from ..domain.instructions import InstructionPart, InstructionTrust, RunInstructionSet
from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes


class InstructionAssembler:
    """Assemble trusted and untrusted parts without I/O."""

    def assemble(self, parts: "tuple[InstructionPart, ...]") -> RunInstructionSet:
        value = tuple(part.model_dump(mode="json") for part in parts)
        return RunInstructionSet(parts=parts, digest=sha256_digest(canonical_json_bytes(value)))


__all__ = ["InstructionAssembler"]
