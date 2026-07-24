#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ArtifactDigest: the validated SHA-256 value object for the artifact domain.

A digest is simultaneously the content address, the per-digest coordination key,
a filesystem path component (blob shard name, lock-file name) and a stored
record field. It MUST be exactly 64 lowercase hex characters before it reaches
any of those surfaces. Constructing/parseing at the boundary (``ArtifactDigest``
/ ``ArtifactDigest.parse`` / ``ArtifactDigest.from_bytes``) guarantees that no
uppercase, whitespace, path separator, Unicode, or overflow input can become a
coordinator key, a lock path, or a blob address.

The class does not ``lower()`` or ``strip()``: a value that is not already
canonical is rejected outright, so a caller can never smuggle a variant past a
defense that normalized it away. The error never echoes the rejected value.
"""

import hashlib
import re
from dataclasses import dataclass

from ..errors import InvalidArtifactDigestError

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ArtifactDigest:
    """An exact 64-lowercase-hex SHA-256 digest.

    Construction validates the format (``__post_init__``); the two classmethods
    are the boundary entry points -- ``parse`` for a caller-supplied string,
    ``from_bytes`` for content the domain has read and hashed itself. Comparisons
    and hashing are by ``value`` (the frozen dataclass gives that for free)."""

    value: str

    def __post_init__(self) -> None:
        # fullmatch (anchored on both ends): no surrounding characters, no
        # uppercase, no whitespace, no separators. A rejected value is never
        # echoed in the message.
        if not _SHA256_PATTERN.fullmatch(self.value):
            raise InvalidArtifactDigestError(
                "artifact digest must be exactly 64 lowercase hexadecimal characters"
            )

    @classmethod
    def parse(cls, value: str) -> "ArtifactDigest":
        return cls(value=value)

    @classmethod
    def from_bytes(cls, content: bytes) -> "ArtifactDigest":
        return cls(hashlib.sha256(content).hexdigest())

    def __str__(self) -> str:
        return self.value


__all__: "list[str]" = ["ArtifactDigest"]
