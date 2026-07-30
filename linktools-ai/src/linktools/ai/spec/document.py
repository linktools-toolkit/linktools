"""Persisted specification documents and revision changes."""

import hashlib
from dataclasses import dataclass

from ..errors import SpecConflictError


@dataclass(frozen=True, slots=True)
class SpecDocumentInfo:
    path: str
    kind: str
    version: int
    etag: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class SpecDocument:
    info: SpecDocumentInfo
    content: bytes

    def validate_etag(self) -> None:
        """Assert ``info.etag`` equals ``sha256(content)``. A content change
        without an etag change would let other processes serve a stale cache,
        so every writer calls this before storing."""
        if self.info.etag != compute_spec_etag(self.content):
            raise SpecConflictError(
                f"spec etag mismatch for {self.info.path!r}: "
                f"etag must equal sha256(content)"
            )


def compute_spec_etag(content: bytes) -> str:
    """Canonical content etag for a spec document: SHA-256 hex digest."""
    return hashlib.sha256(content).hexdigest()


__all__ = ["SpecDocument", "SpecDocumentInfo", "compute_spec_etag"]
