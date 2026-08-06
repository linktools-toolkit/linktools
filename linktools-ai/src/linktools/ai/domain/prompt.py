#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Frozen prompt snapshot metadata."""

from pydantic import BaseModel, ConfigDict

from ..foundation.digest import verify_digest


class RepoContextSnapshot(BaseModel):
    """Untrusted repository context captured by an allowed Activity."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    digest: str
    source_manifest_ref: str

    def verify_manifest(self, digest: str) -> bool:
        """Return whether the source manifest is unchanged."""
        return digest == self.digest


class SkillSnapshot(BaseModel):
    """Signed platform Skill snapshot."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    digest: str
    signature: str

    def verify_signature(self, signature: str) -> bool:
        """Return whether signature metadata matches."""
        return signature == self.signature


class PromptSnapshot(BaseModel):
    """Immutable prompt source and digest."""

    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    prompt_name: str
    digest: str
    content_ref: str

    def verify_digest(self, content: bytes) -> None:
        """Verify snapshot content."""
        verify_digest(content, self.digest)


__all__ = ["PromptSnapshot", "RepoContextSnapshot", "SkillSnapshot"]
