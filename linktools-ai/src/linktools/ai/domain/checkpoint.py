#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Opaque workspace checkpoint metadata."""

from pydantic import BaseModel, ConfigDict


class WorkspaceCheckpoint(BaseModel):
    """Validated checkpoint reference, not workspace bytes."""

    model_config = ConfigDict(frozen=True)

    checkpoint_id: str
    workspace_digest: str
    manifest_ref: str
    file_count: int
    total_size: int

    def verify_manifest(self, digest: str) -> bool:
        """Return whether the supplied manifest matches."""
        return digest == self.workspace_digest
