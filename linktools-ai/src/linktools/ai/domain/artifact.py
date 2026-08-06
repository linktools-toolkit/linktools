#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Artifact metadata and retention values."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ArtifactRetention(StrEnum):
    """Supported product retention classes."""

    STANDARD = "standard"
    PERMANENT = "permanent"


class Artifact(BaseModel):
    """Tenant-scoped immutable artifact metadata."""

    model_config = ConfigDict(frozen=True)

    artifact_id: str
    tenant_id: str
    execution_id: str
    digest: str
    content_type: str
    retention: ArtifactRetention

    def can_download(self, tenant_id: str) -> bool:
        """Return whether the tenant owns this artifact."""
        return tenant_id == self.tenant_id
