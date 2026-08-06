#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Blob ledger state transitions."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class BlobState(StrEnum):
    """Blob metadata lifecycle."""

    STAGING = "STAGING"
    COMMITTED = "COMMITTED"
    TOMBSTONED = "TOMBSTONED"
    DELETING = "DELETING"
    DELETED = "DELETED"


class BlobObject(BaseModel):
    """Content-addressed blob metadata."""

    model_config = ConfigDict(frozen=True)

    blob_id: str
    tenant_id: str
    digest: str
    object_key: str
    size: int
    state: BlobState = BlobState.STAGING
    generation: int = 1
    delete_lease_id: "str | None" = None
    encryption_key_id: "str | None" = None
    reference_count: int = 0

    def can_reference(self) -> bool:
        """Return whether a new reference may be added."""
        return self.state == BlobState.COMMITTED

    def can_delete(self, lease_id: str) -> bool:
        """Return whether the supplied lease may finish deletion."""
        return self.state == BlobState.DELETING and self.delete_lease_id == lease_id


class BlobReference(BaseModel):
    """Tenant-scoped owner reference to one Blob generation."""

    model_config = ConfigDict(frozen=True)

    reference_id: str
    blob_id: str
    tenant_id: str
    owner_id: str
    generation: int
    tombstoned: bool = False

    def is_active(self) -> bool:
        """Return whether this reference can keep a Blob alive."""
        return not self.tombstoned
