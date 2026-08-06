#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversation ownership and visibility values."""

from pydantic import BaseModel, ConfigDict


class Conversation(BaseModel):
    """Tenant-owned conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str
    tenant_id: str
    subject_id: str

    def can_access(self, tenant_id: str, subject_id: str) -> bool:
        """Return whether the principal can access this conversation."""
        return self.tenant_id == tenant_id and self.subject_id == subject_id
