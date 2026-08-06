#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tenant principal values used to construct trusted storage context."""

from pydantic import BaseModel, ConfigDict


class TenantPrincipalRef(BaseModel):
    """Opaque authenticated tenant and subject reference."""

    model_config = ConfigDict(frozen=True)

    tenant_id: str
    subject_id: str


class ACPSession(BaseModel):
    """Local ACP session bound to a user, workspace, and Bundle digest."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    os_user: str
    workspace_root: str
    bundle_digest: str

    def can_load(self, os_user: str, workspace_root: str, bundle_digest: str) -> bool:
        """Return whether a reconnecting client matches the session scope."""
        return (
            self.os_user == os_user
            and self.workspace_root == workspace_root
            and self.bundle_digest == bundle_digest
        )


__all__ = ["ACPSession", "TenantPrincipalRef"]
