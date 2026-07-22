#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AuthorizationService: check a PrincipalContext against an action + asset.

The contract is a single ``authorize()`` that returns ``None`` on success and
raises :class:`PrincipalAccessDeniedError` on denial. The first version ships
two deliberately simple implementations:

* :class:`AllowOwnerAuthorization` -- allow when the principal's tenant owns
  the asset; deny (fail-closed) when the asset has no tenant.
* :class:`DenyAllAuthorization` -- deny every request; the safe default for a
  Runtime constructed without an explicit authorization service.

A complex RBAC DSL is explicitly out of scope: ship the Protocol and
the two primitives, let downstream callers compose policy on top.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ...errors import PrincipalAccessDeniedError
from ...identity.principal import PrincipalContext


@dataclass(frozen=True, slots=True)
class AuthorizationTarget:
    """The target of a sensitive action. ``tenant_id`` is the asset's owner
    tenant (the only field AllowOwnerAuthorization inspects); ``kind`` and
    ``id`` carry the asset type + identifier for audit and for any
    finer-grained service layered above."""

    kind: str
    tenant_id: "str | None" = None
    id: "str | None" = None


@runtime_checkable
class AuthorizationService(Protocol):
    async def authorize(
        self,
        principal: PrincipalContext,
        action: str,
        asset: AuthorizationTarget,
    ) -> None:
        """Raise PrincipalAccessDeniedError when the principal may not perform
        ``action`` on ``asset``; return None otherwise."""
        ...


class AllowOwnerAuthorization:
    """Allow when the principal's tenant owns the asset.

    Assets without a tenant are denied (fail-closed): without a
    asset tenant to compare against, ownership cannot be confirmed, so the
    operation is rejected rather than allowed on a guessable id."""

    async def authorize(
        self,
        principal: PrincipalContext,
        action: str,
        asset: AuthorizationTarget,
    ) -> None:
        principal.require_tenant(asset.tenant_id)


class SameTenantAuthorization(AllowOwnerAuthorization):
    """Tenant-isolation policy; it intentionally grants no scope semantics."""


class ScopeAuthorization:
    """Default scoped policy: tenant match plus an explicit action scope."""

    async def authorize(self, principal, action, asset) -> None:
        principal.require_tenant(asset.tenant_id)
        if not principal.scopes.unrestricted and not principal.scopes.contains(action):
            raise PrincipalAccessDeniedError(
                f"principal lacks required scope: {action}"
            )


class DenyAllAuthorization:
    """Deny every request. The safe default for a Runtime constructed without
    an explicit authorization service, so the absence of policy can never be
    read as permission."""

    async def authorize(
        self,
        principal: PrincipalContext,
        action: str,
        asset: AuthorizationTarget,
    ) -> None:
        raise PrincipalAccessDeniedError(
            f"DenyAllAuthorization denies {action!r} on {asset.kind}"
        )
