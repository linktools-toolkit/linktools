#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identity value types: the canonical home for ``PrincipalContext``,
``ActorRef`` and ``ScopeSet``.

These types are the trust boundary for every sensitive operation
(cancel / resume / approve / reject / memory + artifact access): a bare id
(``run_id``, ``session_id``, ...) is never authorization. A sensitive call
requires a ``PrincipalContext`` carrying ``tenant_id``, ``user_id``, the
acting ``ActorRef`` and the granted ``ScopeSet``.

This package depends on nothing downstream -- not jobs/task, run, agent or any
storage backend. The task and security domains import identity; identity never
imports them.
"""

from dataclasses import dataclass

from ..errors import PrincipalAccessDeniedError


@dataclass(frozen=True, slots=True)
class ActorRef:
    kind: str
    id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("ActorRef.kind must be non-empty")
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("ActorRef.id must be non-empty")


@dataclass(frozen=True, slots=True)
class ScopeSet:
    unrestricted: bool = False
    values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(sorted({
            item.strip() for item in self.values
            if isinstance(item, str) and item.strip()
        }))
        if self.unrestricted and normalized:
            raise ValueError("unrestricted ScopeSet cannot contain explicit scopes")
        object.__setattr__(self, "values", normalized)

    @classmethod
    def allow_all(cls) -> "ScopeSet":
        return cls(unrestricted=True)

    @classmethod
    def empty(cls) -> "ScopeSet":
        return cls()

    @classmethod
    def of(cls, *scopes: str) -> "ScopeSet":
        return cls(values=tuple(scopes))

    @classmethod
    def from_any(cls, value):
        if value is None:
            raise TypeError("ScopeSet cannot be constructed from None")
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            raise TypeError("ScopeSet requires an iterable, not a string")
        return cls(values=tuple(value))

    @property
    def is_empty(self) -> bool:
        return not self.unrestricted and not self.values

    def contains(self, scope: str) -> bool:
        return self.unrestricted or scope in self.values


@dataclass(frozen=True, slots=True)
class PrincipalContext:
    """The identity authorizing a sensitive operation.

    ``tenant_id`` is always required (the isolation boundary); ``user_id`` is
    the human account the actor acts for, if any; ``actor`` is the concrete
    caller (an agent, a service, a subagent); ``scopes`` is the resolved set
    of delegated scopes -- always a concrete ``ScopeSet``, never None.
    """

    tenant_id: str
    user_id: "str | None"
    actor: ActorRef
    scopes: ScopeSet

    def __post_init__(self) -> None:
        if not isinstance(self.tenant_id, str) or not self.tenant_id.strip():
            raise ValueError("PrincipalContext.tenant_id must be a non-empty string")
        if not isinstance(self.actor, ActorRef):
            raise TypeError("PrincipalContext.actor must be an ActorRef")
        if not isinstance(self.scopes, ScopeSet):
            raise TypeError("PrincipalContext.scopes must be an explicit ScopeSet")

    def require_tenant(self, tenant_id: "str | None") -> None:
        """Fail-closed tenant check. Raises if the target asset has
        no tenant to compare against, or if this principal's tenant does not
        own it."""
        if tenant_id is None:
            raise PrincipalAccessDeniedError(
                "target asset has no tenant; refusing to authorize against "
                f"principal tenant {self.tenant_id!r}"
            )
        if self.tenant_id != tenant_id:
            raise PrincipalAccessDeniedError(
                f"principal tenant {self.tenant_id!r} cannot access tenant "
                f"{tenant_id!r}"
            )

    @property
    def resolved_by(self) -> str:
        """The audit identity derived from the trusted Principal, never
        caller-supplied. Used as ``resolved_by`` on approve/reject and the
        ``cancel_requested_by`` / resume audit fields."""
        return f"{self.actor.kind}:{self.actor.id}"


def trusted_local_principal(*, tenant_id: str = "local") -> PrincipalContext:
    """Explicit principal for single-user/local deployments and tests."""
    return PrincipalContext(
        tenant_id=tenant_id,
        user_id=None,
        actor=ActorRef(kind="system", id="trusted-local"),
        scopes=ScopeSet.allow_all(),
    )


__all__: "list[str]" = [
    "ActorRef",
    "PrincipalContext",
    "ScopeSet",
    "trusted_local_principal",
]
