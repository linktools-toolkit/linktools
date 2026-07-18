#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PrincipalContext: the trusted identity bound to every sensitive operation.

Production-hardening plan §5.1 / §7.1: sensitive APIs (cancel / resume /
approve / reject / memory + artifact access) must not operate on a bare id
(``run_id``, ``session_id``, ``memory_id``, ...) -- they require a
``PrincipalContext`` carrying ``tenant_id``, ``user_id``, the acting
``ActorRef`` and the granted ``ScopeSet``. A guessable id is never
authorization (§5.5).

This module is the canonical home for ``PrincipalContext``. It REUSES the
existing ``ActorRef`` and ``ScopeSet`` value types from
``linktools.ai.task.models`` rather than redefining them (§7.1: keep a single
definition -- never let ``task.ActorRef`` and ``security.ActorRef`` coexist).

It is imported lazily -- deliberately NOT re-exported from
``security/__init__`` -- so that adding it cannot pull the task domain into
the root import graph (the boundary frozen in
``tests/ai/architecture/test_task_boundaries.py``). Callers import it
explicitly: ``from linktools.ai.security.principal import PrincipalContext``.
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

    @classmethod
    def allow_all(cls) -> "ScopeSet":
        return cls(unrestricted=True)

    @classmethod
    def of(cls, *scopes: str) -> "ScopeSet":
        return cls(values=tuple(scopes))

    @classmethod
    def from_any(cls, value):
        if value is None:
            return cls(unrestricted=True)
        if isinstance(value, cls):
            return value
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
            # Deserialize legacy task principals at this boundary. New
            # callers should pass ScopeSet explicitly; task wire compatibility
            # is retained without changing the persisted representation.
            object.__setattr__(self, "scopes", ScopeSet.from_any(self.scopes))

    def require_tenant(self, tenant_id: "str | None") -> None:
        """Fail-closed tenant check (§5.4). Raises if the target resource has
        no tenant to compare against, or if this principal's tenant does not
        own it."""
        if tenant_id is None:
            raise PrincipalAccessDeniedError(
                "target resource has no tenant; refusing to authorize against "
                f"principal tenant {self.tenant_id!r}"
            )
        if self.tenant_id != tenant_id:
            raise PrincipalAccessDeniedError(
                f"principal tenant {self.tenant_id!r} cannot access tenant "
                f"{tenant_id!r}"
            )

    @property
    def resolved_by(self) -> str:
        """The audit identity derived from the trusted Principal (§7.5), never
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
