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
from ..task.models import ActorRef, ScopeSet


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
        # Normalize legacy tuple/None at the boundary so a persisted
        # PrincipalContext is never None-typed -- mirrors
        # ActorChain.delegated_scopes normalization. None is treated as
        # unrestricted (ScopeSet.from_any) to preserve legacy "no scope limit".
        if not isinstance(self.scopes, ScopeSet):
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
