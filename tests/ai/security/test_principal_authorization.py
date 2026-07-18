#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PrincipalContext + AuthorizationService: §7.1 / §7.2 of the
production-hardening plan.

Covers the identity/authorization MODEL layer only (this is the plan's
commit-1 split -- wiring ``principal`` into Runtime.cancel / resume lands with
the cancellation-semantics phase). Asserts:

* PrincipalContext reuses the existing task.models ActorRef / ScopeSet (single
  definition, no duplicate security.ActorRef);
* tenant_id is required and fail-closed (require_tenant rejects a resource
  with no tenant, and rejects a tenant mismatch);
* resolved_by is derived from the trusted actor (never caller-supplied);
* AllowOwnerAuthorization allows same-tenant, denies cross-tenant and
  tenant-less resources; DenyAllAuthorization always denies;
* the principal module stays lazily loaded -- importing linktools.ai.security
  does not pull the task domain into the root import graph.
"""

import asyncio
import subprocess
import sys

import pytest

from linktools.ai.errors import (
    LinktoolsAIError,
    PrincipalAccessDeniedError,
    SecurityError,
)
from linktools.ai.security.authorization import (
    AllowOwnerAuthorization,
    AuthorizationResource,
    AuthorizationService,
    DenyAllAuthorization,
)
from linktools.ai.security.principal import PrincipalContext
from linktools.ai.task.models import ActorRef, ScopeSet


def _principal(
    tenant_id: str = "t1",
    *,
    user_id: "str | None" = "alice",
    actor: "ActorRef | None" = None,
    scopes: "ScopeSet | None" = None,
) -> PrincipalContext:
    return PrincipalContext(
        tenant_id=tenant_id,
        user_id=user_id,
        actor=actor or ActorRef(kind="user", id="alice"),
        scopes=scopes if scopes is not None else ScopeSet.allow_all(),
    )


# --- Construction + validation ----------------------------------------------


def test_principal_context_requires_nonempty_tenant_id():
    with pytest.raises(ValueError):
        _principal(tenant_id="")
    with pytest.raises(ValueError):
        _principal(tenant_id="   ")


def test_principal_context_actor_must_be_actor_ref():
    with pytest.raises(TypeError):
        PrincipalContext(
            tenant_id="t1",
            user_id=None,
            actor=("user", "alice"),  # not an ActorRef
            scopes=ScopeSet.allow_all(),
        )


def test_principal_context_reuses_canonical_actor_and_scope_types():
    # §7.1: single definition -- PrincipalContext.actor / scopes ARE the
    # task.models value types, not a security-local duplicate.
    p = _principal()
    assert isinstance(p.actor, ActorRef)
    assert isinstance(p.scopes, ScopeSet)


def test_principal_context_normalizes_legacy_scope_input():
    # None -> unrestricted (legacy "no scope limit"); tuple -> values.
    assert _principal(scopes=None).scopes == ScopeSet(unrestricted=True)
    assert _principal(scopes=("read", "write")).scopes == ScopeSet(
        values=("read", "write")
    )


# --- require_tenant (fail-closed) -------------------------------------------


def test_require_tenant_same_tenant_passes():
    _principal("t1").require_tenant("t1")  # no raise


def test_require_tenant_cross_tenant_denied():
    with pytest.raises(PrincipalAccessDeniedError):
        _principal("t1").require_tenant("t2")


def test_require_tenant_resource_without_tenant_denied():
    # §5.4 fail-closed: cannot confirm ownership without a resource tenant.
    with pytest.raises(PrincipalAccessDeniedError):
        _principal("t1").require_tenant(None)


# --- resolved_by derived from trusted actor ---------------------------------


def test_resolved_by_derived_from_actor():
    p = PrincipalContext(
        tenant_id="t1",
        user_id="alice",
        actor=ActorRef(kind="service", id="orchestrator"),
        scopes=ScopeSet.allow_all(),
    )
    assert p.resolved_by == "service:orchestrator"


# --- AuthorizationService implementations -----------------------------------


def test_authorization_service_is_a_protocol():
    # The two impls satisfy the AuthorizationService Protocol structurally.
    assert isinstance(AllowOwnerAuthorization(), AuthorizationService)
    assert isinstance(DenyAllAuthorization(), AuthorizationService)


def test_allow_owner_authorizes_same_tenant():
    auth = AllowOwnerAuthorization()
    resource = AuthorizationResource(kind="run", tenant_id="t1", id="run-1")
    asyncio.run(auth.authorize(_principal("t1"), "cancel", resource))  # no raise


def test_allow_owner_denies_cross_tenant():
    auth = AllowOwnerAuthorization()
    resource = AuthorizationResource(kind="run", tenant_id="t2", id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", resource))


def test_allow_owner_denies_resource_without_tenant():
    auth = AllowOwnerAuthorization()
    resource = AuthorizationResource(kind="run", tenant_id=None, id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", resource))


def test_deny_all_denies_every_request():
    auth = DenyAllAuthorization()
    resource = AuthorizationResource(kind="run", tenant_id="t1", id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", resource))


# --- Error hierarchy --------------------------------------------------------


def test_principal_access_denied_is_security_and_linktools_error():
    assert issubclass(SecurityError, LinktoolsAIError)
    assert issubclass(PrincipalAccessDeniedError, SecurityError)
    assert issubclass(PrincipalAccessDeniedError, LinktoolsAIError)


# --- Lazy-loading invariant -------------------------------------------------


def test_importing_security_does_not_load_principal_or_task():
    # principal.py imports task.models; it must stay lazily loaded so the
    # security package (loaded at root) does not drag the task domain in.
    code = (
        "import sys\n"
        "import linktools.ai.security\n"
        "for mod in ('linktools.ai.security.principal',"
        " 'linktools.ai.task.models'):\n"
        "    assert mod not in sys.modules, mod\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
