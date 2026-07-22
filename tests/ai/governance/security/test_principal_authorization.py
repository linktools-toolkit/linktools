#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PrincipalContext + AuthorizationService: of the
.

Covers the identity/authorization MODEL layer only (this is 's
commit-1 split -- wiring ``principal`` into Runtime.cancel / resume lands with
the cancellation-semantics phase). Asserts:

* PrincipalContext reuses the existing jobs.models ActorRef / ScopeSet (single
  definition, no duplicate security.ActorRef);
* tenant_id is required and fail-closed (require_tenant rejects a asset
  with no tenant, and rejects a tenant mismatch);
* resolved_by is derived from the trusted actor (never caller-supplied);
* AllowOwnerAuthorization allows same-tenant, denies cross-tenant and
  tenant-less assets; DenyAllAuthorization always denies;
* the principal module stays lazily loaded -- importing linktools.ai.governance.security
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
from linktools.ai.governance.security.authorization import (
    AllowOwnerAuthorization,
    AuthorizationTarget,
    AuthorizationService,
    DenyAllAuthorization,
)
from linktools.ai.identity.principal import PrincipalContext
from linktools.ai.jobs.models import ActorRef, ScopeSet


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
    # : single definition -- PrincipalContext.actor / scopes ARE the
    # jobs.models value types, not a security-local duplicate.
    p = _principal()
    assert isinstance(p.actor, ActorRef)
    assert isinstance(p.scopes, ScopeSet)


def test_principal_context_requires_explicit_scope_set():
    with pytest.raises(TypeError):
        PrincipalContext(tenant_id="t1", user_id="alice",
            actor=ActorRef(kind="user", id="alice"), scopes=None)
    with pytest.raises(TypeError):
        PrincipalContext(tenant_id="t1", user_id="alice",
            actor=ActorRef(kind="user", id="alice"), scopes=("read", "write"))


# --- require_tenant (fail-closed) -------------------------------------------


def test_require_tenant_same_tenant_passes():
    _principal("t1").require_tenant("t1")  # no raise


def test_require_tenant_cross_tenant_denied():
    with pytest.raises(PrincipalAccessDeniedError):
        _principal("t1").require_tenant("t2")


def test_require_tenant_resource_without_tenant_denied():
    # fail-closed: cannot confirm ownership without a asset tenant.
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
    asset = AuthorizationTarget(kind="run", tenant_id="t1", id="run-1")
    asyncio.run(auth.authorize(_principal("t1"), "cancel", asset))  # no raise


def test_allow_owner_denies_cross_tenant():
    auth = AllowOwnerAuthorization()
    asset = AuthorizationTarget(kind="run", tenant_id="t2", id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", asset))


def test_allow_owner_denies_resource_without_tenant():
    auth = AllowOwnerAuthorization()
    asset = AuthorizationTarget(kind="run", tenant_id=None, id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", asset))


def test_deny_all_denies_every_request():
    auth = DenyAllAuthorization()
    asset = AuthorizationTarget(kind="run", tenant_id="t1", id="run-1")
    with pytest.raises(PrincipalAccessDeniedError):
        asyncio.run(auth.authorize(_principal("t1"), "cancel", asset))


# --- Error hierarchy --------------------------------------------------------


def test_principal_access_denied_is_security_and_linktools_error():
    assert issubclass(SecurityError, LinktoolsAIError)
    assert issubclass(PrincipalAccessDeniedError, SecurityError)
    assert issubclass(PrincipalAccessDeniedError, LinktoolsAIError)


# --- Lazy-loading invariant -------------------------------------------------


def test_importing_security_does_not_load_jobs():
    # security must not depend on the task domain. (moved the shared
    # identity types out to ``identity`` precisely so security no longer pulls
    # jobs in.) Importing the security package at root must leave jobs.models
    # unloaded.
    code = (
        "import sys\n"
        "import linktools.ai.governance.security\n"
        "assert 'linktools.ai.jobs.models' not in sys.modules, "
        "'linktools.ai.jobs.models'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
