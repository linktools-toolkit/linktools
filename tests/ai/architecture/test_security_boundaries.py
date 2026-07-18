#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Security-architecture boundary freeze for linktools.ai.

Phase 0 of the production-hardening plan
(``.docs/linktools-ai-production-hardening-plan.md``) snapshots the CURRENT
security-relevant architecture BEFORE later phases change it. Every assertion
describes the code as it is on the branch base; when a later phase
legitimately changes one of these invariants, update the snapshot in the same
change so the change is intentional and visible rather than silent.

The general architecture invariants (root exports exactly ``Runtime``, the
rejected ``linktools.ai.durable`` namespace stays absent, importing the root
keeps the heavy extension domains out) are already frozen in
``test_task_boundaries.py`` and are not duplicated here. This file freezes the
security-specific surfaces only.

Snapshot invariants (a later phase WILL change these -- update here then):

* Approval identity and execution revisions are bound fields; service-level
  approval derives ``resolved_by`` from Principal.

Landed invariants:

* §7.1 / §7.2 -- ``PrincipalContext`` (reusing ``task.models`` ActorRef /
  ScopeSet) + ``AuthorizationService`` / AllowOwner / DenyAll;
* §7.3 -- ``Runtime.cancel`` / ``resume`` accept ``principal`` and reject a
  missing principal unless ``local_trusted_mode`` (default-strict).
* §12.3 / §12.4 -- ``MemoryRecord`` is tenant-scoped (tenant_id +
  user/workspace/session sub-scopes) and ``MemoryStore.search`` takes a
  required ``MemoryScope`` with no ``scope=None`` global-search path.
"""

import dataclasses
import importlib.util
import inspect
import subprocess
import sys


# --- Hard invariant: core dependency lightness --------------------------------


def test_linktools_core_does_not_require_redis() -> None:
    # §5.2: redis is never the source of truth; the core framework must not
    # acquire it as an import-time dependency.
    code = "import sys; import linktools.core; assert 'redis' not in sys.modules"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# --- Sensitive-operation signatures (§7.3 landed) ---------------------------


def test_runtime_cancel_accepts_principal() -> None:
    # §7.3 landed: cancel is keyword-gated by ``principal`` (+ ``reason``);
    # run_id is no longer sufficient authorization.
    from linktools.ai.runtime import Runtime

    params = inspect.signature(Runtime.cancel).parameters
    assert {"self", "run_id", "principal", "reason"} <= set(params), dict(params)


def test_runtime_resume_accepts_principal() -> None:
    # §7.3 landed: resume is keyword-gated by ``principal``.
    from linktools.ai.runtime import Runtime

    params = inspect.signature(Runtime.resume).parameters
    assert {"self", "run_id", "principal"} <= set(params), dict(params)


# --- Identity / authorization models (§7.1 / §7.2 landed) -------------------


def test_principal_context_is_defined_and_reuses_actor_types() -> None:
    # §7.1 landed: PrincipalContext exists in security.principal and REUSES the
    # canonical task.models ActorRef / ScopeSet -- no duplicate definition.
    assert importlib.util.find_spec("linktools.ai.security.principal") is not None
    from linktools.ai.security.principal import PrincipalContext
    from linktools.ai.task.models import ActorRef, ScopeSet

    field_types = {f.name: str(f.type) for f in dataclasses.fields(PrincipalContext)}
    assert {"tenant_id", "user_id", "actor", "scopes"} <= set(field_types)
    assert "ActorRef" in field_types["actor"]
    assert "ScopeSet" in field_types["scopes"]
    # The reused types are exactly the task.models value types.
    assert ActorRef.__module__ == "linktools.ai.security.principal"
    assert ScopeSet.__module__ == "linktools.ai.security.principal"


def test_authorization_service_is_defined() -> None:
    # §7.2 landed: AuthorizationService Protocol + AllowOwner / DenyAll impls.
    assert importlib.util.find_spec("linktools.ai.security.authorization") is not None
    from linktools.ai.security.authorization import (
        AllowOwnerAuthorization,
        AuthorizationService,
        DenyAllAuthorization,
    )

    assert AuthorizationService is not None
    assert AllowOwnerAuthorization is not None
    assert DenyAllAuthorization is not None


# --- Snapshot: approval shape (§11 will change) ------------------------------


def test_approval_request_redacts_arguments_snapshot() -> None:
    # §11.1 / §11.4 landed: ApprovalRequest no longer persists the raw call
    # arguments (they may carry secrets). It stores a redacted audit copy
    # (redacted_arguments) + an identity fingerprint (arguments_hash). The
    # handler still receives the real arguments in memory; this record is for
    # approval / audit only.
    from linktools.ai.agent.approval import ApprovalRequest

    fields = {f.name for f in dataclasses.fields(ApprovalRequest)}
    assert "redacted_arguments" in fields
    assert "arguments_hash" in fields
    assert "arguments" not in fields  # raw arguments no longer persisted
    for bound in (
        "tenant_id", "descriptor_fingerprint", "handler_revision",
        "provider_revision", "policy_revision", "capability_revision",
        "schema_version",
    ):
        assert bound in fields


# --- Snapshot: memory identity (§12 landed) ----------------------------------


def test_memory_record_is_tenant_scoped() -> None:
    # §12.3 landed: MemoryRecord now carries tenant_id (the hard isolation
    # boundary) plus optional user_id / workspace_id / session_id sub-scopes.
    # owner_id is retained as a display/compat field but is NOT an authorization
    # boundary.
    from linktools.ai.memory.models import MemoryRecord

    fields = {f.name for f in dataclasses.fields(MemoryRecord)}
    assert "tenant_id" in fields
    assert "owner_id" in fields  # display/compat only
    for sub in ("user_id", "workspace_id", "session_id"):
        assert sub in fields, sub


def test_memory_store_search_requires_scope() -> None:
    # §12.4 landed: search takes a required MemoryScope and has no owner_id
    # kwarg and no scope=None global-search default.
    from linktools.ai.memory.scope import MemoryScope
    from linktools.ai.memory.store import MemoryStore

    params = inspect.signature(MemoryStore.search).parameters
    assert "scope" in params
    assert "owner_id" not in params
    # scope is required (no default).
    assert params["scope"].default is inspect.Parameter.empty
    # And MemoryScope itself is the access-scope object.
    assert MemoryScope is not None
