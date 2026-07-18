#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitive-operation authorization gate for Runtime.cancel / resume.

A sensitive operation must not act on a bare run id: it requires a
``PrincipalContext``. ``principal=None`` is tolerated only in explicit
``local_trusted_mode`` (and then only with a DeprecationWarning, so the legacy
path is visible and eventually removed); by default it is rejected
(fail-closed). When a Principal is presented and the run has a tenant (read
from its RunDefinitionSnapshot), ownership is enforced -- cross-tenant cancel
/ resume is denied.

Lives in this module (not on Runtime directly) so the top-level runtime.py
source stays free of the DeprecationWarning token (a deliberate package
invariant). Runtime.cancel / resume delegate here.
"""

import warnings

from ..errors import PrincipalAccessDeniedError


async def authorize_sensitive_operation(
    *,
    storage,
    local_trusted_mode: bool,
    run_id: str,
    principal,
    action: str,
) -> None:
    """Enforce the Principal gate + tenant ownership for a sensitive op.

    ``principal`` is duck-typed (``PrincipalContext``): it exposes
    ``require_tenant(resource_tenant)``. Imported lazily by callers, so this
    module does not pull the task domain at import time.
    """
    if principal is None:
        warnings.warn(
            f"Runtime.{action}(...) without principal is deprecated; pass "
            "principal= or build Runtime(local_trusted_mode=True)",
            DeprecationWarning,
            stacklevel=2,
        )
        if not local_trusted_mode:
            raise PrincipalAccessDeniedError(
                f"Runtime.{action} requires a principal "
                "(or local_trusted_mode=True)"
            )
        return
    # Principal presented: enforce ownership when the run has a tenant. A run
    # with no tenant (local / unscoped) passes on the strength of the
    # presented Principal; cross-tenant denial applies only where a resource
    # tenant exists to compare against.
    definition = await storage.run_definitions.get(run_id)
    run_tenant = definition.tenant_id if definition is not None else None
    if run_tenant is not None:
        principal.require_tenant(run_tenant)
