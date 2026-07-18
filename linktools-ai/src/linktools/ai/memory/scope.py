#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryScope: the tenant-bound access scope required by every
``MemoryStore`` search (and by ``MemoryManager.recall`` / retrieval adapters).
``search`` is the tenant isolation boundary: there is no unscoped /
cross-tenant query path, and a missing tenant fails closed (no results, never
the whole table).

``get`` / ``update`` / ``forget`` are NOT scope-gated -- they key off
``memory_id`` alone (the id is the capability). They are admin/lifecycle paths,
not the search/isolation boundary; ``search`` is what enforces the tenant
boundary.

``tenant_id`` is required and is the hard isolation boundary (tenant-a/alice
and tenant-b/alice must not see each other's memories even though they share
an owner_id). ``user_id`` / ``workspace_id`` / ``session_id`` are optional
sub-scopes; when set, search narrows to records whose corresponding field is
NULL (shared with the whole tenant) OR equal to the scope value. This lets a
tenant-scoped record be visible to every session of that tenant/user while a
session-private record stays session-scoped."""

from dataclasses import dataclass

#: Reserved tenant id for records persisted before tenant-scoping existed.
#: Such records are read back with this tenant id synthesized, so they are
#: NEVER matched by a real tenant's scope -- a caller must explicitly query
#: with this id to see them, which the default RunContext-driven policies never
#: do (they use ``context.tenant_id``). This is the migration quarantine: old
#: data is not silently exposed to any tenant; an admin must explicitly claim
#: it by re-writing it under a real tenant id.
LEGACY_TENANT_ID = "__legacy__"


def is_legacy_tenant(tenant_id: "object") -> bool:
    return tenant_id == LEGACY_TENANT_ID


@dataclass(frozen=True, slots=True)
class MemoryScope:
    tenant_id: str
    user_id: "str | None" = None
    workspace_id: "str | None" = None
    session_id: "str | None" = None
