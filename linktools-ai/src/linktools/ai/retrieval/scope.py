#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RetrievalScope: the tenant-bound access scope required by every knowledge
``Retriever.search``. Mirrors :class:`~linktools.ai.memory.scope.MemoryScope`
but for the knowledge/retrieval domain, and adds ``allowed_resource_ids``:
when non-empty, retrieval is additionally restricted to that explicit
allow-list (e.g. the document set the run was launched against).

``tenant_id`` is required and is the hard isolation boundary -- a missing
tenant fails closed (no retrieval, never a global search). Every backend must
apply the tenant filter; vector indexes must carry it as a pre-filter so a
neighbor query can never cross tenant boundaries."""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetrievalScope:
    tenant_id: str
    user_id: "str | None" = None
    workspace_id: "str | None" = None
    session_id: "str | None" = None
    allowed_resource_ids: "tuple[str, ...]" = field(default_factory=tuple)
