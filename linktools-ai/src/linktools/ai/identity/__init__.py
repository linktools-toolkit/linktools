#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Identity domain: the trusted-principal value types.

Owns ``PrincipalContext``, ``ActorRef`` and ``ScopeSet``. This package depends
only on the core error types -- never on jobs/task, run, agent or a storage
backend. It is the canonical home for these types; other domains import from
here.
"""

from .principal import (
    ActorRef,
    PrincipalContext,
    ScopeSet,
    trusted_local_principal,
)

__all__: "list[str]" = [
    "ActorRef",
    "PrincipalContext",
    "ScopeSet",
    "trusted_local_principal",
]
