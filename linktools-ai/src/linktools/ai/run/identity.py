#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ParentRunIdentity: the single, stable identity object passed to a
SubagentExecutor/EntrypointExecutor when spawning a child run. Replaces
scattered ``parent_run_id`` / ``root_run_id`` / ``parent_session_id`` /
``user_id`` / ``tenant_id`` / ``workspace`` parameters threaded individually
through every call site -- every spawner (subagent toolset, package
entrypoint toolset) builds one of these from its CapabilityContext and hands
it to the executor unchanged, so lineage/identity propagation is defined in
exactly one place."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ParentRunIdentity:
    run_id: str
    root_run_id: str
    session_id: str
    user_id: "str | None" = None
    tenant_id: "str | None" = None
    workspace: Any = None
