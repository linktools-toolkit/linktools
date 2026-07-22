#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public API contract: ``linktools.ai`` exports only ``Runtime`` at the root
(public API). Every other type lives behind its domain submodule; the root
imports (AgentSpec, Storage, SwarmSpec, ModelPolicy, ...) must FAIL so a stale
caller is caught at import time rather than silently binding to a moved symbol."""

import linktools.ai as ai


def test_root_exports_only_runtime():
    assert ai.Runtime is not None
    assert ai.__all__ == ["Runtime"]


def test_old_root_exports_are_gone():
    # These used to be re-exported at the root; the simplified API moved them
    # behind their domain submodules, so they must no longer be reachable here.
    for gone in (
        "AgentSpec",
        "PromptSpec",
        "ToolRef",
        "MiddlewareRef",
        "ModelPolicy",
        "ModelResolver",
        "RuntimeModelConfig",
        "Storage",
        "FilesystemStorage",
        "SwarmSpec",
    ):
        assert not hasattr(ai, gone), f"linktools.ai should no longer export {gone}"


def test_domain_imports_succeed():
    # The types that left the root are reachable via their domain submodules.
    from linktools.ai.agent import AgentSpec
    from linktools.ai.capability import CapabilityRuntimeOptions
    from linktools.ai.model.policy import ModelPolicy
    from linktools.ai.model.router import ModelResolver
    from linktools.ai.storage import FilesystemStorage, Storage
    from linktools.ai.swarm import SwarmSpec

    assert AgentSpec and CapabilityRuntimeOptions and ModelPolicy and ModelResolver
    assert Storage and FilesystemStorage and SwarmSpec


def test_root_does_not_re_export_sqlalchemy_storage():
    # SqlAlchemyStorage is NOT a root export (optional dependency).
    assert not hasattr(ai, "SqlAlchemyStorage")


def test_registry_error_hierarchy():
    from linktools.ai.errors import (
        LinktoolsAIError,
        RegistryError,
        RegistryNotFoundError,
        RegistryConflictError,
        RegistryParseError,
        InvalidSpecError,
    )

    assert issubclass(RegistryError, LinktoolsAIError)
    assert issubclass(RegistryNotFoundError, RegistryError)
    assert issubclass(RegistryConflictError, RegistryError)
    assert issubclass(RegistryParseError, RegistryError)
    assert issubclass(InvalidSpecError, RegistryError)
