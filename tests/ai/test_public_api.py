#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import linktools.ai as ai


def test_public_api_exports_agent_spec():
    from linktools.ai.agent.spec import AgentSpec as _Real
    assert ai.AgentSpec is _Real


def test_public_api_exports_runtime():
    from linktools.ai.runtime import Runtime as _Real
    assert ai.Runtime is _Real


def test_public_api_exports_file_storage():
    from linktools.ai.storage.facade import FileStorage as _Real
    assert ai.FileStorage is _Real


def test_public_api_exports_storage():
    from linktools.ai.storage.facade import Storage as _Real
    assert ai.Storage is _Real


def test_public_api_exports_swarm_spec():
    from linktools.ai.swarm.spec import SwarmSpec as _Real
    assert ai.SwarmSpec is _Real


def test_public_api_exports_model_hot_types():
    # contract: ModelPolicy / ModelRouter / RuntimeModelConfig are root exports.
    from linktools.ai.model import ModelPolicy, ModelRouter, RuntimeModelConfig
    assert ai.ModelPolicy is ModelPolicy
    assert ai.ModelRouter is ModelRouter
    assert ai.RuntimeModelConfig is RuntimeModelConfig


def test_public_api_exports_prompt_and_refs():
    from linktools.ai.agent.spec import PromptSpec, ToolRef, MiddlewareRef
    assert ai.PromptSpec is PromptSpec
    assert ai.ToolRef is ToolRef
    assert ai.MiddlewareRef is MiddlewareRef


def test_root_does_not_re_export_sqlalchemy_storage():
    # contract: SqlAlchemyStorage is NOT a root export (optional dependency).
    assert not hasattr(ai, "SqlAlchemyStorage")


def test_sqlalchemy_storage_accessible_lazily_from_storage_package():
    # contract: lazy __getattr__ keeps the short import working on demand.
    from linktools.ai.storage import SqlAlchemyStorage
    from linktools.ai.storage.sqlalchemy.facade import SqlAlchemyStorage as _Real
    assert SqlAlchemyStorage is _Real


def test_public_api_does_not_re_export_internals():
    for internal in ("AgentCompiler", "AgentRunner", "CompiledAgent", "Middleware",
                     "MiddlewarePipeline", "PolicyEngine", "ToolExecutor", "RunStore",
                     "SessionStore", "EventStore", "ResourceStore",
                     "SwarmRunner", "SwarmStore",
                     "CoordinatorDelegationStrategy", "ParallelFanOutStrategy"):
        assert not hasattr(ai, internal), f"linktools.ai should not export {internal}"


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


def test_approval_error_hierarchy():
    from linktools.ai.errors import (
        LinktoolsAIError,
        ApprovalError,
        ApprovalNotFoundError,
        ApprovalConflictError,
        InvalidApprovalTransitionError,
    )
    assert issubclass(ApprovalError, LinktoolsAIError)
    assert issubclass(ApprovalNotFoundError, ApprovalError)
    assert issubclass(ApprovalConflictError, ApprovalError)
    assert issubclass(InvalidApprovalTransitionError, ApprovalError)


def test_tool_approval_required_error_still_tool_error():
    from linktools.ai.errors import ToolError, ToolApprovalRequiredError
    assert issubclass(ToolApprovalRequiredError, ToolError)
