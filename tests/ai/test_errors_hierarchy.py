#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable error hierarchy: callers must identify failures by type, not string.
Covers the capability/skill/mcp/package/subagent/model trees and the storage
capability sub-errors added in the capability-runtime refactor."""

import pytest

from linktools.ai import errors as E
from linktools.ai.errors import LinktoolsAIError


@pytest.mark.parametrize("exc_cls", [
    E.CapabilityResolutionError, E.CapabilityNotFoundError, E.CapabilityConflictError,
    E.SkillNotFoundError, E.MCPServerNotFoundError, E.MCPConnectionError, E.MCPToolError,
    E.PackageNotFoundError, E.PackageResourceNotFoundError, E.PackageResourceAccessDeniedError,
    E.PackageEntrypointNotFoundError, E.PackageEntrypointDeniedError,
    E.SubagentNotFoundError, E.SubagentDepthExceededError, E.SubagentExecutionError,
    E.ModelOutputValidationError, E.ModelTurnLimitExceededError,
    E.StorageTransactionNotSupportedError, E.StorageConcurrencyNotSupportedError,
    E.StorageLeaseNotSupportedError,
])
def test_all_errors_are_linktools_ai_errors(exc_cls):
    assert issubclass(exc_cls, LinktoolsAIError)


def test_capability_tree():
    assert issubclass(E.CapabilityNotFoundError, E.CapabilityResolutionError)
    assert issubclass(E.CapabilityConflictError, E.CapabilityResolutionError)
    for leaf in (E.SkillNotFoundError, E.MCPServerNotFoundError, E.PackageNotFoundError,
                 E.PackageResourceNotFoundError, E.PackageEntrypointNotFoundError,
                 E.SubagentNotFoundError):
        assert issubclass(leaf, E.CapabilityNotFoundError)


def test_policy_backed_errors():
    # Denied / depth-exceeded are policy decisions, not resolution misses.
    for leaf in (E.PackageResourceAccessDeniedError, E.PackageEntrypointDeniedError,
                 E.SubagentDepthExceededError):
        assert issubclass(leaf, E.PolicyError)
    assert issubclass(E.PolicyError, LinktoolsAIError)


def test_storage_capability_tree():
    base = E.StorageCapabilityError
    for leaf in (E.StorageTransactionNotSupportedError,
                 E.StorageConcurrencyNotSupportedError,
                 E.StorageLeaseNotSupportedError):
        assert issubclass(leaf, base)
    assert issubclass(base, E.StorageError)


def test_model_errors():
    assert issubclass(E.ModelTurnLimitExceededError, E.ModelPolicyExceededError)
    assert issubclass(E.ModelOutputValidationError, LinktoolsAIError)


def test_subagent_depth_carries_bounds():
    exc = E.SubagentDepthExceededError("too deep", depth=4, max_depth=3)
    assert exc.depth == 4 and exc.max_depth == 3


def test_subagent_execution_carries_error():
    exc = E.SubagentExecutionError("child failed", error={"status": "failed"})
    assert exc.error == {"status": "failed"}
