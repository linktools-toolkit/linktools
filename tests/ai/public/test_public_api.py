#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final public API surface.

These are the only imports the simplified architecture promises. The test
exists so any silent public API change fails here instead of in downstream code.
"""


def test_top_level_runtime_import():
    from linktools.ai import Runtime

    assert Runtime is not None
    import linktools.ai as ai_pkg

    assert "Runtime" in ai_pkg.__all__


def test_removed_public_and_compatibility_surfaces_stay_absent(tmp_path):
    import importlib.util

    from linktools.ai.runtime import Runtime
    from linktools.ai.storage.facade import FilesystemStorage

    assert not hasattr(Runtime, "assemble")
    runtime = Runtime.build(storage=FilesystemStorage(root=tmp_path))
    assert not hasattr(runtime, "runner")
    assert not hasattr(runtime, "compiler")
    assert not hasattr(runtime, "capability_assembler")
    for module_name in (
        "linktools.ai.capability.resolver",
        "linktools.ai.tool.policy_adapter",
        "linktools.ai.tool.idempotency_key",
        "linktools.ai.tool.legacy",
    ):
        assert importlib.util.find_spec(module_name) is None

    import linktools.ai.capability as capability

    assert not hasattr(capability, "CapabilityResolver")


def test_agent_domain_imports():
    from linktools.ai.agent import AgentSpec

    assert AgentSpec is not None


def test_capability_domain_imports():
    from linktools.ai.capability import CapabilityInspection, CapabilityRuntimeOptions

    assert CapabilityInspection is not None
    assert CapabilityRuntimeOptions is not None


def test_providers_domain_imports():
    from linktools.ai.runtime import RuntimeDependencies

    assert RuntimeDependencies is not None


def test_security_domain_imports():
    from linktools.ai.governance.security import SecurityBaseline

    assert SecurityBaseline is not None


def test_storage_domain_imports():
    from linktools.ai.storage import Storage

    assert Storage is not None


def test_swarm_domain_imports():
    from linktools.ai.swarm import SwarmSpec

    assert SwarmSpec is not None


def test_tool_domain_imports():
    from linktools.ai.tool import (
        EffectiveToolPolicy,
        ManagedToolDefinition,
        ResolvedToolPolicy,
        ToolDescriptor,
    )

    assert EffectiveToolPolicy is not None
    assert ManagedToolDefinition is not None
    assert ResolvedToolPolicy is not None
    assert ToolDescriptor is not None


def test_storage_optional_dependency_is_lazy():
    """Importing linktools.ai.storage must not require SQLAlchemy. The lazy
    SqlAlchemyStorage accessor is exercised separately in the storage suite."""
    import linktools.ai.storage as storage_pkg

    assert hasattr(storage_pkg, "Storage")
    assert hasattr(storage_pkg, "FilesystemStorage")


def test_importing_root_does_not_pull_sqlalchemy():
    """``import linktools.ai`` must succeed without the optional SQLAlchemy
    extra installed (the dependency is loaded lazily only on access)."""
    import os
    import subprocess
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(root, "linktools", "src"),
            os.path.join(root, "linktools-ai", "src"),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import linktools.ai; assert 'sqlalchemy' not in sys.modules",
        ],
        check=True,
        env=env,
    )
