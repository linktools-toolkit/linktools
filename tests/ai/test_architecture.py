#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture checker behavior and final package ownership tests."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.check.ai.architecture import ArchitecturePolicyChecker
from scripts.check.ai.cohesion import check_files
from scripts.check.ai.names import check_names


def test_modules_are_importable() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    for path in sorted(root.rglob("*.py")):
        name = "linktools.ai." + ".".join(path.relative_to(root).with_suffix("").parts)
        name = name.removesuffix(".__init__")
        importlib.import_module(name)


def test_current_architecture_gates_are_clean() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    result = ArchitecturePolicyChecker().check(root)
    assert result.passed, "\n".join(result.errors)
    assert check_names(root) == ()
    assert check_files(root) == ()


def test_name_gate_rejects_nested_namespace_collision(tmp_path: Path) -> None:
    root = tmp_path / "names"
    (root / "aaa" / "ccc").mkdir(parents=True)
    (root / "aaa" / "__init__.py").write_text("", encoding="utf-8")
    (root / "aaa" / "ccc" / "__init__.py").write_text("", encoding="utf-8")
    (root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (root / "aaa" / "ccc" / "_bbb.py").write_text("", encoding="utf-8")
    errors = check_names(root)
    assert len(errors) == 1
    assert errors[0].startswith("namespace semantic-name collision:\n")


def test_removed_owners_are_gone_and_runtime_owners_exist() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    removed = (
        "agent/_executor.py",
        "agent/_builder.py",
        "agent/_capabilities.py",
        "agent/_input.py",
        "asset/_repository.py",
        "asset/_logical.py",
        "capability/_contract.py",
        "spec/_assets.py",
        "workspace/_factory.py",
        "workspace/_tools.py",
        "adapter/_history.py",
        "adapter/_mcp.py",
    )
    required = (
        "capability/_context.py",
        "capability/_group.py",
        "capability/_workspace.py",
        "runtime/_agent.py",
        "runtime/_agent_executor.py",
        "runtime/_capabilities.py",
        "runtime/_history.py",
        "runtime/_input.py",
        "runtime/_memory.py",
        "runtime/_plan.py",
        "runtime/_runtime_service.py",
    )
    assert all(not (root / path).exists() for path in removed)
    assert all((root / path).is_file() for path in required)


def test_public_composition_surface_is_final() -> None:
    from linktools import ai
    from linktools.ai import adapter, capability, workspace

    assert ai.__all__ == [
        "Agent",
        "CapabilityGroup",
        "Execution",
        "RunContext",
        "Runtime",
        "Session",
        "Workspace",
    ]
    assert adapter.__all__ == [
        "NatsPublisher",
        "ProviderClient",
        "StaticPrincipalProvider",
    ]
    assert capability.__all__ == [
        "CapabilityContribution",
        "CapabilityGroup",
        "CapabilityLoader",
        "capability_fingerprint",
        "contribution_semantic_contract",
        "RunContext",
        "SKILL_TOOL_NAMES",
        "SkillCapability",
        "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
        "WORKSPACE_FILESYSTEM_TOOL_NAMES",
        "WORKSPACE_SHELL_TOOL_NAMES",
        "materialize_mcp_servers",
        "mcp_selector_server",
        "mcp_server_namespace",
        "mcp_server_selector",
        "mcp_tool_name",
        "workspace_capabilities",
        "workspace_tool_class",
        "workspace_tool_contributions",
    ]
    assert workspace.__all__ == [
        "DisabledSandbox",
        "Sandbox",
        "Workspace",
        "WorkspacePolicy",
        "trusted_workspace_principal",
    ]


def test_package_policy_matches_final_owner_graph() -> None:
    policy = json.loads(
        Path("scripts/check/ai/matrix/linktools-ai-package-policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["dependencies"]["asset"] == ["core", "storage"]
    assert policy["dependencies"]["workspace"] == ["core"]
    assert set(policy["dependencies"]["capability"]) == {
        "core",
        "asset",
        "spec",
        "workspace",
    }
    assert set(policy["dependencies"]["agent"]) == {
        "core",
        "spec",
        "model",
        "capability",
    }
    assert set(policy["dependencies"]["runtime"]) == {
        "core",
        "storage",
        "asset",
        "spec",
        "model",
        "observe",
        "capability",
        "task",
        "agent",
        "workspace",
    }
    removed_modules = {
        "workspace._factory",
        "workspace._tools",
        "adapter._history",
        "adapter._mcp",
        "agent._executor",
        "agent._capabilities",
        "asset._repository",
        "capability._contract",
        "spec._assets",
    }
    assert removed_modules.isdisjoint(policy["module_dependencies"])
    assert policy["module_dependencies"]["capability._workspace"] == ["workspace"]
    assert policy["module_dependencies"]["runtime._agent_executor"] == [
        "core",
        "capability",
        "agent",
    ]
    assert policy["module_dependencies"]["runtime._capabilities"] == [
        "core",
        "capability",
    ]
    assert set(policy["module_dependencies"]["runtime._factory"]) == {
        "core",
        "storage",
        "asset",
        "spec",
        "model",
        "capability",
        "task",
        "agent",
        "workspace",
    }


def test_contract_map_has_no_removed_composition_contracts() -> None:
    contracts = json.loads(
        Path("scripts/check/ai/matrix/linktools-ai-contract-map.json").read_text(
            encoding="utf-8"
        )
    )["contracts"]
    assert "logical_asset" not in contracts
    assert "runtime_access" not in contracts
    assert contracts["raw_asset_storage"] == {
        "owner": "asset",
        "replacement": "linktools.ai.asset.AssetStore",
    }
    assert contracts["capability_composition"] == {
        "owner": "capability",
        "replacement": "linktools.ai.capability.CapabilityGroup",
    }
    assert contracts["runtime_composition"] == {
        "owner": "runtime",
        "replacement": "linktools.ai.runtime.Runtime.open",
    }
    assert contracts["agent_runtime"] == {
        "owner": "runtime",
        "replacement": "linktools.ai.runtime.Agent",
    }


def test_optional_dependency_isolation() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    blocker = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in TARGETS):
            raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        return None

TARGETS = ('sqlalchemy', 'temporalio', 'acp')
for target in TARGETS:
    sys.meta_path.insert(0, Blocker())
for name in ('linktools.ai.adapter', 'linktools.ai.asset', 'linktools.ai.temporal'):
    importlib.import_module(name)
for name in TARGETS:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", blocker], env=environment, check=True)
