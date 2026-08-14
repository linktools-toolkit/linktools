#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cold-start architecture and import checks. MT-51 MT-52."""

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scripts.build.architecture import ArchitecturePolicyChecker, build_report
from scripts.build.cohesion import check_files
from scripts.build.names import check_names


def test_source_graph_is_acyclic_and_static() -> None:
    report = build_report("linktools-ai/src/linktools/ai")
    assert report["scc"] == []
    assert report["package_scc"] == []
    assert report["dynamic_imports"] == []
    assert ArchitecturePolicyChecker().check("linktools-ai/src/linktools/ai").passed


def test_names_and_module_imports_are_clean() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    assert check_names(root) == ()
    assert check_files(root) == ()
    for path in sorted(root.rglob("*.py")):
        name = "linktools.ai." + ".".join(path.relative_to(root).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        importlib.import_module(name)


def test_name_gate_allows_parallel_names_and_rejects_namespace_collisions(tmp_path: Path) -> None:
    root = tmp_path / "parallel"
    for package in ("aaa", "ccc"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (root / "ccc" / "bbb.py").write_text("", encoding="utf-8")
    assert check_names(root) == ()

    (root / "bbb.py").write_text("", encoding="utf-8")
    errors = check_names(root)
    assert sum(error.startswith("namespace semantic-name collision:\n") for error in errors) == 2


def test_name_gate_includes_modules_and_packages_in_namespace_model(tmp_path: Path) -> None:
    module_root = tmp_path / "module"
    (module_root / "aaa").mkdir(parents=True)
    (module_root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (module_root / "bbb.py").write_text("", encoding="utf-8")
    errors = check_names(module_root)
    assert len(errors) == 1
    assert errors[0].startswith("namespace semantic-name collision:\n")

    package_root = tmp_path / "package"
    (package_root / "aaa").mkdir(parents=True)
    (package_root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (package_root / "bbb").mkdir()
    (package_root / "bbb" / "__init__.py").write_text("", encoding="utf-8")
    errors = check_names(package_root)
    assert len(errors) == 1
    assert errors[0].startswith("namespace semantic-name collision:\n")

    same_parent_root = tmp_path / "same-parent"
    (same_parent_root / "aaa").mkdir(parents=True)
    (same_parent_root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (same_parent_root / "aaa" / "bbb").mkdir()
    (same_parent_root / "aaa" / "bbb" / "__init__.py").write_text("", encoding="utf-8")
    errors = check_names(same_parent_root)
    assert len(errors) == 1
    assert errors[0].startswith("namespace semantic-name collision:\n")

    nested_root = tmp_path / "nested"
    (nested_root / "aaa" / "ccc").mkdir(parents=True)
    (nested_root / "aaa" / "_bbb.py").write_text("", encoding="utf-8")
    (nested_root / "aaa" / "ccc" / "_bbb.py").write_text("", encoding="utf-8")
    errors = check_names(nested_root)
    assert len(errors) == 1
    assert errors[0].startswith("namespace semantic-name collision:\n")


def test_name_gate_treats_private_marker_as_semantic_only(tmp_path: Path) -> None:
    root = tmp_path / "names"
    for package in ("app", "runtime"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "_data.py").write_text("", encoding="utf-8")
    (root / "runtime" / "data.py").write_text("", encoding="utf-8")
    assert check_names(root) == ()

    (root / "_internal").mkdir()
    (root / "_internal" / "__init__.py").write_text("", encoding="utf-8")
    assert str(root / "_internal") in check_names(root)


def test_architecture_gate_normalizes_relative_module_policy_and_rejects_stale_entries(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    source_root = package_root / "src" / "linktools" / "ai"
    (source_root / "core").mkdir(parents=True)
    (source_root / "app").mkdir(parents=True)
    (source_root / "task").mkdir(parents=True)
    (source_root / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "core" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "app" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "task" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "app" / "facade.py").write_text("from ..core import Value\n\nVALUE = Value\n", encoding="utf-8")
    policy_path = package_root / "scripts" / "build" / "matrix" / "linktools-ai-package-policy.json"
    policy_path.parent.mkdir(parents=True)
    policy = {
        "top_level_packages": ["core", "app", "task"],
        "dependencies": {"core": [], "app": ["core"], "task": []},
        "module_dependencies": {"app.facade": ["core"]},
        "public_modules": ["app.facade"],
    }
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    checker = ArchitecturePolicyChecker()
    assert checker.check(source_root).passed
    (source_root / "app" / "facade.py").write_text("from ..task import Value\n\nVALUE = Value\n", encoding="utf-8")
    forbidden_edge = checker.check(source_root)
    assert not forbidden_edge.passed
    assert any("dependency policy: app -> task" in error for error in forbidden_edge.errors)
    (source_root / "app" / "facade.py").write_text("from ..core import Value\n\nVALUE = Value\n", encoding="utf-8")
    policy["module_dependencies"]["app.missing"] = []
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    result = checker.check(source_root)
    assert not result.passed
    assert any(error.startswith("stale module dependency policy:") for error in result.errors)


def test_public_private_classification_gate_is_exact(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    source_root = package_root / "src" / "linktools" / "ai"
    (source_root / "app").mkdir(parents=True)
    (source_root / "app" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "app" / "facade.py").write_text("VALUE = 1\n", encoding="utf-8")
    policy_path = package_root / "scripts" / "build" / "matrix" / "linktools-ai-package-policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps({"top_level_packages": ["app"], "public_modules": ["app.facade"]}), encoding="utf-8")
    checker = ArchitecturePolicyChecker()
    assert checker.check(source_root).passed
    (source_root / "app" / "extra.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = checker.check(source_root)
    assert any(error == "unclassified public module: app.extra" for error in result.errors)
    (source_root / "app" / "_extra.py").write_text("VALUE = 1\n", encoding="utf-8")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["public_modules"].append("app._extra")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    result = checker.check(source_root)
    assert any(error == "private module listed as public: app._extra" for error in result.errors)
    policy["public_modules"].append("app.missing")
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    assert any(error == "stale public module policy entry: app.missing" for error in checker.check(source_root).errors)


def test_private_cross_package_import_gate_covers_runtime_type_checking_and_nested_packages(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    source_root = package_root / "src" / "linktools" / "ai"
    for directory in (
        source_root / "app",
        source_root / "adapter",
        source_root / "temporal" / "workflow",
    ):
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (source_root / "adapter" / "_persistence.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "temporal" / "workflow" / "_execution.py").write_text("VALUE = 1\n", encoding="utf-8")
    worker = source_root / "temporal" / "_worker.py"
    worker.write_text("from .workflow._execution import VALUE\n", encoding="utf-8")
    app = source_root / "app" / "foo.py"
    app.write_text(
        "from typing import TYPE_CHECKING\n"
        "from ..adapter._persistence import VALUE\n"
        "from ..adapter import _persistence as PERSISTENCE_MODULE\n"
        "if TYPE_CHECKING:\n"
        "    from ..adapter._persistence import VALUE as TYPE_VALUE\n",
        encoding="utf-8",
    )
    policy_path = package_root / "scripts" / "build" / "matrix" / "linktools-ai-package-policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        json.dumps(
            {
                "top_level_packages": ["app", "adapter", "temporal"],
                "dependencies": {"app": ["adapter"], "adapter": [], "temporal": []},
                "public_modules": ["app.foo"],
            }
        ),
        encoding="utf-8",
    )
    result = ArchitecturePolicyChecker().check(source_root)
    assert sum("private cross-package import:" in error for error in result.errors) == 3
    app.write_text("from ..adapter import build_in_memory_runtime\n", encoding="utf-8")
    worker.write_text("from .workflow import ExecutionWorkflow\n", encoding="utf-8")
    assert not any("private cross-package import:" in error for error in ArchitecturePolicyChecker().check(source_root).errors)
    (source_root / "temporal" / "workflow" / "__init__.py").write_text("from ._execution import ExecutionWorkflow\n", encoding="utf-8")
    assert not any("private cross-package import:" in error for error in ArchitecturePolicyChecker().check(source_root).errors)


def test_private_conversion_tree_is_exact() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    expected_private_modules = (
        "agent/_compiler.py", "agent/_executor.py", "asset/_repository.py", "core/_value.py",
        "runtime/_execution.py", "storage/_cache.py", "task/_service_impl.py", "temporal/_gateway.py",
    )
    assert all((root / path).is_file() for path in expected_private_modules)
    assert not (root / "core" / "_allowlist.py").exists()
    assert (root / "errors.py").is_file()
    assert not (root / "core" / "errors.py").is_file()
    assert check_names(root) == ()
    policy = json.loads(Path("linktools-ai/scripts/build/matrix/linktools-ai-package-policy.json").read_text(encoding="utf-8"))
    assert policy["public_modules"] == ["errors", "acp"]


def test_package_public_surface_and_optional_dependency_isolation() -> None:
    from linktools.ai import adapter, asset, capability, model, runtime, workspace

    command_modules = tuple(
        importlib.import_module(f"linktools.commands.ai.{name}")
        for name in ("acp", "doctor", "run", "smoke")
    )
    assert adapter and asset and capability and model and runtime and workspace
    assert all(command_modules)
    assert adapter.__all__ == [
        "NatsPublisher",
        "ProviderClient",
        "PydanticMCPRuntime",
        "RuntimeMemoryStore",
        "StaticPrincipalProvider",
        "StepExecutionHistoryReader",
    ]
    assert model.__all__ == ["ModelBinding", "ModelRegistry", "ModelResolver"]
    assert workspace.__all__ == [
        "DisabledSandbox",
        "Sandbox",
        "Workspace",
        "WorkspacePolicy",
        "open_workspace_runtime",
        "trusted_workspace_principal",
    ]
    assert not any("private cross-package import:" in error for error in ArchitecturePolicyChecker().check("linktools-ai/src/linktools/ai").errors)
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "linktools-ai/src"), str(source_root / "linktools/src")))
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
for TARGET in TARGETS:
    sys.meta_path.insert(0, Blocker())
for name in ('linktools.ai.adapter', 'linktools.ai.asset', 'linktools.ai.temporal'):
    importlib.import_module(name)
for name in TARGETS:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", blocker], env=environment, check=True)


def test_facade_launcher_boundary_is_class_scoped() -> None:
    path = Path("linktools-ai/src/linktools/ai/workspace/_factory.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "open_workspace_runtime" in functions


def test_runtime_step_contract_matrix_is_current() -> None:
    root = Path("linktools-ai/scripts/build/matrix")
    requirements_path = root / "runtime-step-requirements.json"
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    entries = requirements["requirements"]
    assert [entry["id"] for entry in entries] == [f"DOD-{index:03d}" for index in range(1, 90)]
    assert all(entry.get("evidence") for entry in entries)
    by_id = {entry["id"]: entry for entry in entries}
    assert "LOCAL_CODING" not in by_id["DOD-026"]["requirement"]
    assert "profile" not in by_id["DOD-043"]["requirement"]
    assert "LOCAL/PRODUCTION" not in by_id["DOD-044"]["requirement"]
    assert "LOCAL_CODING" not in by_id["DOD-059"]["requirement"]
    assert by_id["DOD-036"]["requirement"]

    matrix = json.loads((root / "requirement-matrix.json").read_text(encoding="utf-8"))
    matrix_entries = {entry["id"]: entry for entry in matrix["requirements"]}
    for index in range(23, 39):
        entry = matrix_entries[f"T-{index}"]
        assert entry["status"] in {"PENDING", "PASS"}
        assert entry["finding_mapping"]
        for test in entry["tests"]:
            path_text, separator, test_name = test.partition("::")
            assert separator
            path = Path(path_text)
            assert path.is_file(), test

    evidence = json.loads((root / "linktools-ai-evidence.json").read_text(encoding="utf-8"))
    source_commit = evidence.get("validated_source_commit")
    if isinstance(source_commit, str) and len(source_commit) == 40:
        assert subprocess.run(["git", "cat-file", "-e", f"{source_commit}^{{commit}}"], check=False).returncode == 0
