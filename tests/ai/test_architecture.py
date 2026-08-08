#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cold-start architecture and import checks. MT-51 MT-52."""

import ast
import importlib
import json
import subprocess
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


def test_name_gate_rejects_non_frozen_duplicate_and_package_collision(tmp_path: Path) -> None:
    root = tmp_path / "names"
    for package in ("app", "task", "runtime"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "data.py").write_text("", encoding="utf-8")
    (root / "task" / "data.py").write_text("", encoding="utf-8")
    (root / "app" / "runtime.py").write_text("", encoding="utf-8")
    errors = check_names(root)
    assert any(error.startswith("duplicate module basename: data") for error in errors)
    assert any(error.startswith("module/package stem collision:") for error in errors)


def test_name_gate_allows_only_the_existing_asset_storage_duplicate_group(tmp_path: Path) -> None:
    root = tmp_path / "names"
    for package in ("asset", "storage"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
        for name in ("cache.py", "files.py", "model.py"):
            (root / package / name).write_text("VALUE = 1\n", encoding="utf-8")
    (root / "model").mkdir()
    (root / "model" / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    assert check_names(root) == ()
    (root / "adapter").mkdir()
    (root / "adapter" / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert any(error.startswith("duplicate module basename: cache") for error in check_names(root))


def test_name_gate_rejects_non_grandfathered_duplicate_and_package_collision(tmp_path: Path) -> None:
    root = tmp_path / "names"
    for package in ("asset", "storage", "adapter", "task"):
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    (root / "asset" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "storage" / "new.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "asset" / "task.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "asset" / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "storage" / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "adapter" / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    errors = check_names(root)
    assert any(error.startswith("duplicate module basename: new") for error in errors)
    assert any(error.startswith("module/package stem collision:") and "asset/task.py" in error for error in errors)
    assert any(error.startswith("duplicate module basename: cache") for error in errors)


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


def test_facade_launcher_boundary_is_class_scoped() -> None:
    path = Path("linktools-ai/src/linktools/ai/app/workbench.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    assert {"WorkspaceExecutionLauncher", "WorkspaceAgentRuntime"} <= classes.keys()
    terminal_owners = {
        class_name
        for class_name, node in classes.items()
        if any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "commit_terminal" for call in ast.walk(node))
    }
    assert terminal_owners == {"WorkspaceExecutionLauncher"}
    runtime = classes["WorkspaceAgentRuntime"]
    assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "commit_terminal" for call in ast.walk(runtime))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "domain"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_stores"
        for node in ast.walk(runtime)
    )


def test_runtime_step_contract_matrix_is_current() -> None:
    root = Path("linktools-ai/scripts/build/matrix")
    requirements_path = root / "runtime-step-requirements.json"
    requirements = json.loads(requirements_path.read_text(encoding="utf-8"))
    entries = requirements["requirements"]
    assert [entry["id"] for entry in entries] == [f"DOD-{index:03d}" for index in range(1, 84)]
    for entry in entries:
        for evidence in entry.get("evidence", []):
            path_text, separator, test_name = evidence.partition("::")
            path = Path(path_text)
            assert path.is_file(), evidence
            if separator:
                tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
                assert any(node.name == test_name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), evidence
    by_id = {entry["id"]: entry for entry in entries}
    assert "LOCAL_CODING" not in by_id["DOD-026"]["requirement"]
    assert "profile" not in by_id["DOD-043"]["requirement"]
    assert "LOCAL/PRODUCTION" not in by_id["DOD-044"]["requirement"]
    assert "LOCAL_CODING" not in by_id["DOD-059"]["requirement"]
    assert "app/facade.py 与 app/facade.py" not in by_id["DOD-036"]["requirement"]
    dod_072_evidence = tuple(by_id["DOD-072"]["evidence"])
    assert "tests/ai/test_architecture.py::test_facade_launcher_boundary_is_class_scoped" in dod_072_evidence
    assert "linktools-ai/src/linktools/ai/app/workbench.py" in dod_072_evidence
    assert not any("test_file_step_store.py" in item or "test_harness_contract.py" in item or "adapter/step.py" in item for item in dod_072_evidence)

    matrix = json.loads((root / "requirement-matrix.json").read_text(encoding="utf-8"))
    matrix_entries = {entry["id"]: entry for entry in matrix["requirements"]}
    for index in range(23, 30):
        entry = matrix_entries[f"T-{index}"]
        assert entry["status"] in {"PENDING", "PASS"}
        assert entry["finding_mapping"]
        for test in entry["tests"]:
            path_text, separator, test_name = test.partition("::")
            assert separator
            path = Path(path_text)
            assert path.is_file(), test
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            assert any(node.name == test_name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), test

    evidence = json.loads((root / "linktools-ai-evidence.json").read_text(encoding="utf-8"))
    source_commit = evidence.get("validated_source_commit")
    if isinstance(source_commit, str) and len(source_commit) == 40:
        source = subprocess.run(
            ["git", "show", f"{source_commit}:linktools-ai/scripts/build/matrix/runtime-step-requirements.json"],
            capture_output=True,
            check=False,
            text=True,
        )
        if source.returncode == 0:
            source_entries = {entry["id"]: entry for entry in json.loads(source.stdout)["requirements"]}
            assert source_entries["DOD-023"]["status"] != "complete"
            assert source_entries["DOD-024"]["status"] != "complete"
