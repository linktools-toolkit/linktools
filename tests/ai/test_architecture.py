#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cold-start architecture and import checks. MT-51 MT-52."""

import importlib
import json
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
    assert check_names(root, "linktools-ai/scripts/build/name-exceptions.json") == ()
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
        (root / package / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert check_names(root) == ()
    (root / "adapter").mkdir()
    (root / "adapter" / "cache.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert any(error.startswith("duplicate module basename: cache") for error in check_names(root))


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
