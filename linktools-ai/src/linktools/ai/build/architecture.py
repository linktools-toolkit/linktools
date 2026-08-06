#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""AST architecture graph and strongly connected component analysis."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from .cohesion import check_files
from .names import check_names


def module_name(path: Path, source_root: Path) -> str:
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts
    return "linktools.ai" if parts == ("__init__",) else "linktools.ai." + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _resolve(module: str, level: int, current: str, *, current_is_package: bool = False) -> str:
    if level == 0:
        return module
    package = current if current_is_package else current.rsplit(".", 1)[0]
    parts = package.split(".")[: len(package.split(".")) - (level - 1)]
    return ".".join((*parts, module)) if module else ".".join(parts)


def _scc(graph: "dict[str, set[str]]") -> "tuple[tuple[str, ...], ...]":
    index = 0
    stack: "list[str]" = []
    on_stack: "set[str]" = set()
    indexes: "dict[str, int]" = {}
    low: "dict[str, int]" = {}
    result: "list[tuple[str, ...]]" = []

    def visit(node: str) -> None:
        nonlocal index
        indexes[node] = low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, ()):
            if target not in indexes:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indexes[target])
        if low[node] == indexes[node]:
            component = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            if len(component) > 1 or component[0] in graph.get(component[0], set()):
                result.append(tuple(sorted(component)))

    for node in graph:
        if node not in indexes:
            visit(node)
    return tuple(sorted(result))


def build_report(source_root: "str | Path") -> "dict[str, object]":
    root = Path(source_root)
    modules = {module_name(path, root): path for path in root.rglob("*.py")}
    runtime: "dict[str, set[str]]" = {name: set() for name in modules}
    checking: "dict[str, set[str]]" = {name: set() for name in modules}
    dynamic: "list[str]" = []
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        checking_lines = {node.lineno for node in ast.walk(tree) if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING"}
        for node in ast.walk(tree):
            targets: "list[str]" = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                targets = [
                    _resolve(
                        node.module or "",
                        node.level,
                        name,
                        current_is_package=path.name == "__init__.py",
                    )
                ]
            for target in targets:
                resolved = target if target in modules else target.rsplit(".", 1)[0] if target.rsplit(".", 1)[0] in modules else None
                if resolved is not None:
                    target_graph = checking if any(line <= node.lineno for line in checking_lines) else runtime
                    target_graph[name].add(resolved)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"import_module", "__import__"}:
                dynamic.append(f"{path}:{node.lineno}")
    return {"modules": tuple(sorted(modules)), "runtime": {key: tuple(sorted(value)) for key, value in runtime.items()}, "type_checking": {key: tuple(sorted(value)) for key, value in checking.items()}, "scc": _scc(runtime), "dynamic_imports": tuple(sorted(dynamic))}


@dataclass(frozen=True, slots=True)
class ArchitectureCheckResult:
    passed: bool
    errors: "tuple[str, ...]"
    report: "dict[str, object]"


class ArchitecturePolicyChecker:
    """Run graph, naming and module ownership checks together."""

    def check(self, source_root: "str | Path") -> ArchitectureCheckResult:
        root = Path(source_root)
        report = build_report(root)
        errors = [
            *(f"runtime SCC: {component}" for component in report["scc"]),
            *(f"dynamic import: {item}" for item in report["dynamic_imports"]),
            *check_names(root),
            *check_files(root),
        ]
        policy_path = root.parent.parent.parent / "linktools-ai-package-dependency-policy.json"
        if policy_path.exists():
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            packages = policy.get("packages", {})
            for module, targets in report["runtime"].items():
                source_package = "" if module == "linktools.ai" else module.removeprefix("linktools.ai.").split(".", 1)[0]
                for target in targets:
                    target_package = "" if target == "linktools.ai" else target.removeprefix("linktools.ai.").split(".", 1)[0]
                    if source_package and source_package != target_package and target_package not in packages.get(source_package, ()):
                        errors.append(f"dependency policy: {module} -> {target}")
        forbidden = ("agent_runtime", "evaluation_runtime", "application/use_cases", "storage/testing", "storage/sql/sync")
        for marker in forbidden:
            if any(marker in path.as_posix() for path in root.rglob("*.py")):
                errors.append(f"forbidden module path: {marker}")
        return ArchitectureCheckResult(not errors, tuple(errors), report)


__all__ = ["ArchitectureCheckResult", "ArchitecturePolicyChecker", "build_report", "module_name"]
