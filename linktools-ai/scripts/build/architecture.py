#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static package, module, and ownership gates for the AI source tree."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from linktools.ai.core import JsonValue

from .cohesion import check_files
from .names import check_names


def module_name(path: Path, source_root: Path) -> str:
    """Return the import name represented by a source file."""
    relative = path.relative_to(source_root).with_suffix("")
    parts = relative.parts
    if parts == ("__init__",):
        return "linktools.ai"
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return "linktools.ai." + ".".join(parts)


def _resolve(module: str, level: int, current: str, *, current_is_package: bool) -> str:
    if level == 0:
        return module
    package = current if current_is_package else current.rsplit(".", 1)[0]
    parts = package.split(".")
    if level > len(parts):
        return module
    base = parts[: len(parts) - level + 1]
    return ".".join((*base, module)) if module else ".".join(base)


def _scc(graph: 'dict[str, set[str]]') -> 'tuple[tuple[str, ...], ...]':
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indexes: dict[str, int] = {}
    low: dict[str, int] = {}
    result: list[tuple[str, ...]] = []

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
        if low[node] != indexes[node]:
            return
        component: list[str] = []
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


def _internal_target(target: str, modules: 'dict[str, Path]') -> 'str | None':
    if target in modules:
        return target
    package = target.rsplit(".", 1)[0]
    return package if package in modules else None


def _module_package(module: str, modules: 'dict[str, Path]') -> str:
    relative = module.removeprefix("linktools.ai").strip(".")
    if not relative:
        return ""
    parts = relative.split(".")
    path = modules.get(module)
    if path is not None and path.name == "__init__.py":
        return parts[0]
    return parts[0] if len(parts) > 1 else ""


def _import_targets(node: ast.ImportFrom, current: str, path: Path, modules: 'dict[str, Path]') -> 'tuple[str, ...]':
    if node.level == 0 and node.module and node.module.startswith("linktools.ai"):
        return (f"absolute:{path}:{node.lineno}",)
    base = _resolve(node.module or "", node.level, current, current_is_package=path.name == "__init__.py")
    if node.module:
        return (base,)
    candidates = tuple(f"{base}.{item.name}" for item in node.names if item.name != "*")
    return candidates or (base,)


def _dynamic_import_aliases(tree: ast.AST) -> 'tuple[set[str], set[str]]':
    importlib_module_aliases: set[str] = set()
    import_module_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_module_aliases.add(alias.asname or "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or "import_module")
    return importlib_module_aliases, import_module_aliases


def _is_dynamic_import_call(
    node: ast.Call,
    importlib_module_aliases: set[str],
    import_module_aliases: set[str],
) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id == "__import__" or node.func.id in import_module_aliases
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "import_module"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in importlib_module_aliases
    )


def _is_type_checking_import(node: ast.AST, checking_lines: 'set[int]') -> bool:
    return node.lineno in checking_lines


def build_report(source_root: "str | Path") -> "dict[str, JsonValue]":
    """Build a deterministic import graph from source files only."""
    root = Path(source_root)
    modules = {module_name(path, root): path for path in root.rglob("*.py") if "__pycache__" not in path.parts}
    runtime: dict[str, set[str]] = {name: set() for name in modules}
    type_checking: dict[str, set[str]] = {name: set() for name in modules}
    package_runtime: dict[str, set[str]] = {}
    dynamic_imports: list[str] = []
    reexports: dict[str, list[str]] = {}
    forbidden_calls = {"eval", "exec", "getattr", "import_module"}
    for name, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        importlib_module_aliases, import_module_aliases = _dynamic_import_aliases(tree)
        if path.name == "__init__.py":
            reexports[name] = sorted(
                target
                for node in tree.body
                if isinstance(node, ast.ImportFrom)
                for target in _import_targets(node, name, path, modules)
                if not target.startswith("absolute:")
            )
        checking_lines: set[int] = set()
        for checking_block in ast.walk(tree):
            if (
                isinstance(checking_block, ast.If)
                and isinstance(checking_block.test, ast.Name)
                and checking_block.test.id == "TYPE_CHECKING"
            ):
                checking_lines.update(
                    child.lineno
                    for statement in checking_block.body
                    for child in ast.walk(statement)
                    if isinstance(child, (ast.Import, ast.ImportFrom))
                )
        relative = path.relative_to(root).parts
        source_package = relative[0] if len(relative) > 1 else ""
        if source_package:
            package_runtime.setdefault(source_package, set())
        for node in ast.walk(tree):
            targets: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                targets = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets = _import_targets(node, name, path, modules)
                if any(target.startswith("absolute:") for target in targets):
                    dynamic_imports.extend(target.removeprefix("absolute:") for target in targets)
                    continue
            for target in targets:
                resolved = _internal_target(target, modules)
                if resolved is None:
                    continue
                target_graph = type_checking if _is_type_checking_import(node, checking_lines) else runtime
                target_graph[name].add(resolved)
                target_package = _module_package(resolved, modules)
                if source_package and target_package and target_package != source_package:
                    package_runtime[source_package].add(target_package)
            if isinstance(node, ast.Call) and _is_dynamic_import_call(
                node,
                importlib_module_aliases,
                import_module_aliases,
            ):
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else f"{ast.unparse(node.func.value)}.{node.func.attr}"
                )
                dynamic_imports.append(f"{path}:{node.lineno}:{function_name}")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                dynamic_imports.append(f"{path}:{node.lineno}:{node.func.id}")
            if isinstance(node, ast.Attribute) and node.attr in {"__dict__", "__getattr__"}:
                dynamic_imports.append(f"{path}:{node.lineno}:{node.attr}")
    return {
        "modules": sorted(modules),
        "runtime": {key: sorted(value) for key, value in sorted(runtime.items())},
        "type_checking": {key: sorted(value) for key, value in sorted(type_checking.items())},
        "dependency": {
            key: sorted(runtime[key] | type_checking[key])
            for key in sorted(modules)
        },
        "package_runtime": {key: sorted(value) for key, value in sorted(package_runtime.items())},
        "scc": [list(component) for component in _scc(runtime)],
        "dependency_scc": [
            list(component)
            for component in _scc({key: runtime[key] | type_checking[key] for key in modules})
        ],
        "package_scc": [list(component) for component in _scc(package_runtime)],
        "dynamic_imports": sorted(dynamic_imports),
        "reexports": reexports,
    }


def _source_packages(root: Path) -> 'tuple[str, ...]':
    return tuple(sorted(path.name for path in root.iterdir() if path.is_dir() and any(path.glob("*.py"))))


def _layout_errors(root: Path, expected_packages: 'tuple[str, ...]', public_modules: 'set[str]') -> 'list[str]':
    errors: list[str] = []
    actual = _source_packages(root)
    packages = expected_packages
    missing = sorted(set(expected_packages) - set(actual))
    unexpected = sorted(set(actual) - set(expected_packages))
    errors.extend(f"missing target package: {package}" for package in missing)
    errors.extend(f"unexpected target package: {package}" for package in unexpected)
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).parts
        if path.name == "__init__.py":
            continue
        if len(relative) == 1:
            module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
            if module not in public_modules:
                errors.append(f"module outside package: {path}")
            continue
        if not relative or relative[0] not in packages:
            errors.append(f"module outside package: {path}")
        elif relative[0] == "temporal":
            if len(relative) == 2 and relative[1] in {"_activity.py", "_context.py", "_gateway.py", "_worker.py"}:
                continue
            if len(relative) != 3 or relative[1] != "workflow":
                errors.append(f"invalid temporal depth: {path}")
        elif relative[0] == "runtime" and len(relative) == 3 and relative[1] == "state":
            continue
        elif len(relative) != 2:
            errors.append(f"invalid module depth: {path}")
    for package in packages:
        init = root / package / "__init__.py"
        if not init.is_file():
            errors.append(f"missing package init: {init}")
    return errors


def _public_module_errors(root: Path, public_modules: "set[str]") -> "list[str]":
    errors: list[str] = []
    actual_modules: set[str] = set()
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        module = path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        actual_modules.add(module)
        if path.stem.startswith("_"):
            if module in public_modules:
                errors.append(f"private module listed as public: {module}")
        elif module not in public_modules:
            errors.append(f"unclassified public module: {module}")
    for module in sorted(public_modules - actual_modules):
        errors.append(f"stale public module policy entry: {module}")
    for module in sorted(public_modules):
        if module.rsplit(".", 1)[-1].startswith("_"):
            errors.append(f"private module listed as public: {module}")
    return errors


def _private_import_errors(
    source_roots: "tuple[tuple[Path, str], ...]",
) -> "list[str]":
    errors: list[str] = []
    for root, prefix in source_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root).with_suffix("")
            current = f"{prefix}." + ".".join(relative.parts)
            current_is_package = path.name == "__init__.py"
            if current_is_package:
                current = current.rsplit(".__init__", 1)[0]
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            importlib_module_aliases, import_module_aliases = _dynamic_import_aliases(tree)
            for node in ast.walk(tree):
                targets: list[tuple[str, int]] = []
                if isinstance(node, ast.Import):
                    targets.extend((alias.name, node.lineno) for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    base = _resolve(node.module or "", node.level, current, current_is_package=current_is_package)
                    targets.append((base, node.lineno))
                    targets.extend(
                        (f"{base}.{alias.name}", node.lineno)
                        for alias in node.names
                        if alias.name.startswith("_")
                    )
                if (
                    isinstance(node, ast.Call)
                    and _is_dynamic_import_call(
                        node,
                        importlib_module_aliases,
                        import_module_aliases,
                    )
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    targets.append((node.args[0].value, node.lineno))
                for target, lineno in targets:
                    if not _is_supported_source_module(target):
                        continue
                    source_owner = _owner_package(current)
                    target_owner = _owner_package(target)
                    owner_parts = target_owner.split(".")
                    target_parts = target.split(".")
                    private_path = any(part.startswith("_") for part in target_parts[len(owner_parts):])
                    if source_owner != target_owner and private_path:
                        errors.append(
                            f"private cross-package import: {path}:{lineno}: "
                            f"{current} -> {target}"
                        )
    return errors


def _owner_package(module: str) -> str:
    if module == "linktools.ai":
        return "linktools.ai"
    if module.startswith("linktools.ai."):
        parts = module.split(".")
        return ".".join(parts[:3])
    if module == "linktools.commands.ai" or module.startswith("linktools.commands.ai."):
        return "linktools.commands.ai"
    return module


def _is_supported_source_module(module: str) -> bool:
    return module in {"linktools.ai", "linktools.commands.ai"} or module.startswith(
        ("linktools.ai.", "linktools.commands.ai.")
    )


def _init_errors(root: Path) -> 'list[str]':
    errors: list[str] = []
    for path in root.rglob("__init__.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                continue
            errors.append(f"non-static package init: {path}:{node.lineno}")
    return errors


@dataclass(frozen=True, slots=True)
class ArchitectureCheckResult:
    passed: bool
    errors: "tuple[str, ...]"
    report: "dict[str, JsonValue]"


class ArchitecturePolicyChecker:
    """Run the package, graph, naming, and ownership gates."""

    def check(self, source_root: "str | Path") -> ArchitectureCheckResult:
        root = Path(source_root)
        report = build_report(root)
        commands_root = root.parents[2] / "src" / "linktools" / "commands" / "ai"
        source_roots = ((root, "linktools.ai"), (commands_root, "linktools.commands.ai"))
        policy_path = (
            root.parents[2]
            / "scripts"
            / "build"
            / "matrix"
            / "linktools-ai-package-policy.json"
        )
        policy: dict[str, JsonValue] = {}
        policy_errors: list[str] = []
        if not policy_path.is_file():
            policy_errors.append(f"missing architecture policy: {policy_path}")
        else:
            try:
                loaded = json.loads(policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                policy_errors.append(f"invalid architecture policy: {policy_path}: {error}")
            else:
                if not isinstance(loaded, dict):
                    policy_errors.append(f"architecture policy must be an object: {policy_path}")
                else:
                    policy = cast("dict[str, JsonValue]", loaded)
        packages_value = policy.get("top_level_packages", [])
        expected_packages = tuple(item for item in packages_value if isinstance(item, str)) if isinstance(packages_value, list) else ()
        public_modules_value = policy.get("public_modules", [])
        public_modules = (
            {item for item in public_modules_value if isinstance(item, str)}
            if isinstance(public_modules_value, list)
            else set()
        )
        errors = [
            *policy_errors,
            *_layout_errors(root, expected_packages, public_modules),
            *_init_errors(root),
            *check_names(root),
            *check_files(root),
        ]
        errors.extend(_public_module_errors(root, public_modules))
        if "private_imports" in policy:
            errors.append("private_imports policy is forbidden")
        errors.extend(_private_import_errors(source_roots))
        errors.extend(f"runtime SCC: {component}" for component in report["scc"] if isinstance(component, list))
        errors.extend(f"dependency SCC: {component}" for component in report["dependency_scc"] if isinstance(component, list))
        errors.extend(f"package SCC: {component}" for component in report["package_scc"] if isinstance(component, list))
        errors.extend(f"forbidden import or reflection: {item}" for item in report["dynamic_imports"] if isinstance(item, str))
        dependencies = policy.get("dependencies", {})
        dependency_map = cast("dict[str, JsonValue]", dependencies) if isinstance(dependencies, dict) else {}
        package_runtime = report["package_runtime"]
        if isinstance(package_runtime, dict):
            for source_package, targets in package_runtime.items():
                allowed_value = dependency_map.get(source_package, [])
                allowed = (
                    {item for item in allowed_value if isinstance(item, str)}
                    if isinstance(allowed_value, list)
                    else set()
                )
                if isinstance(targets, list):
                    errors.extend(
                        f"dependency policy: {source_package} -> {target}"
                        for target in targets
                        if isinstance(target, str) and target not in allowed
                    )
        module_dependencies = policy.get("module_dependencies", {})
        if isinstance(module_dependencies, dict):
            runtime_graph = report["runtime"]
            if isinstance(runtime_graph, dict):
                modules = set(report["modules"]) if isinstance(report.get("modules"), list) else set()
                module_paths = {
                    module_name(path, root): path
                    for path in root.rglob("*.py")
                    if "__pycache__" not in path.parts
                }
                for source_module, allowed_value in module_dependencies.items():
                    if not isinstance(source_module, str):
                        continue
                    normalized_source = source_module if source_module.startswith("linktools.ai.") else f"linktools.ai.{source_module}"
                    if normalized_source not in modules:
                        errors.append(f"stale module dependency policy: {source_module}")
                        continue
                    allowed = (
                        {item for item in allowed_value if isinstance(item, str)}
                        if isinstance(allowed_value, list)
                        else set()
                    )
                    targets = runtime_graph.get(normalized_source, [])
                    if not isinstance(targets, list):
                        continue
                    for target in targets:
                        if not isinstance(target, str):
                            continue
                        target_package = _module_package(target, module_paths)
                        source_package = _module_package(normalized_source, module_paths)
                        if target_package and target_package != source_package and target_package not in allowed:
                            errors.append(
                                f"module dependency policy: {normalized_source} -> {target}"
                            )
        forbidden_nodes = {"Spec" + "Store", "__getattr__"}
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in forbidden_nodes:
                    errors.append(f"forbidden symbol: {path}:{node.lineno}:{node.id}")
                elif isinstance(node, ast.Attribute) and node.attr in forbidden_nodes:
                    errors.append(f"forbidden symbol: {path}:{node.lineno}:{node.attr}")
        return ArchitectureCheckResult(not errors, tuple(dict.fromkeys(errors)), report)


__all__ = ["ArchitectureCheckResult", "ArchitecturePolicyChecker", "build_report", "module_name"]
