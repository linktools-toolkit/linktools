#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-lived architecture invariants for LinkTools AI production code."""

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

_AI_ROOT = "linktools.ai"


@dataclass(frozen=True)
class ArchitectureCheckResult:
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class _ModuleInfo:
    name: str
    path: Path
    tree: ast.Module
    exports: tuple[str, ...] | None
    bindings: frozenset[str]


@dataclass(frozen=True)
class _Inventory:
    modules: Mapping[str, _ModuleInfo]
    runtime_edges: Mapping[str, frozenset[str]]


class ArchitecturePolicyChecker:
    """Check runtime acyclicity and LinkTools-owned public/private boundaries."""

    def check(
        self,
        source_root: str | Path,
        *,
        external_roots: Iterable[str | Path] = (),
    ) -> ArchitectureCheckResult:
        root = Path(source_root).resolve()
        modules, parse_errors = _build_modules(root)
        errors = list(parse_errors)
        inventory = _build_inventory(modules)
        errors.extend(_validate_exports(inventory.modules))
        errors.extend(_validate_runtime_cycles(inventory))
        errors.extend(_validate_cross_owner_access(inventory, inventory.modules.values()))

        external_modules: list[_ModuleInfo] = []
        for external_root in external_roots:
            external_modules.extend(_build_external_modules(Path(external_root).resolve()))
        errors.extend(_validate_cross_owner_access(inventory, external_modules))
        return ArchitectureCheckResult(tuple(sorted(set(errors))))


def _build_modules(root: Path) -> tuple[dict[str, _ModuleInfo], tuple[str, ...]]:
    modules: dict[str, _ModuleInfo] = {}
    errors: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        name = ".".join([_AI_ROOT, *parts]) if parts else _AI_ROOT
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            errors.append(f"cannot parse {path}: {error}")
            continue
        exports, export_error = _parse_all(tree)
        if export_error is not None:
            errors.append(f"{name}: {export_error}")
        modules[name] = _ModuleInfo(name, path, tree, exports, _bindings(tree))
    return modules, tuple(errors)


def _build_external_modules(root: Path) -> list[_ModuleInfo]:
    modules: list[_ModuleInfo] = []
    if not root.is_dir():
        return modules
    for path in sorted(root.rglob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "linktools.ai" not in source:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        relative = path.relative_to(root)
        parts = list(relative.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        name = ".".join(["linktools", *parts]) if parts else "linktools"
        modules.append(_ModuleInfo(name, path, tree, None, frozenset()))
    return modules


def _build_inventory(modules: Mapping[str, _ModuleInfo]) -> _Inventory:
    known = frozenset(modules)
    edges: dict[str, frozenset[str]] = {}
    for name, module in modules.items():
        targets: set[str] = set()
        for node in _runtime_imports(module.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in known:
                        targets.add(alias.name)
            else:
                base = _resolve_from_module(name, module.path.name == "__init__.py", node)
                if base is None:
                    continue
                if base in known:
                    targets.add(base)
                for alias in node.names:
                    child = f"{base}.{alias.name}"
                    if child in known:
                        targets.add(child)
        edges[name] = frozenset(targets)
    return _Inventory(modules, edges)


def _validate_exports(modules: Mapping[str, _ModuleInfo]) -> tuple[str, ...]:
    errors: list[str] = []
    for module in modules.values():
        if module.exports is None:
            continue
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in module.exports:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            errors.append(
                f"{module.name}: duplicate __all__ exports: {', '.join(sorted(duplicates))}"
            )
        missing = sorted(set(module.exports) - module.bindings)
        if missing:
            errors.append(
                f"{module.name}: __all__ exports unbound names: {', '.join(missing)}"
            )
    return tuple(errors)


def _validate_runtime_cycles(inventory: _Inventory) -> tuple[str, ...]:
    errors: list[str] = []
    for component in _strongly_connected_components(inventory.runtime_edges):
        if _is_cycle(component, inventory.runtime_edges):
            errors.append(f"runtime module cycle: {' -> '.join(sorted(component))}")

    owner_edges: dict[str, set[str]] = {}
    for source, targets in inventory.runtime_edges.items():
        source_owner = _owner(source)
        if source_owner in {None, "__root__"}:
            continue
        owner_edges.setdefault(source_owner, set())
        for target in targets:
            target_owner = _owner(target)
            if target_owner not in {None, "__root__"} and target_owner != source_owner:
                owner_edges[source_owner].add(target_owner)
                owner_edges.setdefault(target_owner, set())
    frozen = {owner: frozenset(targets) for owner, targets in owner_edges.items()}
    for component in _strongly_connected_components(frozen):
        if _is_cycle(component, frozen):
            errors.append(f"runtime owner cycle: {' -> '.join(sorted(component))}")
    return tuple(errors)


def _is_cycle(
    component: tuple[str, ...],
    graph: Mapping[str, frozenset[str]],
) -> bool:
    return len(component) > 1 or (
        len(component) == 1 and component[0] in graph.get(component[0], frozenset())
    )


def _validate_cross_owner_access(
    inventory: _Inventory,
    sources: Iterable[_ModuleInfo],
) -> tuple[str, ...]:
    errors: list[str] = []
    known = frozenset(inventory.modules)
    for source in sources:
        source_owner = _owner(source.name)
        aliases: dict[str, str] = {}
        for node in _runtime_imports(source.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in known:
                        continue
                    target = alias.name
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    aliases[bound] = target if alias.asname else alias.name.split(".", 1)[0]
                    errors.extend(_check_module_boundary(source, source_owner, target))
            else:
                base = _resolve_from_module(source.name, source.path.name == "__init__.py", node)
                if base is None or not base.startswith(_AI_ROOT) or base not in known:
                    continue
                errors.extend(_check_module_boundary(source, source_owner, base))
                for alias in node.names:
                    if alias.name == "*":
                        if _owner(base) != source_owner and inventory.modules[base].exports is None:
                            errors.append(
                                f"{source.name}: cross-owner star import from {base} "
                                "requires static __all__"
                            )
                        continue
                    errors.extend(
                        _check_symbol_boundary(
                            inventory,
                            source,
                            source_owner,
                            base,
                            alias.name,
                        )
                    )
                    child = f"{base}.{alias.name}"
                    if child in known:
                        errors.extend(_check_module_boundary(source, source_owner, child))
                        aliases[alias.asname or alias.name] = child
                    else:
                        aliases[alias.asname or alias.name] = f"{base}.{alias.name}"

        for node in ast.walk(source.tree):
            if not isinstance(node, ast.Attribute):
                continue
            chain = _attribute_chain(node)
            if chain is None:
                continue
            root, parts = chain[0], chain[1:]
            if root in aliases:
                base = aliases[root]
                full = ".".join([base, *parts])
            else:
                full = ".".join(chain)
            module_name, symbol = _module_and_symbol(full, known)
            if module_name is None or symbol is None:
                continue
            errors.extend(
                _check_symbol_boundary(
                    inventory,
                    source,
                    source_owner,
                    module_name,
                    symbol,
                )
            )
    return tuple(errors)


def _check_module_boundary(
    source: _ModuleInfo,
    source_owner: str | None,
    target: str,
) -> tuple[str, ...]:
    target_owner = _owner(target)
    if target_owner is None or target_owner == source_owner:
        return ()
    relative = target[len(_AI_ROOT) + 1 :] if target != _AI_ROOT else ""
    parts = relative.split(".") if relative else []
    if any(part.startswith("_") for part in parts):
        return (f"{source.name}: cross-owner private module access: {target}",)
    return ()


def _check_symbol_boundary(
    inventory: _Inventory,
    source: _ModuleInfo,
    source_owner: str | None,
    target_module: str,
    symbol: str,
) -> tuple[str, ...]:
    target_owner = _owner(target_module)
    if target_owner is None or target_owner == source_owner:
        return ()
    module_errors = _check_module_boundary(source, source_owner, target_module)
    if module_errors:
        return module_errors
    module = inventory.modules[target_module]
    if symbol.startswith("_"):
        return (f"{source.name}: cross-owner private symbol access: {target_module}.{symbol}",)
    exports = module.exports or ()
    if symbol not in exports:
        return (f"{source.name}: cross-owner non-exported symbol access: {target_module}.{symbol}",)
    return ()


def _parse_all(tree: ast.Module) -> tuple[tuple[str, ...] | None, str | None]:
    top_level: list[ast.AST] = []
    for node in tree.body:
        value = _all_assignment_value(node)
        if value is not None:
            top_level.append(value)
        elif _writes_all(node):
            return None, "__all__ must be one static string sequence"
        if _nested_module_all_write(node):
            return None, "__all__ must be one static string sequence"

    if not top_level:
        return None, None
    if len(top_level) != 1:
        return None, "__all__ must be assigned exactly once"
    value = top_level[0]
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None, "__all__ must be a static list or tuple of strings"
    exports: list[str] = []
    for item in value.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None, "__all__ must contain only string literals"
        exports.append(item.value)
    return tuple(exports), None


def _nested_module_all_write(node: ast.AST) -> bool:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return False
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if _writes_all(child) or _nested_module_all_write(child):
            return True
    return False


def _all_assignment_value(node: ast.AST) -> ast.AST | None:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
    ):
        return node.value
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "__all__"
    ):
        return node.value
    return None


def _writes_all(node: ast.AST) -> bool:
    if isinstance(node, ast.Assign):
        return any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    if isinstance(node, ast.AnnAssign):
        return isinstance(node.target, ast.Name) and node.target.id == "__all__"
    if isinstance(node, ast.AugAssign):
        return isinstance(node.target, ast.Name) and node.target.id == "__all__"
    return False


def _bindings(tree: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(_target_names(target))
    return frozenset(names)


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        result: set[str] = set()
        for item in target.elts:
            result.update(_target_names(item))
        return result
    return set()


def _runtime_imports(tree: ast.Module) -> tuple[ast.Import | ast.ImportFrom, ...]:
    result: list[ast.Import | ast.ImportFrom] = []

    def visit(node: ast.AST, type_checking: bool = False) -> None:
        if isinstance(node, ast.If) and _is_type_checking_test(node.test):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, type_checking)
            return
        if not type_checking and isinstance(node, (ast.Import, ast.ImportFrom)):
            result.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child, type_checking)

    visit(tree)
    return tuple(result)


def _is_type_checking_test(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "TYPE_CHECKING" or (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr == "TYPE_CHECKING"
    )


def _resolve_from_module(current: str, is_package: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = current if is_package else current.rsplit(".", 1)[0]
    parts = package.split(".")
    up = node.level - 1
    if up > len(parts):
        return None
    base = parts[: len(parts) - up]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _module_and_symbol(full: str, known: frozenset[str]) -> tuple[str | None, str | None]:
    parts = full.split(".")
    for index in range(len(parts) - 1, 0, -1):
        module = ".".join(parts[:index])
        if module in known:
            return module, parts[index]
    return None, None


def _attribute_chain(node: ast.Attribute) -> tuple[str, ...] | None:
    parts: list[str] = [node.attr]
    current: ast.AST = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def _owner(module: str) -> str | None:
    if module == _AI_ROOT:
        return "__root__"
    prefix = _AI_ROOT + "."
    if not module.startswith(prefix):
        return None
    return module[len(prefix) :].split(".", 1)[0]


def _strongly_connected_components(
    graph: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, frozenset()):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] != indices[node]:
            return
        component: list[str] = []
        while True:
            item = stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == node:
                break
        components.append(tuple(component))

    nodes = set(graph)
    for targets in graph.values():
        nodes.update(targets)
    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return tuple(components)


__all__ = ["ArchitectureCheckResult", "ArchitecturePolicyChecker"]
