#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Snake-case filename and namespace-scoped package policy."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _NameNode:
    semantic_leaf: str
    parent_namespace: tuple[str, ...]
    path: Path


def _semantic_stem(path: Path) -> str:
    return path.stem.removeprefix("_")


def _is_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return right[: len(left)] == left


def _module_nodes(root: Path) -> tuple[_NameNode, ...]:
    nodes: list[_NameNode] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or path.stem == "__init__":
            continue
        relative = path.relative_to(root).with_suffix("")
        nodes.append(_NameNode(_semantic_stem(path), relative.parts[:-1], path))
    return tuple(nodes)


def _package_nodes(root: Path) -> tuple[_NameNode, ...]:
    nodes: list[_NameNode] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir() or path.name == "__pycache__" or not (path / "__init__.py").is_file():
            continue
        relative = path.relative_to(root)
        nodes.append(_NameNode(path.name, relative.parts[:-1], path))
    return tuple(nodes)


def _collision_errors(nodes: tuple[_NameNode, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if left.semantic_leaf != right.semantic_leaf:
                continue
            if _is_required_owner_pair(left.path, right.path):
                continue
            if not (
                _is_prefix(left.parent_namespace, right.parent_namespace)
                or _is_prefix(right.parent_namespace, left.parent_namespace)
            ):
                continue
            errors.append(
                "namespace semantic-name collision:\n"
                f"  {left.path}\n"
                f"  {right.path}"
            )
    return tuple(errors)


def _is_required_owner_pair(left: Path, right: Path) -> bool:
    paths = {left.as_posix(), right.as_posix()}
    required_pairs = (
        ("/agent", "/runtime/_agent.py"),
        ("/workspace", "/capability/_workspace.py"),
        ("/runtime/_history.py", "/runtime/state/_history.py"),
        ("/runtime/_memory.py", "/runtime/state/_memory.py"),
        ("/runtime/_plan.py", "/runtime/state/_plan.py"),
    )
    return any(
        any(path.endswith(first) for path in paths)
        and any(path.endswith(second) for path in paths)
        for first, second in required_pairs
    )


def check_names(source_root: "str | Path") -> "tuple[str, ...]":
    root = Path(source_root)
    modules = _module_nodes(root)
    packages = _package_nodes(root)
    errors: list[str] = []
    for node in modules:
        if not re.fullmatch(r"_?[a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)*", node.path.stem):
            errors.append(str(node.path))
    for node in packages:
        if not re.fullmatch(
            r"[a-z][a-z0-9]*",
            node.path.name,
        ):
            errors.append(str(node.path))
    errors.extend(
        _collision_errors(
            tuple(sorted((*modules, *packages), key=lambda node: str(node.path)))
        )
    )
    return tuple(errors)


__all__ = ["check_names"]
