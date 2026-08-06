#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Static cohesion checks for ownership, forwarding modules, and public types."""

import ast
from pathlib import Path


def check_files(source_root: "str | Path") -> "tuple[str, ...]":
    errors: list[str] = []
    for path in Path(source_root).rglob("*.py"):
        if path.name == "__init__.py" or "__pycache__" in path.parts:
            continue
        if any(token in path.name.lower() for token in ("helper", "utils", "manager", "common")):
            errors.append(str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        public_definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
        ]
        for node in public_definitions:
            annotations = list(_annotations(node))
            for annotation in annotations:
                for child in ast.walk(annotation):
                    if isinstance(child, ast.Name) and child.id in {"Any", "object"}:
                        errors.append(f"public untyped annotation: {path}:{child.lineno}:{child.id}")
                    elif isinstance(child, ast.Attribute) and child.attr == "Any":
                        errors.append(f"public untyped annotation: {path}:{child.lineno}:Any")
        if _is_forwarding_module(tree):
            errors.append(f"empty forwarding module: {path}")
        if path.parts[-2:-1] == ("asset",) and _duplicates_storage_abstraction(public_definitions):
            errors.append(f"duplicate storage abstraction in asset module: {path}")
    return tuple(errors)


def _annotations(node: ast.AST) -> tuple[ast.expr, ...]:
    values: list[ast.expr] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        values.extend(argument.annotation for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if argument.annotation is not None)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            values.append(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            values.append(node.args.kwarg.annotation)
        if node.returns is not None:
            values.append(node.returns)
    elif isinstance(node, ast.ClassDef):
        values.extend(statement.annotation for statement in node.body if isinstance(statement, ast.AnnAssign))
    return tuple(values)


def _is_forwarding_module(tree: ast.Module) -> bool:
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
    if definitions:
        return False
    meaningful = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        and not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets))
    ]
    return not meaningful and any(isinstance(node, ast.ImportFrom) for node in tree.body)


def _duplicates_storage_abstraction(definitions: list[ast.stmt]) -> bool:
    forbidden = ("StorageLayer", "StorageComposition", "ContentCache", "RevisionSource", "VersionedStorage", "StorageLock")
    return any(isinstance(node, ast.ClassDef) and node.name.startswith(forbidden) for node in definitions)


__all__ = ["check_files"]
