#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static scan for Python 3.7+-only syntax and stdlib usage."""

import argparse
import ast
import json
import sys
import typing
from pathlib import Path

_IGNORED_DIR_NAMES = {"__pycache__", "build", "dist", ".venv"}
_VERSIONED_STDLIB = {
    "zoneinfo": "3.9",
    "tomllib": "3.11",
    "graphlib": "3.9",
}
_PEP585_GENERIC_NAMES = {"list", "dict", "set", "frozenset", "tuple", "type"}
_PEP585_GENERIC_ATTRS = {"Mapping", "MutableMapping", "Sequence", "Iterable", "Iterator"}


class _Violation:
    def __init__(self, path, line, kind, detail):
        self.path = path
        self.line = line
        self.kind = kind
        self.detail = detail

    def __str__(self):
        return "%s:%d: [%s] %s" % (self.path, self.line, self.kind, self.detail)

    def to_dict(self) -> "typing.Dict[str, typing.Any]":
        return {"path": self.path, "line": self.line, "kind": self.kind, "detail": self.detail}


def _iter_py_files(paths):
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError("path does not exist: %s" % path)
        if path.is_file():
            if path.suffix != ".py":
                raise ValueError("file is not Python source: %s" % path)
            candidates = (path,)
        elif path.is_dir():
            candidates = sorted(path.rglob("*.py"))
        else:
            raise ValueError("unsupported path: %s" % path)
        for candidate in candidates:
            if any(part in _IGNORED_DIR_NAMES for part in candidate.parts):
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _literal_value(node):
    for type_name, attribute in (
        ("Constant", "value"),
        ("Str", "s"),
        ("Num", "n"),
        ("NameConstant", "value"),
    ):
        node_type = getattr(ast, type_name, None)
        if node_type is not None and isinstance(node, node_type):
            return True, getattr(node, attribute)
    return False, None


def _string_annotation_value(node):
    is_literal, value = _literal_value(node)
    if is_literal and isinstance(value, str):
        return value
    return None


def _is_string_annotation(node):
    return _string_annotation_value(node) is not None


def _annotation_nodes(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            candidates = (
                list(getattr(args, "posonlyargs", ()))
                + list(args.args)
                + list(args.kwonlyargs)
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            for arg in candidates:
                if arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


def _generic_name(value_node):
    if isinstance(value_node, ast.Name):
        return value_node.id
    if isinstance(value_node, ast.Attribute):
        return value_node.attr
    return None


def _annotation_violation(node):
    if _is_string_annotation(node):
        return None
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.BitOr):
            return "unquoted `X | Y` union annotation (PEP 604, Python 3.10+)"
        if isinstance(child, ast.Subscript):
            name = _generic_name(child.value)
            if name in _PEP585_GENERIC_NAMES or name in _PEP585_GENERIC_ATTRS:
                return "unquoted `%s[...]` generic annotation (PEP 585, Python 3.9+)" % name
    return None


def _collect_union_leaves(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _collect_union_leaves(node.left) + _collect_union_leaves(node.right)
    return [node]


def _union_member_key(node):
    is_literal, value = _literal_value(node)
    if is_literal:
        return repr(value)
    if isinstance(node, ast.Name):
        return node.id
    return None


def _duplicate_union_violation(node):
    annotation = _string_annotation_value(node)
    if annotation is not None:
        try:
            node = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return None
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)):
        return None
    seen = set()
    for member in _collect_union_leaves(node):
        key = _union_member_key(member)
        if key is None:
            continue
        if key in seen:
            return "duplicate union member `%s`" % key
        seen.add(key)
    return None


def _scan_file(path):
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return [_Violation(str(path), 0, "unreadable", str(error))]
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        return [_Violation(str(path), error.lineno or 0, "syntax-error", str(error))]

    for node in ast.walk(tree):
        node_type_name = type(node).__name__
        if node_type_name == "Match":
            violations.append(_Violation(str(path), node.lineno, "match-statement", "`match`/`case` requires Python 3.10+"))
        elif node_type_name == "NamedExpr":
            violations.append(_Violation(str(path), node.lineno, "walrus-operator", "`:=` requires Python 3.8+"))
        elif node_type_name == "TryStar":
            violations.append(_Violation(str(path), node.lineno, "except-star", "`except*` requires Python 3.11+"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node.args, "posonlyargs", None):
            violations.append(_Violation(str(path), node.lineno, "positional-only-params", "`/` positional-only parameters require Python 3.8+"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and any(alias.name == "annotations" for alias in node.names):
                violations.append(_Violation(str(path), node.lineno, "future-annotations", "`from __future__ import annotations` changes annotation evaluation semantics; quote annotations instead"))
            elif node.module == "typing" and any(alias.name == "Self" for alias in node.names):
                violations.append(_Violation(str(path), node.lineno, "typing-self", "`typing.Self` requires Python 3.11+"))
            elif node.module in _VERSIONED_STDLIB and "except" not in text:
                violations.append(_Violation(str(path), node.lineno, "versioned-stdlib", "`%s` needs Python %s+ (no try/except fallback found in this file)" % (node.module, _VERSIONED_STDLIB[node.module])))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _VERSIONED_STDLIB and "except" not in text:
                    violations.append(_Violation(str(path), node.lineno, "versioned-stdlib", "`%s` needs Python %s+ (no try/except fallback found in this file)" % (alias.name, _VERSIONED_STDLIB[alias.name])))

    for annotation in _annotation_nodes(tree):
        detail = _annotation_violation(annotation)
        if detail:
            violations.append(_Violation(str(path), getattr(annotation, "lineno", 0), "unquoted-annotation", detail))
    for annotation in _annotation_nodes(tree):
        detail = _duplicate_union_violation(annotation)
        if detail:
            violations.append(_Violation(str(path), getattr(annotation, "lineno", 0), "duplicate-union", detail))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="as_json", action="store_true", help="output JSON")
    parser.add_argument("paths", nargs="+", help="Python files or source directories to scan")
    args = parser.parse_args()
    try:
        files = tuple(_iter_py_files(args.paths))
    except ValueError as error:
        parser.error(str(error))
    violations = []
    for path in files:
        violations.extend(_scan_file(path))
    if args.as_json:
        print(json.dumps([violation.to_dict() for violation in violations], indent=2))
    elif violations:
        for violation in violations:
            print(str(violation))
        print("\n%d potential Python 3.6 incompatibilit%s found." % (len(violations), "y" if len(violations) == 1 else "ies"))
    else:
        print("No Python 3.6-incompatible syntax found in %d file(s)." % len(files))
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
