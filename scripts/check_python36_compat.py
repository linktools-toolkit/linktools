#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Static scan for Python 3.7+-only syntax/stdlib usage outside linktools-ai.

This is a first-pass, AST-based heuristic -- it flags syntax that cannot
possibly run under Python 3.6 (match/case, walrus, positional-only
parameters, `from __future__ import annotations`, except*, unquoted PEP
604/585 annotations) plus a short list of stdlib imports that need a
backport or a try/except fallback below 3.7/3.8. It is not a substitute for
actually running ``python3.6 -m compileall`` -- that remains the final
authority (see the project's Python 3.6 CI job).

Usage:
    python scripts/check_python36_compat.py
    python scripts/check_python36_compat.py --json
"""
import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Packages that must stay importable/compileable on Python 3.6.
SCANNED_ROOTS = [
    "linktools/src",
    "linktools-common/src",
    "linktools-mobile/src",
    "linktools-cntr/src",
]

# linktools-ai (and its tests) are the one package allowed to require >=3.10.
EXCLUDED_DIR_NAMES = {"__pycache__", "linktools-ai"}

# Stdlib modules that are only unconditionally usable from these versions --
# flagged only when imported without an accompanying try/except fallback in
# the same file (the project's established backport pattern).
_VERSIONED_STDLIB = {
    "zoneinfo": "3.9",
    "tomllib": "3.11",
    "graphlib": "3.9",
}


class _Violation:
    def __init__(self, path, line, kind, detail):
        self.path = path
        self.line = line
        self.kind = kind
        self.detail = detail

    def __str__(self):
        return "%s:%d: [%s] %s" % (self.path, self.line, self.kind, self.detail)

    def to_dict(self):
        return {"path": self.path, "line": self.line, "kind": self.kind, "detail": self.detail}


def _iter_py_files():
    for root in SCANNED_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
                continue
            yield path


def _is_string_annotation(node):
    """True if ``node`` is a quoted-string annotation (the project's
    required style for any annotation containing ``|``/``[...]``), which
    is inert at runtime on any Python version regardless of its contents.
    """
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _annotation_nodes(tree):
    """Yield every live (non-string) annotation expression in the module:
    function arg/return annotations and variable annotations."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            # `posonlyargs` was added to the ast.arguments grammar in Python
            # 3.8 (PEP 570); this scanner itself runs under Python 3.6 in CI,
            # where the attribute does not exist at all.
            candidates = (
                list(getattr(args, "posonlyargs", ())) + list(args.args) + list(args.kwonlyargs)
                + ([args.vararg] if args.vararg else [])
                + ([args.kwarg] if args.kwarg else [])
            )
            for arg in candidates:
                if arg.annotation is not None:
                    yield arg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            if node.annotation is not None:
                yield node.annotation


_PEP585_GENERIC_NAMES = {"list", "dict", "set", "frozenset", "tuple", "type"}
# Attribute-form generics from typing/collections.abc that also only support
# unquoted `[...]` subscripting from Python 3.9+ (PEP 585 covers these too).
_PEP585_GENERIC_ATTRS = {"Mapping", "MutableMapping", "Sequence", "Iterable", "Iterator"}


def _generic_name(value_node):
    """Return the subscripted generic's bare name for both `list[...]`
    (ast.Name) and `collections.abc.Mapping[...]` (ast.Attribute) forms."""
    if isinstance(value_node, ast.Name):
        return value_node.id
    if isinstance(value_node, ast.Attribute):
        return value_node.attr
    return None


def _annotation_violation(node):
    """Return a short description if ``node`` -- or anything nested inside
    it (e.g. `Optional[int | str]`'s inner `int | str`) -- is a live
    (unquoted) PEP 604 (`X | Y`) or PEP 585 (`list[str]`) annotation
    expression, both only usable unquoted from Python 3.10/3.9
    respectively. Returns None for anything else, including a
    quoted-string annotation (checked only at the top level: a string
    nested inside a live expression, e.g. `Optional["Foo"]`, is inert)."""
    if _is_string_annotation(node):
        return None
    for sub in ast.walk(node):
        if isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.BitOr):
            return "unquoted `X | Y` union annotation (PEP 604, Python 3.10+)"
        if isinstance(sub, ast.Subscript):
            name = _generic_name(sub.value)
            if name in _PEP585_GENERIC_NAMES or name in _PEP585_GENERIC_ATTRS:
                return "unquoted `%s[...]` generic annotation (PEP 585, Python 3.9+)" % name
    return None


def scan_file(path):
    violations = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [_Violation(str(path), 0, "unreadable", str(exc))]

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [_Violation(str(path), exc.lineno or 0, "syntax-error", str(exc))]

    for node in ast.walk(tree):
        node_type_name = type(node).__name__
        if node_type_name == "Match":
            violations.append(_Violation(str(path), node.lineno, "match-statement",
                                         "`match`/`case` requires Python 3.10+"))
        elif node_type_name == "NamedExpr":
            violations.append(_Violation(str(path), node.lineno, "walrus-operator",
                                         "`:=` requires Python 3.8+"))
        elif node_type_name == "TryStar":
            violations.append(_Violation(str(path), node.lineno, "except-star",
                                         "`except*` requires Python 3.11+"))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if getattr(node.args, "posonlyargs", None):
                violations.append(_Violation(str(path), node.lineno, "positional-only-params",
                                             "`/` positional-only parameters require Python 3.8+"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" and any(a.name == "annotations" for a in node.names):
                violations.append(_Violation(str(path), node.lineno, "future-annotations",
                                             "`from __future__ import annotations` changes annotation "
                                             "evaluation semantics; this project quotes annotations "
                                             "manually instead"))
            elif node.module == "typing" and any(a.name == "Self" for a in node.names):
                violations.append(_Violation(str(path), node.lineno, "typing-self",
                                             "`typing.Self` requires Python 3.11+"))
            elif node.module in _VERSIONED_STDLIB and "except" not in text:
                violations.append(_Violation(
                    str(path), node.lineno, "versioned-stdlib",
                    "`%s` needs Python %s+ (no try/except fallback found in this file)"
                    % (node.module, _VERSIONED_STDLIB[node.module])))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in _VERSIONED_STDLIB and "except" not in text:
                    violations.append(_Violation(
                        str(path), node.lineno, "versioned-stdlib",
                        "`%s` needs Python %s+ (no try/except fallback found in this file)"
                        % (alias.name, _VERSIONED_STDLIB[alias.name])))

    for annotation in _annotation_nodes(tree):
        detail = _annotation_violation(annotation)
        if detail:
            violations.append(_Violation(str(path), getattr(annotation, "lineno", 0),
                                         "unquoted-annotation", detail))

    return violations


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", dest="as_json", action="store_true", help="output JSON")
    args = parser.parse_args()

    all_violations = []
    for path in _iter_py_files():
        all_violations.extend(scan_file(path))

    if args.as_json:
        print(json.dumps([v.to_dict() for v in all_violations], indent=2))
    else:
        if not all_violations:
            print("No Python 3.6-incompatible syntax found (static scan; run the real "
                  "Python 3.6 compileall job for the final check).")
        else:
            for v in all_violations:
                print(str(v))
            print("\n%d potential Python 3.6 incompatibilit%s found." % (
                len(all_violations), "y" if len(all_violations) == 1 else "ies"))

    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
