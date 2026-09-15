#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
from pathlib import Path

from .architecture import ArchitecturePolicyChecker


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "linktools-ai" / "src" / "linktools" / "ai"
CORE_SOURCE_ROOT = ROOT / "linktools" / "src" / "linktools"


def _has_reserved_prefix(name):
    return name.lstrip("_").lower().startswith("linktools")


def _identifier_errors():
    errors = set()
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            errors.add("cannot parse %s: %s" % (path, error))
            continue
        for node in ast.walk(tree):
            name = None
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                name = node.id
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                name = node.attr
            if name is not None and _has_reserved_prefix(name):
                relative = path.relative_to(SOURCE_ROOT)
                errors.add(
                    "%s:%s: AI-owned identifier uses reserved linktools prefix: %s"
                    % (relative, getattr(node, "lineno", 0), name)
                )
    return tuple(sorted(errors))


def main() -> int:
    result = ArchitecturePolicyChecker().check(
        SOURCE_ROOT,
        external_roots=(CORE_SOURCE_ROOT,),
    )
    errors = tuple(sorted(set(result.errors).union(_identifier_errors())))
    if not errors:
        print("[+] linktools-ai architecture gate passed")
        return 0
    for error in errors:
        print("[-] %s" % error)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
