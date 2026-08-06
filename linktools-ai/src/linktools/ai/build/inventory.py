#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Source file and public symbol inventory."""

import ast
from pathlib import Path


def build_inventory(source_root: "str | Path") -> "dict[str, object]":
    files = []
    symbols = []
    for path in sorted(Path(source_root).rglob("*.py")):
        files.append(str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        symbols.extend(f"{path}:{node.name}" for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)))
    return {"files": tuple(files), "symbols": tuple(sorted(symbols))}


class SourceInventoryBuilder:
    """Build the deterministic source and public-symbol inventory."""

    def build(self, source_root: "str | Path") -> "dict[str, object]":
        return build_inventory(source_root)


__all__ = ["SourceInventoryBuilder", "build_inventory"]
