#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cold-start architecture and import checks. MT-51 MT-52."""

import importlib
from pathlib import Path

from linktools.ai.build.architecture import build_report
from linktools.ai.build.cohesion import check_files
from linktools.ai.build.names import check_names


def test_source_graph_is_acyclic_and_static() -> None:
    report = build_report("linktools-ai/src/linktools/ai")
    assert report["scc"] == ()
    assert report["dynamic_imports"] == ()


def test_names_and_module_imports_are_clean() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    assert check_names(root, "linktools-ai/build/name-exceptions.json") == ()
    assert check_files(root) == ()
    for path in sorted(root.rglob("*.py")):
        name = "linktools.ai." + ".".join(path.relative_to(root).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[:-9]
        importlib.import_module(name)
