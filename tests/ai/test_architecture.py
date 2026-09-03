#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Import safety and optional dependency boundary checks."""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def test_modules_are_importable() -> None:
    root = Path("linktools-ai/src/linktools/ai")
    for path in sorted(root.rglob("*.py")):
        name = "linktools.ai." + ".".join(
            path.relative_to(root).with_suffix("").parts
        )
        name = name.removesuffix(".__init__")
        importlib.import_module(name)


def test_optional_dependency_isolation() -> None:
    environment = dict(os.environ)
    source_root = Path(__file__).parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(source_root / "linktools-ai/src"), str(source_root / "linktools/src"))
    )
    blocker = """
import importlib
import sys
from importlib.abc import MetaPathFinder

class Blocker(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in TARGETS):
            raise ModuleNotFoundError(fullname, name=fullname.split('.')[0])
        return None

TARGETS = ('sqlalchemy', 'acp')
for target in TARGETS:
    sys.meta_path.insert(0, Blocker())
importlib.import_module('linktools.ai.asset')
for name in TARGETS:
    assert name not in sys.modules, name
"""
    subprocess.run([sys.executable, "-c", blocker], env=environment, check=True)
