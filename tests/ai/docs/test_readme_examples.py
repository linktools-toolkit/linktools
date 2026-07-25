#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runs the README example modules under ``examples/`` so the documented code
cannot silently drift. The examples use a canned pydantic-ai ``FunctionModel``
and a tmp-path storage root, so they run fully offline."""

import asyncio
import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_EXAMPLES = _REPO / "examples"


def _load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_minimal_runtime_example_runs_offline(tmp_path):
    minimal = _load_example("minimal_runtime")
    output = asyncio.run(minimal.run(tmp_path))
    assert "hello from linktools-ai" in str(output)


def test_sandbox_runtime_example_runs_offline(tmp_path):
    sandbox_example = _load_example("sandbox_runtime")
    workdir = tmp_path / "work"
    workdir.mkdir()
    output = asyncio.run(sandbox_example.run(tmp_path, workdir))
    assert output is not None


def test_examples_use_only_public_imports():
    # No private (underscore-prefixed) linktools.ai imports in either example.
    for name in ("minimal_runtime", "sandbox_runtime"):
        path = _EXAMPLES / f"{name}.py"
        text = path.read_text(encoding="utf-8")
        assert "from linktools.ai._" not in text, f"{name} imports a private module"
        assert "import linktools.ai._" not in text, f"{name} imports a private module"
