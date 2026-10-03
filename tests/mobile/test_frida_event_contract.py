#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise emitted hook events without loading Frida or touching a device."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
FRIDA = ROOT / "linktools-mobile" / "agents" / "frida"
ASSETS = ROOT / "linktools-mobile" / "src" / "linktools" / "assets"
HARNESS = Path(__file__).parent / "fixtures" / "frida_event_contract.js"


def _check_events(asset: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the isolated Frida event tests")
    result = subprocess.run(
        [node, str(HARNESS), str(asset)],
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert json.loads(result.stdout) == {"cases": 336}


def test_committed_frida_event_contract() -> None:
    _check_events(ASSETS / "frida.js")


def test_minified_frida_event_contract(tmp_path: Path) -> None:
    minifier = FRIDA / "node_modules" / ".bin" / "uglifyjs"
    if not minifier.is_file():
        pytest.skip(
            "Install the locked Frida npm dependencies to generate the minified asset"
        )
    asset = tmp_path / "frida.min.js"
    subprocess.run(
        [str(minifier), str(ASSETS / "frida.js"), "--mangle", "--output", str(asset)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_events(asset)


def test_frida_source_matches_committed_asset(tmp_path: Path) -> None:
    compiler = FRIDA / "node_modules" / ".bin" / "frida-compile"
    if not compiler.is_file():
        pytest.skip(
            "Install the locked Frida npm dependencies to compile the TypeScript source"
        )
    output = tmp_path / "frida.js"
    subprocess.run(
        [str(compiler), str(FRIDA / "index.ts"), "-o", str(output), "-B", "iife"],
        cwd=str(FRIDA),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert output.read_bytes() == (ASSETS / "frida.js").read_bytes()
    _check_events(output)
