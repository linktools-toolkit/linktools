#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the real Gradle asset task body against temporary files, without an SDK."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "linktools-mobile" / "agents" / "android" / "tools" / "build.gradle"
HARNESS = Path(__file__).parent / "fixtures" / "apk_manifest_contract.groovy"


def test_apk_manifest_checksum_short_circuit(tmp_path: Path) -> None:
    groovy = shutil.which("groovy")
    if groovy:
        command = [groovy]
    else:
        classpath = os.environ.get("LINKTOOLS_GROOVY_CLASSPATH")
        java = shutil.which("java")
        if not classpath or not java:
            pytest.skip(
                "Groovy or LINKTOOLS_GROOVY_CLASSPATH pointing to Gradle's Groovy jars is required"
            )
        command = [java, "-cp", classpath, "groovy.ui.GroovyMain"]
    result = subprocess.run(
        command + [str(HARNESS), str(BUILD), str(tmp_path)],
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "APK manifest contract passed" in result.stdout
