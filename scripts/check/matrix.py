#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared CI execution plan and pytest file-family partition."""

import json
import sys
import typing
from pathlib import Path

PYTHON_VERSIONS = ("3.10", "3.x")
AI_GROUPS = ("evaluation", "runtime")


def ai_test_group(path: Path) -> str:
    if path.parts[:2] not in (("tests", "ai"), ("linktools-ai", "tests")):
        return "all"
    return "evaluation" if path.name.startswith("test_evaluation") or "capture" in path.name else "runtime"


def package_checks(packages: "typing.Iterable[str]") -> "typing.List[typing.Dict[str, str]]":
    checks = []
    for package in packages:
        groups = AI_GROUPS if package == "linktools-ai" else ("all",)
        for group in groups:
            checks.append({
                "package": package,
                "group": group,
                # Core checks exercise every installed package's command entry points.
                "install": "" if package == "linktools" else package,
                "name": "%s checks%s" % (
                    package, " (%s)" % group if group != "all" else "",
                ),
                "pytest-args": "-n 4 --dist=loadfile --capture=fd -rs --ai-group=%s%s" % (
                    group, " --durations=50" if group != "all" else "",
                ),
            })
    return checks


if __name__ == "__main__":
    print(json.dumps({
        "python-versions": PYTHON_VERSIONS,
        "checks": package_checks(json.loads(sys.argv[1])),
    }))
