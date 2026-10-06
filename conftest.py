#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit test tiers and manifest-owned package groups."""

import json
import os
import typing
from fnmatch import fnmatchcase
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "asyncio_default_fixture_loop_scope", "Default loop scope for async fixtures",
        type="string", default="function",
    )
    parser.addoption(
        "--test-tier", choices=("daily", "merge", "all"), default="merge",
        help="daily: representative regressions; merge: automatic acceptance; all: include manual probes",
    )
    parser.addoption(
        "--test-group", default="all",
        help="Select a package test group declared in linktools.yml via manage.py check",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: "typing.List[pytest.Item]") -> None:
    tier = config.getoption("--test-tier")
    group = config.getoption("--test-group")
    try:
        check = json.loads(os.environ.get("LINKTOOLS_PYTEST", "{}"))
    except ValueError as error:
        raise pytest.UsageError("Invalid package pytest configuration: %s" % error)
    groups = check.get("groups", {})
    if group != "all" and group not in groups:
        raise pytest.UsageError(
            "Unknown test group %r; use manage.py check with one of: %s"
            % (group, ", ".join(("all",) + tuple(groups)))
        )
    roots = tuple(Path(path) for path in check.get("paths", ()))
    fallback = next((name for name, patterns in groups.items() if not patterns), None)
    selected = []
    deselected = []
    for item in items:
        excluded = (
            tier != "all" and item.get_closest_marker("manual") is not None
        ) or (
            tier == "daily" and item.get_closest_marker("merge") is not None
        )
        if groups and any(root == item.path or root in item.path.parents for root in roots):
            matches = [
                name for name, patterns in groups.items()
                if any(fnmatchcase(item.path.name, pattern) for pattern in patterns)
            ]
            if len(matches) > 1:
                raise pytest.UsageError("Ambiguous test groups for %s: %s" % (item.path, ", ".join(matches)))
            family = matches[0] if matches else fallback
            excluded = excluded or (group != "all" and family != group)
        (deselected if excluded else selected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
