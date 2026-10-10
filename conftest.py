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


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(_ManifestGroups(config), "linktools-manifest-groups")


class _ManifestGroups:

    def __init__(self, config: pytest.Config) -> None:
        self.group = config.getoption("--test-group")
        try:
            check = json.loads(os.environ.get("LINKTOOLS_PYTEST", "{}"))
        except ValueError as error:
            raise pytest.UsageError("Invalid package pytest configuration: %s" % error)
        self.groups = check.get("groups", {})
        if self.group != "all" and self.group not in self.groups:
            raise pytest.UsageError(
                "Unknown test group %r; use manage.py check with one of: %s"
                % (self.group, ", ".join(("all",) + tuple(self.groups)))
            )
        self.roots = tuple(Path(path) for path in check.get("paths", ()))
        self.fallback = next((name for name, patterns in self.groups.items() if not patterns), None)

    def _excludes(self, path: Path, validate: bool = True) -> bool:
        if not self.groups or not any(root == path or root in path.parents for root in self.roots):
            return False
        matches = [
            name for name, patterns in self.groups.items()
            if any(fnmatchcase(path.name, pattern) for pattern in patterns)
        ]
        if len(matches) > 1:
            if validate:
                raise pytest.UsageError("Ambiguous test groups for %s: %s" % (path, ", ".join(matches)))
            # Collect conflicts so item validation still fails, without rejecting support files.
            return False
        family = matches[0] if matches else self.fallback
        return self.group != "all" and family != self.group

    def _ignore_collect(self, path: Path) -> "typing.Optional[bool]":
        # Filename groups cannot exclude directories, which may contain other families.
        # Older pytest also needs __init__.py to establish package collectors and setup.
        if (
            self.group != "all" and path.name != "__init__.py"
            and path.is_file() and self._excludes(path, validate=False)
        ):
            return True
        return None

    # pytest 7 added pathlib hook arguments; pytest 9 removed the legacy arguments.
    if int(pytest.__version__.split(".")[0]) >= 7:
        def pytest_ignore_collect(self, collection_path: Path) -> "typing.Optional[bool]":
            return self._ignore_collect(collection_path)
    else:
        def pytest_ignore_collect(self, path: object) -> "typing.Optional[bool]":
            return self._ignore_collect(Path(str(path)))

    def pytest_collection_modifyitems(self, config: pytest.Config, items: "typing.List[pytest.Item]") -> None:
        tier = config.getoption("--test-tier")
        selected = []
        deselected = []
        for item in items:
            path = getattr(item, "path", None)
            if path is None:
                path = Path(str(item.fspath))
            excluded = self._excludes(path) or (
                tier != "all" and item.get_closest_marker("manual") is not None
            ) or (
                tier == "daily" and item.get_closest_marker("merge") is not None
            )
            (deselected if excluded else selected).append(item)
        items[:] = selected
        if deselected:
            config.hook.pytest_deselected(items=deselected)
