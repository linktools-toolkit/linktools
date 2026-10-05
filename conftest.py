#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit test execution tiers shared by pytest and manage.py."""

import typing

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
        "--ai-group", choices=("all", "evaluation", "runtime"), default="all",
        help="Partition AI tests by file family without changing tier coverage",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: "typing.List[pytest.Item]") -> None:
    tier = config.getoption("--test-tier")
    group = config.getoption("--ai-group")
    selected = []
    deselected = []
    for item in items:
        excluded = (
            tier != "all" and item.get_closest_marker("manual") is not None
        ) or (
            tier == "daily" and item.get_closest_marker("merge") is not None
        )
        path = item.path.relative_to(config.rootpath)
        if group != "all" and path.parts[:2] == ("tests", "ai"):
            family = "evaluation" if (
                path.name.startswith("test_evaluation") or "capture" in path.name
            ) else "runtime"
            excluded = excluded or family != group
        (deselected if excluded else selected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
