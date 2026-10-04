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


def pytest_collection_modifyitems(config: pytest.Config, items: "typing.List[pytest.Item]") -> None:
    tier = config.getoption("--test-tier")
    selected = []
    deselected = []
    for item in items:
        excluded = (
            tier != "all" and item.get_closest_marker("manual") is not None
        ) or (
            tier == "daily" and item.get_closest_marker("merge") is not None
        )
        (deselected if excluded else selected).append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
