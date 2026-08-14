#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External SQL route contract checks."""

from linktools.ai.runtime.state import RuntimeStateRoute


def test_sqlite_route_is_durable() -> None:
    route = RuntimeStateRoute.sqlite("runtime.db")
    assert route.retention.value == "durable"
