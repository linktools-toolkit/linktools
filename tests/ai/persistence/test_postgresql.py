#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External SQL route contract checks."""

from linktools.ai.runtime.state import RuntimeStorageRoute


def test_sqlite_route_is_durable() -> None:
    route = RuntimeStorageRoute.sqlite("runtime.db")
    assert route.retention.value == "durable"
