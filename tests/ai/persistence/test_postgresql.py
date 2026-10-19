#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External SQL route contract checks."""

from linktools.ai.storage import StorageRoute


def test_sqlite_route_is_durable() -> None:
    route = StorageRoute.sqlite("runtime.db")
    assert route.retention.value == "durable"
