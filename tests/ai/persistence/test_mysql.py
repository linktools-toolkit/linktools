#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External SQL routes reject unsupported in-memory ownership."""

import pytest
from linktools.ai.storage import StorageRoute


def test_sqlite_memory_route_is_not_accepted_as_external_sql() -> None:
    with pytest.raises(ValueError):
        StorageRoute.sqlite(":memory:")
