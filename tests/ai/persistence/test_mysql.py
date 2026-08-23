"""External SQL routes reject unsupported in-memory ownership."""

import pytest
from linktools.ai.runtime.state import RuntimeStateRoute


def test_sqlite_memory_route_is_not_accepted_as_external_sql() -> None:
    with pytest.raises(ValueError):
        RuntimeStateRoute.sqlite(":memory:")
