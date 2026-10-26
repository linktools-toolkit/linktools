#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native MySQL and PostgreSQL partition transaction parity."""

import pytest

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceStorage

from .test_state_group_partitions import _partition_contract


@pytest.mark.asyncio
async def test_state_group_partition_native_server_atomicity(server_metrics) -> None:
    _, engine, _ = server_metrics
    await provision_database(engine)
    async with Persistence.open("partitions", storage=PersistenceStorage.sql(engine)) as persistence:
        await _partition_contract(persistence.execution.executions.state_store)
