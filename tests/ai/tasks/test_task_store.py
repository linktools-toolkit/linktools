import pytest

from linktools.ai.tasks.persistence.local import LocalTaskBackend

from tests.ai.tasks.contracts import (
    assert_task_store_contract,
    assert_usage_round_trips_through_complete,
)


@pytest.mark.asyncio
async def test_local_task_store_meets_contract():
    store = LocalTaskBackend()
    await assert_task_store_contract(store)


@pytest.mark.asyncio
async def test_local_task_store_round_trips_usage():
    store = LocalTaskBackend()
    await assert_usage_round_trips_through_complete(store)
