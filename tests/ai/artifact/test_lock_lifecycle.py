import asyncio

import pytest

from linktools.ai.storage.local.locks import KeyedLocks


@pytest.mark.asyncio
async def test_keyed_locks_reap_and_serialize_keys():
    locks = KeyedLocks()
    order: list[str] = []

    async def critical(name: str) -> None:
        async with locks.acquire(("run", "same")):
            order.append(f"{name}-in")
            await asyncio.sleep(0)
            order.append(f"{name}-out")

    await asyncio.gather(critical("a"), critical("b"), critical("c"))
    assert order == ["a-in", "a-out", "b-in", "b-out", "c-in", "c-out"]


@pytest.mark.asyncio
async def test_keyed_locks_allow_distinct_keys_in_parallel():
    locks = KeyedLocks()
    active = 0
    maximum = 0
    counter = asyncio.Lock()

    async def critical(key: str) -> None:
        nonlocal active, maximum
        async with locks.acquire(("run", key)):
            async with counter:
                active += 1
                maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            async with counter:
                active -= 1

    await asyncio.gather(*(critical(str(i)) for i in range(4)))
    assert maximum > 1
