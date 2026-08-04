#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bounded cancellation helpers for ACP-owned asyncio tasks."""

import asyncio
from typing import Any


async def cancel_and_wait(task: "asyncio.Task[Any]", *, timeout: float) -> bool:
    """Cancel a task and observe it, returning false if it ignores cancellation."""
    if not task.done():
        task.cancel()
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task not in done:
        task.add_done_callback(_consume_task_result)
        return False
    _consume_task_result(task)
    return True


async def wait_and_observe(task: "asyncio.Task[Any]", *, timeout: float) -> bool:
    """Observe a shared task for a bounded time without cancelling its owner."""
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task not in done:
        task.add_done_callback(_consume_task_result)
        return False
    _consume_task_result(task)
    return True


def _consume_task_result(task: "asyncio.Task[Any]") -> None:
    try:
        task.result()
    except BaseException:
        pass


__all__ = ["cancel_and_wait", "wait_and_observe"]
