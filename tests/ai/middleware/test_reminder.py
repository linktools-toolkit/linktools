#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/middleware/test_reminder.py"""

from types import SimpleNamespace

import pytest

from linktools.ai.middleware.reminder import ReminderMiddleware


@pytest.mark.asyncio
async def test_reminder_not_added_below_threshold():
    middleware = ReminderMiddleware(
        max_messages=10, threshold_ratio=0.7, reminder_text="please wrap up"
    )
    request = SimpleNamespace(messages=["m"] * 5)
    result = await middleware.before_model(context=None, request=request)
    assert result.messages == ["m"] * 5


@pytest.mark.asyncio
async def test_reminder_added_at_or_above_threshold():
    middleware = ReminderMiddleware(
        max_messages=10, threshold_ratio=0.7, reminder_text="please wrap up"
    )
    request = SimpleNamespace(messages=["m"] * 7)
    result = await middleware.before_model(context=None, request=request)
    assert result.messages[-1] == "please wrap up"
    assert len(result.messages) == 8


@pytest.mark.asyncio
async def test_reminder_only_fires_once():
    middleware = ReminderMiddleware(
        max_messages=10, threshold_ratio=0.7, reminder_text="please wrap up"
    )
    request1 = SimpleNamespace(messages=["m"] * 7)
    result1 = await middleware.before_model(context=None, request=request1)
    assert len(result1.messages) == 8

    request2 = SimpleNamespace(messages=["m"] * 9)
    result2 = await middleware.before_model(context=None, request=request2)
    assert result2.messages == ["m"] * 9
