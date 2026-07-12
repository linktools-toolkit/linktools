#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prompt window policies (contract): noop passthrough, recent-N trimming, and
the reserved token-budget slot."""

from dataclasses import dataclass

import pytest

from linktools.ai.prompt import (
    NoopWindowPolicy,
    RecentWindowPolicy,
    SessionWindowPolicy,
    TokenBudgetWindowPolicy,
)


@dataclass
class _Msg:
    sequence: int
    role: str = "user"
    content: str = "x"


def _policy():
    class _Any:
        max_tokens = None

    return _Any()


@pytest.mark.asyncio
async def test_noop_returns_all():
    msgs = [_Msg(1), _Msg(2), _Msg(3)]
    out = await NoopWindowPolicy().select_messages(msgs, _policy())
    assert [m.sequence for m in out] == [1, 2, 3]


@pytest.mark.asyncio
async def test_recent_keeps_last_n():
    msgs = [_Msg(i) for i in range(10)]
    out = await RecentWindowPolicy(max_messages=3).select_messages(msgs, _policy())
    assert [m.sequence for m in out] == [7, 8, 9]


@pytest.mark.asyncio
async def test_recent_returns_all_when_under_limit():
    msgs = [_Msg(1), _Msg(2)]
    out = await RecentWindowPolicy(max_messages=5).select_messages(msgs, _policy())
    assert len(out) == 2


def test_recent_rejects_non_positive():
    with pytest.raises(ValueError):
        RecentWindowPolicy(max_messages=0)


def test_token_budget_reserved():
    with pytest.raises(NotImplementedError):
        TokenBudgetWindowPolicy()


def test_policies_satisfy_protocol():
    assert isinstance(NoopWindowPolicy(), SessionWindowPolicy)
    assert isinstance(RecentWindowPolicy(), SessionWindowPolicy)
