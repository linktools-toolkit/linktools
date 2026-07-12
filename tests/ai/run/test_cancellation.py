#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CancellationToken behavior. The token is the
cooperative-cancellation signal checked at execution points (before/after the
model call). These tests cover the three surface methods -- ``cancel``,
``is_cancelled``, ``raise_if_cancelled`` -- and the idempotency contract."""

import asyncio

import pytest

from linktools.ai.run.cancellation import CancellationToken


def test_token_starts_not_cancelled():
    token = CancellationToken()
    assert token.is_cancelled() is False


@pytest.mark.asyncio
async def test_raise_if_cancelled_is_noop_when_not_set():
    """Before ``cancel()`` is called, ``raise_if_cancelled()`` must be a
    no-op -- the runner awaits it unconditionally at every execution point,
    so a spurious raise would abort a healthy run."""
    token = CancellationToken()
    await token.raise_if_cancelled()  # must not raise


def test_cancel_marks_token_as_cancelled():
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled() is True


@pytest.mark.asyncio
async def test_raise_if_cancelled_raises_after_cancel():
    """After ``cancel()``, ``raise_if_cancelled()`` raises
    ``asyncio.CancelledError`` -- this is how the token propagates
    cancellation into the runner's lifecycle (the outer ``except
    CancelledError`` handler then transitions CANCELLING -> CANCELLED)."""
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await token.raise_if_cancelled()


@pytest.mark.asyncio
async def test_cancel_is_idempotent():
    """A second ``cancel()`` is a no-op (Event.set is idempotent). The token
    stays cancelled; ``raise_if_cancelled()`` keeps raising."""
    token = CancellationToken()
    token.cancel()
    token.cancel()  # second call must not raise
    assert token.is_cancelled() is True
    with pytest.raises(asyncio.CancelledError):
        await token.raise_if_cancelled()


@pytest.mark.asyncio
async def test_raise_if_cancelled_raises_every_call_after_cancel():
    """The token does not "consume" the cancellation -- every subsequent
    ``raise_if_cancelled()`` raises again. The runner relies on this so a
    second check after the model call still aborts even if the first check
    (before the model call) was the one that signalled cancel."""
    token = CancellationToken()
    token.cancel()
    with pytest.raises(asyncio.CancelledError):
        await token.raise_if_cancelled()
    with pytest.raises(asyncio.CancelledError):
        await token.raise_if_cancelled()
