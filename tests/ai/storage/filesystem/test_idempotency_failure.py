#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""regression: FilesystemIdempotencyStore.fail() must take the fenced claim token
(not the old (scope, key, error) signature). A shadowed old signature would
make GovernedToolInvoker's fail(claim, error) hit the wrong method (TypeError) and
leave the FAILED transition unwritten."""

import asyncio

from linktools.ai.storage.filesystem.idempotency import FilesystemIdempotencyStore
from linktools.ai.tool.idempotency import IdempotencyStatus


def test_file_idempotency_failure_uses_claim_token(tmp_path):
    store = FilesystemIdempotencyStore(root=tmp_path)

    async def _run():
        result = await store.claim(
            scope="s", key="k", request_hash="h", owner_id="owner-1"
        )
        assert result.disposition.value == "acquired"
        await store.fail(result.claim, "redacted failure")
        record = await store.get("s", "k")
        assert record.status is IdempotencyStatus.FAILED
        assert record.error == "redacted failure"

    asyncio.run(_run())


def test_file_idempotency_fail_signature_is_single_claim_token(tmp_path):
    """The public fail() accepts exactly (claim, error) -- the old
    (scope, key, error) signature is gone, so a 3-positional-arg call fails
    fast rather than silently hitting a shadow."""
    import inspect

    from linktools.ai.storage.filesystem.idempotency import FilesystemIdempotencyStore

    sig = inspect.signature(FilesystemIdempotencyStore.fail)
    assert list(sig.parameters) == ["self", "claim", "error"], sig.parameters
