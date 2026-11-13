#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Owner validation reuses only live scoped receipts within one flush."""
from types import SimpleNamespace

import pytest

from linktools.ai.errors import AIError
from linktools.ai.persistence import _native_content_index as index


@pytest.mark.asyncio
async def test_flush_batches_overlapping_roots_and_reuses_live_subset_receipts(monkeypatch):
    transaction = object()
    session = SimpleNamespace(info={}, get_nested_transaction=lambda: None,
                              get_transaction=lambda: transaction)
    reads = []
    validations = []

    async def resolve(session, kind, store, roots):
        reads.append((store, tuple(roots)))
        return {root: position for position, root in enumerate(roots)}

    monkeypatch.setattr(index, "resolve_targets", resolve)

    def receipt(session, *, store_key, roots):
        assert store_key == b"a" and tuple(roots) == ("one", "two", "unused")
        validations.append(tuple(roots))
        return (1, 2, 3)

    index.register_publication_targets(session, b"a", ("one", "two", "unused"), receipt)
    validations.clear()

    def root(store, *ids):
        return SimpleNamespace(group_id="group", store_key=store,
                               slices=tuple(SimpleNamespace(first=value) for value in ids))

    pending = {
        "owner1": (root(b"a", "one", "three"),),
        "owner2": (root(b"a", "two", "three", "four"),),
        "other-store": (root(b"b", "one"),),
    }
    session.info[index._PENDING] = pending
    await index.flush_owners(None, session, group_id="group")
    assert validations == [("one", "two", "unused")]
    assert reads == [(b"a", ("four", "three")), (b"b", ("one",))]
    assert not session.info[index._PUBLICATION_HANDOFF][1]

    # Flush-local evidence is not a cache for a later mutation.
    reads.clear()
    session.info[index._PENDING] = pending
    await index.flush_owners(None, session, group_id="group")
    assert reads == [(b"a", ("four", "one", "three", "two")), (b"b", ("one",))]


@pytest.mark.asyncio
@pytest.mark.parametrize("expired_transaction", [False, True])
async def test_flush_rechecks_expired_receipts(monkeypatch, expired_transaction):
    transaction = object()
    session = SimpleNamespace(info={}, get_nested_transaction=lambda: None,
                              get_transaction=lambda: transaction)
    live = True
    validations = []

    def receipt(session, *, store_key, roots):
        validations.append(tuple(roots))
        if not live:
            raise index.PublicationTargetsUnavailable()
        return (1,)

    index.register_publication_targets(session, b"a", ("one",), receipt)
    validations.clear()
    live = False
    if expired_transaction:
        transaction = object()
    session.info[index._PENDING] = {"owner": (SimpleNamespace(
        group_id="group", store_key=b"a", slices=(SimpleNamespace(first="one"),)),)}
    reads = []

    async def resolve(session, kind, store, roots):
        reads.append((store, tuple(roots)))
        raise index.NativeContentUnavailable()

    monkeypatch.setattr(index, "resolve_targets", resolve)
    with pytest.raises(index.NativeContentUnavailable):
        await index.flush_owners(None, session, group_id="group")
    assert reads == [(b"a", ("one",))]
    assert validations == ([] if expired_transaction else [("one",)])
    assert not session.info[index._PUBLICATION_HANDOFF][1]


@pytest.mark.asyncio
async def test_flush_rejects_wrong_group_before_using_receipt():
    session = SimpleNamespace(info={index._PENDING: {"owner": (SimpleNamespace(
        group_id="other", store_key=b"a", slices=(SimpleNamespace(first="one"),)),)}})
    with pytest.raises(AIError):
        await index.flush_owners(None, session, group_id="group")


@pytest.mark.asyncio
async def test_flush_combines_overlapping_receipts_without_reading(monkeypatch):
    transaction = object()
    session = SimpleNamespace(info={}, get_nested_transaction=lambda: None,
                              get_transaction=lambda: transaction)
    validations = []

    def register(identities):
        def receipt(session, *, store_key, roots):
            assert store_key == b"a" and tuple(roots) == identities
            validations.append(identities)
            return tuple(range(len(identities)))
        index.register_publication_targets(session, b"a", identities, receipt)

    register(("one", "two"))
    register(("three", "two"))
    validations.clear()
    session.info[index._PENDING] = {"owner": (SimpleNamespace(
        group_id="group", store_key=b"a", slices=tuple(
            SimpleNamespace(first=value) for value in ("one", "two", "three"))),)}

    async def resolve(*args, **kwargs):
        pytest.fail("All owner roots have live receipt evidence")

    monkeypatch.setattr(index, "resolve_targets", resolve)
    await index.flush_owners(None, session, group_id="group")
    assert validations == [("one", "two"), ("three", "two")]
    assert not session.info[index._PUBLICATION_HANDOFF][1]
