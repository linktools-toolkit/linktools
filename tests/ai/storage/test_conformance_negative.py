#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""the conformance contract REJECTS a non-conformant
ArtifactBlobStore whose ``open()`` returns a bare async iterator instead of an
AsyncContextManager. The contract fixes the consumption shape as
``async with store.open(...) as chunks``; a backend that returned an async
generator would let ``async for chunk in store.open(...)`` work but break the
asset-cleanup guarantee (no __aexit__ to close the fd/connection). This
test pins that the contract SHAPE catches such an impl -- the headline
negative-validation gap the cross-reference found."""

import hashlib
from typing import AsyncIterator

import pytest

from linktools.ai.storage.protocols import BlobInfo


async def _aiter(chunks):
    for c in chunks:
        yield c


class _AsyncGenOpenBlobStore:
    """A DELIBERATELY non-conformant ArtifactBlobStore: ``open()`` returns a
    bare async generator (AsyncIterator[bytes]) instead of an
    AsyncContextManager. This is exactly the -forbidden shape -- it works
    with ``async for`` but provides no ``__aexit__`` asset cleanup."""

    def __init__(self) -> None:
        self._blobs: "dict[str, bytes]" = {}

    async def put_if_absent(
        self, *, digest: str, source: AsyncIterator[bytes], size: "int | None"
    ) -> BlobInfo:
        acc: "list[bytes]" = []
        async for c in source:
            acc.append(c)
        data = b"".join(acc)
        self._blobs[digest] = data
        return BlobInfo(digest=digest, size=len(data), content_type=None)

    async def open(self, *, digest: str) -> AsyncIterator[bytes]:
        # NON-CONFORMANT: bare async generator, NOT @asynccontextmanager. An
        # async-generator object has no __aenter__/__aexit__.
        data = self._blobs.get(digest, b"")
        yield data

    async def stat(self, *, digest: str) -> "BlobInfo | None":
        return None

    async def delete(self, *, digest: str) -> None:
        pass


def test_contract_rejects_async_gen_open_that_is_not_a_context_manager():
    """a blob store whose open() returns a bare async iterator
    (no __aenter__/__aexit__) CANNOT be consumed via the mandated
    ``async with store.open(...)`` shape -- ``async with`` on an async
    generator raises TypeError. The contract shape thus rejects the
    non-conformant impl; a backend cannot slip through by returning an async
    generator instead of an async context manager."""
    import asyncio

    bad = _AsyncGenOpenBlobStore()
    digest = hashlib.sha256(b"x").hexdigest()

    async def _run():
        await bad.put_if_absent(digest=digest, source=_aiter([b"x"]), size=1)
        with pytest.raises(TypeError):
            async with bad.open(digest=digest) as chunks:
                async for _ in chunks:
                    pass

    asyncio.run(_run())


def test_conformant_backend_open_is_usable_as_async_with(tmp_path):
    """The mirror: a CONFORMANT backend's open() (decorated with
    @asynccontextmanager) IS consumable via the ``async with`` shape, so
    the rejection above is specifically about the non-conformant shape, not the
    contract being unusable."""
    import asyncio

    from linktools.ai.storage.filesystem.artifact import FilesystemArtifactBlobStore

    store = FilesystemArtifactBlobStore(blobs_root=tmp_path / "blobs")
    digest = hashlib.sha256(b"roundtrip").hexdigest()

    async def _run():
        await store.put_if_absent(
            digest=digest, source=_aiter([b"roundtrip"]), size=9
        )
        collected: "list[bytes]" = []
        async with store.open(digest=digest) as chunks:
            async for chunk in chunks:
                collected.append(chunk)
        assert b"".join(collected) == b"roundtrip"

    asyncio.run(_run())
