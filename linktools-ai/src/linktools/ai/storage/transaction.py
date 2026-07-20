#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-store transaction managers.

The :class:`StorageTransactionManager` Protocol lives in
:mod:`linktools.ai.storage.protocols`; this module holds the in-repo reference
managers. ``Storage.transactions`` is the canonical cross-store-UoW surface:
``storage.transactions.transaction()`` yields a :class:`StorageUnitOfWork` whose
stores share one atomic scope (a SqlAlchemy AsyncSession + its transaction).

A backend whose stores are independent (FilesystemStorage -- separate file
backends, no shared transaction provider) CANNOT offer a cross-store UoW; its
``features.transactions`` is ``TransactionScope.PROCESS_LOCAL`` (single-store
durability only), and its ``transactions`` field is a
:class:`NoCrossStoreTransactions` manager whose ``transaction()`` raises
:class:`StorageTransactionNotSupportedError` at the call. That is the honest
declaration: the capability is not faked -- it fails explicitly. A backend with
a shared transaction provider (SqlAlchemyStorage -- one AsyncSession across
stores) sets ``features.transactions = TransactionScope.DATABASE`` and supplies
a real manager (see :mod:`linktools.ai.storage.sqlalchemy.facade`).

The legacy ``Storage.transaction()`` method is retained as a thin delegator to
``self.transactions.transaction()`` so existing call sites (the run commit
coordinators, tests) keep working; the field is the canonical surface for new
code. Raising at the call (not inside ``__aenter__``) means
``async with storage.transaction()`` fails at the call with the concrete error,
not with a confusing coroutine TypeError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import StorageTransactionNotSupportedError

if TYPE_CHECKING:
    from .protocols import StorageUnitOfWork


class NoCrossStoreTransactions:
    """A StorageTransactionManager for backends that cannot offer a cross-store
    UoW (e.g. FilesystemStorage, whose stores are independent files with no
    shared transaction provider). ``transaction()`` raises at the call rather
    than yielding a fake atomic scope. This matches
    ``features.transactions = TransactionScope.PROCESS_LOCAL`` (single-store
    durability only -- not a cross-store UoW)."""

    def __init__(self, backend_name: str = "FilesystemStorage") -> None:
        self._backend_name = backend_name

    def transaction(self) -> "StorageUnitOfWork":
        raise StorageTransactionNotSupportedError(
            f"{self._backend_name} does not provide a cross-store UnitOfWork "
            "(its stores are independent; features.transactions is PROCESS_LOCAL, "
            "meaning single-store durability only). Use a Storage with "
            "TransactionScope.DATABASE for cross-store atomic writes."
        )


__all__: "list[str]" = ["NoCrossStoreTransactions"]
