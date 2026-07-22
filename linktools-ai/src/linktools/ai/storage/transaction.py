#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-store transaction managers.

The :class:`StorageTransactionManager` Protocol lives in
:mod:`linktools.ai.storage.protocols`; this module holds the in-repo reference
managers. ``storage.transaction()`` is the single public cross-store-UoW
entry: it yields a :class:`StorageUnitOfWork` whose stores share one atomic
scope (a SqlAlchemy AsyncSession + its transaction). The manager object itself
is an internal Storage dependency (``_transaction_manager``), not a public
surface; callers go through ``storage.transaction()``.

A backend whose stores are independent (FilesystemStorage -- separate file
backends, no shared transaction provider) CANNOT offer a cross-store UoW; its
``features.transaction_scope`` is ``TransactionScope.NONE`` (each store is
independently durable, but there is no general cross-store transaction), and its
manager is a :class:`NoCrossStoreTransactions` whose ``transaction()`` raises
:class:`StorageTransactionNotSupportedError` at the call. That is the honest
declaration: the capability is not faked -- it fails explicitly. A backend with
a shared transaction provider (SqlAlchemyStorage -- one AsyncSession across
stores) sets ``features.transaction_scope = TransactionScope.DATABASE`` and
supplies a real manager (see :mod:`linktools.ai.storage.sqlalchemy.facade`).

Raising at the call (not inside ``__aenter__``) means
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
    ``features.transaction_scope = TransactionScope.NONE`` (each store independently
    durable, but no general cross-store UoW)."""

    def __init__(self, backend_name: str = "FilesystemStorage") -> None:
        self._backend_name = backend_name

    def transaction(self) -> "StorageUnitOfWork":
        raise StorageTransactionNotSupportedError(
            f"{self._backend_name} does not provide a cross-store UnitOfWork "
            "(its stores are independent; features.transaction_scope is NONE -- no "
            "general cross-store transaction). Use a Storage with "
            "TransactionScope.DATABASE for cross-store atomic writes."
        )


__all__: "list[str]" = ["NoCrossStoreTransactions"]
