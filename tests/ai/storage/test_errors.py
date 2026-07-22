#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/test_errors.py"""

import pytest

from linktools.ai.errors import (
    LinktoolsAIError,
    AssetError,
    AssetNotFoundError,
    AssetConflictError,
    AssetPreconditionFailedError,
    AssetReadOnlyError,
    AssetUnsupportedError,
    InvalidAssetPathError,
    StorageError,
    StorageCapabilityError,
    IdempotencyConflictError,
)


@pytest.mark.parametrize(
    "exc_type,base_type",
    [
        (AssetError, LinktoolsAIError),
        (AssetNotFoundError, AssetError),
        (AssetConflictError, AssetError),
        (AssetPreconditionFailedError, AssetError),
        (AssetReadOnlyError, AssetError),
        (AssetUnsupportedError, AssetError),
        (InvalidAssetPathError, AssetError),
        (StorageError, LinktoolsAIError),
        (StorageCapabilityError, StorageError),
        (IdempotencyConflictError, LinktoolsAIError),
    ],
)
def test_error_hierarchy(exc_type, base_type):
    assert issubclass(exc_type, base_type)
    assert issubclass(exc_type, Exception)
    instance = exc_type("boom")
    assert str(instance) == "boom"
