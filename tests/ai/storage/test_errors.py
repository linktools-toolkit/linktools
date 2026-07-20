#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/test_errors.py"""

import pytest

from linktools.ai.errors import (
    LinktoolsAIError,
    ResourceError,
    ResourceNotFoundError,
    ResourceConflictError,
    ResourcePreconditionFailedError,
    ResourceReadOnlyError,
    ResourceUnsupportedError,
    InvalidAssetPathError,
    StorageError,
    StorageCapabilityError,
    IdempotencyConflictError,
)


@pytest.mark.parametrize(
    "exc_type,base_type",
    [
        (ResourceError, LinktoolsAIError),
        (ResourceNotFoundError, ResourceError),
        (ResourceConflictError, ResourceError),
        (ResourcePreconditionFailedError, ResourceError),
        (ResourceReadOnlyError, ResourceError),
        (ResourceUnsupportedError, ResourceError),
        (InvalidAssetPathError, ResourceError),
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
