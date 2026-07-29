#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/test_errors.py"""

import pytest

from linktools.ai.errors import (
    LinktoolsAIError,
    StorageError,
    StorageFeatureSupportError,
    IdempotencyConflictError,
)


@pytest.mark.parametrize(
    "exc_type,base_type",
    [
        (StorageError, LinktoolsAIError),
        (StorageFeatureSupportError, StorageError),
        (IdempotencyConflictError, LinktoolsAIError),
    ],
)
def test_error_hierarchy(exc_type, base_type):
    assert issubclass(exc_type, base_type)
    assert issubclass(exc_type, Exception)
    instance = exc_type("boom")
    assert str(instance) == "boom"
