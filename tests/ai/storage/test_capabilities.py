#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/storage/test_capabilities.py"""

from linktools.ai.storage.capabilities import (
    StorageCapabilities,
    FILE_STORAGE_CAPABILITIES,
    SQLALCHEMY_STORAGE_CAPABILITIES,
)


def test_file_storage_capabilities_match_spec():
    assert FILE_STORAGE_CAPABILITIES == StorageCapabilities(
        cross_store_transactions=False,
        optimistic_concurrency=True,
        append_only_events=True,
        distributed_coordination=False,
        full_text_search=False,
        semantic_search=False,
        multi_process_swarm=False,
    )


def test_sqlalchemy_storage_capabilities_match_spec():
    assert SQLALCHEMY_STORAGE_CAPABILITIES == StorageCapabilities(
        cross_store_transactions=True,
        optimistic_concurrency=True,
        append_only_events=True,
        distributed_coordination=True,
        full_text_search=True,
        semantic_search=False,
        multi_process_swarm=True,
    )


def test_is_frozen():
    import pytest

    with pytest.raises(Exception):
        FILE_STORAGE_CAPABILITIES.cross_store_transactions = True
