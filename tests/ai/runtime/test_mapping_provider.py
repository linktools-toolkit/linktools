#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MappingProvider: a generic mapping-backed spec provider."""

import asyncio

import pytest

from linktools.ai._runtime.dependencies import MappingProvider


def test_mapping_provider_list_ids():
    provider = MappingProvider({"a": 1, "b": 2})
    assert asyncio.run(provider.list_ids()) == ("a", "b")


def test_mapping_provider_get():
    provider = MappingProvider({"a": 1, "b": 2})
    assert asyncio.run(provider.get("a")) == 1


def test_mapping_provider_get_missing_raises():
    provider = MappingProvider({"a": 1})
    with pytest.raises(KeyError):
        asyncio.run(provider.get("missing"))


def test_mapping_provider_empty():
    provider = MappingProvider({})
    assert asyncio.run(provider.list_ids()) == ()
