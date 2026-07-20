#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Registry idempotency_key_field strict type.

The key field must be a string (coercion via ``str()`` is forbidden): non-string
values are rejected, and whitespace-only values are rejected after stripping.
"""

import asyncio

import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.catalog.parsing import SpecLoader
from linktools.ai.tool.catalog import ToolCatalog


def _load_tool(tmp_path, body: str):
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "t.yaml").write_text(body, encoding="utf-8")
    registry = ToolCatalog.from_specloader(SpecLoader.from_filesystem(tools))

    async def _get():
        return await registry.get("t")

    return asyncio.run(_get())


def test_non_string_key_field_is_rejected(tmp_path):
    body = (
        "idempotent: true\n"
        "idempotency_strategy: business_key\n"
        "idempotency_key_field: 123\n"
    )
    with pytest.raises(InvalidSpecError, match="must be a string"):
        _load_tool(tmp_path, body)


def test_whitespace_key_field_is_rejected(tmp_path):
    body = (
        "idempotent: true\n"
        "idempotency_strategy: business_key\n"
        'idempotency_key_field: "   "\n'
    )
    with pytest.raises(InvalidSpecError, match="non-empty"):
        _load_tool(tmp_path, body)


def test_valid_string_key_field_is_accepted_and_stripped(tmp_path):
    body = (
        "idempotent: true\n"
        "idempotency_strategy: business_key\n"
        'idempotency_key_field: "  external_id  "\n'
    )
    spec = _load_tool(tmp_path, body)
    assert spec.idempotency_key_field == "external_id"
