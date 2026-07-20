#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP connection fingerprint (WP-12 §17.3): canonical-JSON payload, so the
governance-relevant config hashes without ambiguous delimiter joins. None vs
empty allowlist, spaced-vs-split command parts, and a rotated secret must each
produce a different fingerprint; the secret plaintext never appears in it."""

from linktools.ai.mcp.client import _config_fingerprint
from linktools.ai.mcp.spec import MCPServerSpec


def _stdio(**env) -> MCPServerSpec:
    return MCPServerSpec(
        id="s", name="s", transport="stdio", command=("python",), env=env
    )


def test_none_allowlist_differs_from_empty_allowlist():
    none_fp = _config_fingerprint(_stdio())
    empty_fp = _config_fingerprint(
        MCPServerSpec(
            id="s",
            name="s",
            transport="stdio",
            command=("python",),
            enabled_tools=(),
        )
    )
    assert none_fp != empty_fp


def test_spaced_vs_split_command_parts_differ():
    a = _config_fingerprint(
        MCPServerSpec(id="s", name="s", transport="stdio", command=("a b", "c"))
    )
    b = _config_fingerprint(
        MCPServerSpec(id="s", name="s", transport="stdio", command=("a", "b c"))
    )
    assert a != b


def test_url_change_changes_fingerprint():
    u1 = _config_fingerprint(
        MCPServerSpec(id="s", name="s", transport="sse", url="http://x/v1")
    )
    u2 = _config_fingerprint(
        MCPServerSpec(id="s", name="s", transport="sse", url="http://y/v1")
    )
    assert u1 != u2


def test_rotated_secret_changes_fingerprint_without_plaintext():
    fp1 = _config_fingerprint(_stdio(API_KEY="secret-one"))
    fp2 = _config_fingerprint(_stdio(API_KEY="secret-two"))
    assert fp1 != fp2
    # The plaintext never enters the fingerprint.
    assert "secret-one" not in fp1
    assert "secret-two" not in fp2


def test_same_config_is_stable():
    a = _config_fingerprint(_stdio(API_KEY="k"))
    b = _config_fingerprint(_stdio(API_KEY="k"))
    assert a == b
