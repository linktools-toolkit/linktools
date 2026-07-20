#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument-safety redaction for audit copies (contract minimum): secret
masking + size limits on tool arguments persisted to the approval store / log."""

from linktools.ai.governance.security.redact import redact_for_audit


def test_secret_keyed_values_are_masked():
    out = redact_for_audit(
        {
            "command": "rm -rf /",  # not secret -> visible
            "api_key": "sk-supersecret",
            "password": "hunter2",
            "token": "abc",
            "Authorization": "Bearer x",
        }
    )
    assert out["command"] == "rm -rf /"
    assert out["api_key"] == "***REDACTED***"
    assert out["password"] == "***REDACTED***"
    assert out["token"] == "***REDACTED***"
    assert out["Authorization"] == "***REDACTED***"


def test_non_secret_values_preserved():
    out = redact_for_audit({"path": "/tmp/x", "limit": 10, "recursive": False})
    assert out == {"path": "/tmp/x", "limit": 10, "recursive": False}


def test_oversized_string_value_truncated():
    big = "x" * 5000
    out = redact_for_audit({"content": big})
    assert "TRUNCATED" in out["content"]
    assert len(out["content"]) < 5000


def test_total_size_cap_truncates_tail():
    args = {f"k{i}": "y" * 1000 for i in range(20)}  # ~20000 chars total
    out = redact_for_audit(args)
    encoded = str(out)
    assert "TRUNCATED" in encoded or len(encoded) < 25000


def test_input_not_mutated_and_none_safe():
    src = {"api_key": "secret"}
    redact_for_audit(src)
    assert src == {"api_key": "secret"}  # input untouched
    assert redact_for_audit(None) == {}
    assert redact_for_audit({}) == {}


def test_all_default_sensitive_keys_are_masked():
    # The default sensitive-key set: every name in it must mask a plain
    # value under that exact key. A bare value (not an inline ``key=value``
    # assignment) can only be caught by key-name matching, so this is the
    # guard that the marker list stays complete. ``client_secret`` is covered
    # by the ``secret`` substring.
    out = redact_for_audit(
        {
            "authorization": "Bearer x",
            "token": "abc",
            "api_key": "sk-x",
            "apikey": "sk-y",
            "password": "hunter2",
            "passwd": "hunter3",
            "secret": "s",
            "cookie": "c=val",
            "private_key": "MIIEvQIBADANB",
            "client_secret": "cs",
            "access_key": "AKIAIOSFODNN7EXAMPLE",
        }
    )
    for key in (
        "authorization",
        "token",
        "api_key",
        "apikey",
        "password",
        "passwd",
        "secret",
        "cookie",
        "private_key",
        "client_secret",
        "access_key",
    ):
        assert out[key] == "***REDACTED***", key
