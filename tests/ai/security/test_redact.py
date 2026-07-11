#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument-safety redaction for audit copies (spec §16.3 minimum): secret
masking + size limits on tool arguments persisted to the approval store / log."""

from linktools.ai.security.redact import redact_for_audit


def test_secret_keyed_values_are_masked():
    out = redact_for_audit({
        "command": "rm -rf /",  # not secret -> visible
        "api_key": "sk-supersecret",
        "password": "hunter2",
        "token": "abc",
        "Authorization": "Bearer x",
    })
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
