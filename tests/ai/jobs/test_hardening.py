#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Payload validation + error redaction tests."""

import pytest

from linktools.ai.governance.security.redact import redact_text
from linktools.ai.jobs.validation import (
    MAX_OUTPUT_PAYLOAD_BYTES,
    validate_child_tasks,
    validate_commands,
    validate_handler_name,
    validate_metadata,
    validate_output_payload,
    validate_task_key,
)


def test_validate_handler_name() -> None:
    validate_handler_name("echo")
    with pytest.raises(ValueError):
        validate_handler_name("")
    with pytest.raises(ValueError):
        validate_handler_name("x" * 256)


def test_validate_task_key() -> None:
    validate_task_key("k")
    with pytest.raises(ValueError):
        validate_task_key("x" * 256)


def test_validate_metadata_size() -> None:
    validate_metadata({"k": "v"})
    with pytest.raises(ValueError, match="exceeds"):
        validate_metadata({"big": "x" * (256 * 1024)})


def test_validate_metadata_depth() -> None:
    deep = {}
    d = deep
    for _ in range(15):
        d["nested"] = {}
        d = d["nested"]
    with pytest.raises(ValueError, match="nesting"):
        validate_metadata(deep)


def test_validate_commands_bounds() -> None:
    validate_commands(0)
    validate_commands(100)
    with pytest.raises(ValueError, match="commands"):
        validate_commands(101)


def test_validate_child_tasks_bounds() -> None:
    validate_child_tasks(100)
    with pytest.raises(ValueError, match="child tasks"):
        validate_child_tasks(101)


def test_validate_output_payload_bounds() -> None:
    validate_output_payload(MAX_OUTPUT_PAYLOAD_BYTES)
    with pytest.raises(ValueError, match="exceeds"):
        validate_output_payload(MAX_OUTPUT_PAYLOAD_BYTES + 1)


def test_redact_text_masks_secrets_but_keeps_plain_text() -> None:
    redacted = redact_text("Authorization: Bearer abc.def.ghi")
    assert "REDACTED" in redacted
    assert "abc.def.ghi" not in redacted
    assert "REDACTED" in redact_text("key sk-live-9xYzAbCd1234")
    # Plain text without secret markers is unchanged.
    assert redact_text("nothing sensitive here") == "nothing sensitive here"


def test_redact_text_masks_db_conn_string_inline_password_and_private_key() -> None:
    # DB connection string with embedded credentials.
    conn = "connect to postgresql://app:s3cr3t@db:5432/prod failed"
    assert "s3cr3t" not in redact_text(conn)
    # Inline credential assignment.
    assert "hunter2" not in redact_text("login with password=hunter2 ok")
    assert "abc123" not in redact_text("config api_key=abc123 loaded")
    # PEM private key block.
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIBsecretkeybytes\n-----END RSA PRIVATE KEY-----"
    out = redact_text(pem)
    assert "MIIBsecretkeybytes" not in out
    assert "PRIVATE KEY" not in out
