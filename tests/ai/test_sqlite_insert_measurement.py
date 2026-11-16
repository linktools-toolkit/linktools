#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage probes count bound message rows independently of SQL audit clocks."""

import pytest

from ._sqlite_insert_measurement import bound_insert_rows
from .test_native_message_content import _Traffic
from .test_native_snapshot_baselines import _NativeWrites
from .test_step_message_cache import _SqlTraffic


@pytest.mark.parametrize("many", (False, True))
@pytest.mark.parametrize("audit_clock", (False, True))
def test_storage_probes_keep_rows_identities_and_body_bytes_with_sql_clocks(many, audit_clock):
    # Include a binary body and a manifest-only row. Put an SQL clock before
    # message as well as after it to expose incorrect positional accounting.
    columns = "store_key, sha256, message, compression, raw_size, payload_json, store_locator"
    values = "?, ?, ?, ?, ?, ?, ?"
    if audit_clock:
        columns = columns.replace("message,", "updated_at, message,") + ", created_at"
        values = "?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP"
    statement = f"INSERT INTO ai_step_messages ({columns}) VALUES ({values})"
    rows = (("store", "first", memoryview(b"compressed"), "zlib", 1000, None, "locator"),
            ("store", "second", None, "raw", 20, '{"name":"文"}', "locator"))
    if not many:
        statement += f", ({values})"
    statement += " ON CONFLICT (sha256) DO NOTHING"
    parameters = rows if many else tuple(value for row in rows for value in row)
    traffic, identities, submissions = _SqlTraffic(), _NativeWrites(), _Traffic()
    for probe in (traffic, identities, submissions):
        probe.capture(None, None, statement, parameters, None, many)
    assert traffic.message_rows == 2
    assert traffic.message_bytes == len(b"compressed")
    assert traffic.bound_bytes == _SqlTraffic._bytes(parameters)
    assert identities.identities["ai_step_messages"] == ["first", "second"]
    assert submissions.submitted_rows["ai_step_messages"] == 2
    assert traffic.statements["message_insert"] == 1


def test_insert_measurement_rejects_incomplete_parameter_rows():
    statement = "INSERT INTO ai_step_messages (sha256, message, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)"
    with pytest.raises(AssertionError):
        list(bound_insert_rows(statement, ("identity",), False))


def test_insert_measurement_rejects_unrecognized_value_expressions():
    statement = "INSERT INTO ai_step_messages (sha256, message) VALUES (?, NULL)"
    with pytest.raises(AssertionError):
        list(bound_insert_rows(statement, ("identity",), False))
