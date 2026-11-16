#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decode bound rows for the SQLite INSERT shapes used by storage probes."""

import re
from collections.abc import Iterator


def bound_insert_rows(statement: str, parameters, many: bool) -> Iterator[dict[str, object]]:
    columns = [column.strip().strip('\"`')
               for column in statement.split("(", 1)[1].split(")", 1)[0].split(",")]
    # Audit clocks are SQL expressions, not driver parameters. Do not count
    # those columns as bound values or shift message/identity positions.
    values = re.search(r"\bVALUES\s*\(([^()]*)\)", statement)
    assert values is not None, statement
    expressions = [value.strip() for value in values.group(1).split(",")]
    assert len(columns) == len(expressions), statement
    assert all(value in {"?", "CURRENT_TIMESTAMP"} for value in expressions), statement
    bound_columns = [column for column, value in zip(columns, expressions) if value == "?"]
    width = len(bound_columns)
    assert width > 0, statement
    for batch in parameters if many else (parameters,):
        assert len(batch) % width == 0, (statement, len(batch), width)
        for offset in range(0, len(batch), width):
            yield dict(zip(bound_columns, batch[offset:offset + width]))
