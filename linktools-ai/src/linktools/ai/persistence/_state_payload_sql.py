#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Storage and SQL link extraction for canonical StateStore JSON text."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import Text
    from sqlalchemy.sql import ColumnElement


def state_payload_text() -> "Text":
    from sqlalchemy import Text
    from sqlalchemy.dialects.mysql import LONGTEXT

    return Text().with_variant(LONGTEXT(), "mysql")


def state_payload_document(column: "ColumnElement", dialect: str) -> "ColumnElement":
    from sqlalchemy import JSON, cast, type_coerce

    # MySQL and SQLite JSON functions already parse text input. PostgreSQL
    # has distinct JSON operators, so only the link expression needs a cast.
    return cast(column, JSON) if dialect == "postgresql" else type_coerce(column, JSON)
