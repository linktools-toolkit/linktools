#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlNamingStrategy: the frozen SqlAlchemyStorageAdapter constructor
parameter. Holds a SQLAlchemy ``naming_convention`` (the standard SQLAlchemy
mechanism for deriving constraint/index names) so a downstream can standardize
them -- e.g. for explicit Alembic migration DDL where stable constraint names
matter.

The default :data:`DEFAULT_SQL_NAMING` carries an empty convention, so the
models' declared ``__tablename__`` / column names are used verbatim (the
in-repo behavior). The adapter applies ``naming.naming_convention`` to the
shared declarative metadata at construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SqlNamingStrategy:
    """A SQLAlchemy naming convention the adapter applies to the metadata.

    ``naming_convention`` follows SQLAlchemy's naming-convention API (keys like
    ``"ix"``, ``"uq"``, ``"ck"``, ``"fk"``, ``"pk"`` mapped to templates). The
    default empty mapping means "use the declared names verbatim"."""

    naming_convention: "Mapping[str, str]" = field(default_factory=dict)


DEFAULT_SQL_NAMING = SqlNamingStrategy()


__all__: "list[str]" = ["SqlNamingStrategy", "DEFAULT_SQL_NAMING"]
