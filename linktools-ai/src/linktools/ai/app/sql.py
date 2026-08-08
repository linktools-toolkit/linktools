#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Application-level SQL bootstrap plan without shared schema ownership."""

from dataclasses import dataclass

from linktools.core import environ


_logger = environ.get_logger("ai.app.sql")


@dataclass(frozen=True, slots=True)
class SqlSchemaPlan:
    asset_digest: str
    runtime_digest: str
    step_digest: "str | None"

    def __post_init__(self) -> None:
        if not self.asset_digest.strip() or not self.runtime_digest.strip():
            raise ValueError("SQL schema plan digests are required")


def build_sql_plan(*, asset_digest: str, runtime_digest: str, step_digest: "str | None" = None) -> SqlSchemaPlan:
    plan = SqlSchemaPlan(asset_digest, runtime_digest, step_digest)
    _logger.info("SQL schema plan prepared: runtime=%s step=%s", plan.runtime_digest, plan.step_digest)
    return plan


__all__ = ["SqlSchemaPlan", "build_sql_plan"]
