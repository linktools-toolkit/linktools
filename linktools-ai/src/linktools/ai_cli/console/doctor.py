#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``lt ai doctor`` business logic.

Delegates every check to :meth:`LocalRuntimeClient.doctor` (which owns the
project bundle) and only renders the resulting :class:`DoctorReport` here. The
``--project`` flag overrides where project discovery starts; ``--remote`` would
target the HTTP client (unsupported in this build, fails explicitly)."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ

from ..client import build_runtime_client

if TYPE_CHECKING:
    from linktools.ai_cli.client import RuntimeClient


def run_doctor(
    *,
    project: "Path | None",
    remote: "str | None",
    json_output: bool,
) -> int:
    return asyncio.run(
        _doctor_async(project=project, remote=remote, json_output=json_output)
    )


async def _doctor_async(
    *,
    project: "Path | None",
    remote: "str | None",
    json_output: bool,
    client: "RuntimeClient | None" = None,
) -> int:
    logger = environ.logger
    if client is None:
        client = build_runtime_client(remote=remote, with_model=False, project=project)
    report = await client.doctor()
    if json_output:
        payload = {
            "checks": [
                {"label": c.label, "ok": c.ok, "detail": c.detail}
                for c in report.checks
            ],
            "failed": len(report.failed),
        }
        print(json.dumps(payload, default=str))
        return 1 if report.failed else 0
    for check in report.checks:
        if check.ok:
            logger.info(f"[ok] {check.label}")
        else:
            logger.error(f"[fail] {check.label}: {check.detail}")
    if report.failed:
        logger.error(f"{len(report.failed)} check(s) failed")
        return 1
    return 0
