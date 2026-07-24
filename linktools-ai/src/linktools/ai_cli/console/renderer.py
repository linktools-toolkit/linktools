#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Console rendering of Runtime dict-events.

``print_event`` renders one streamed event to stdout/_logger; ``announce_paused``
expands a pause event with the approval request's tool/arguments/reason. The
caller fetches that request through :meth:`RuntimeClient.get_approval` -- the
renderer itself never touches Storage (the console operates the backend only
through RuntimeClient)."""

import json
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import logging


def print_event(
    event: "Mapping[str, Any]",
    *,
    json_output: bool,
    logger: "logging.Logger",
) -> None:
    """Render one non-paused stream event.

    ``text`` -> stdout (streamed, no newline); ``tool`` -> _logger (collapsed);
    ``resumed`` -> _logger. ``paused`` is handled by the caller via
    :func:`announce_paused` (it terminates the stream). In ``--json`` mode every
    event is emitted as one JSON line."""
    if json_output:
        print(json.dumps(event, default=str))
        return
    kind = event.get("type")
    if kind == "text":
        print(event.get("text", ""), end="", flush=True)
    elif kind == "tool":
        ok = " ok" if event.get("ok") else ""
        logger.info(f"[tool: {event.get('name')} {event.get('phase')}{ok}]")
    elif kind == "resumed":
        logger.info(f"resumed run: {event.get('run_id')}")


def announce_failed(event: "Mapping[str, Any]", logger: "logging.Logger") -> None:
    """Render a ``failed`` stream event -- the Outcome-model replacement for a
    raised exception (spec section 12.3): the run ended FAILED without ever
    raising out of ``run_stream()``, so the console reports it explicitly
    instead of relying on an uncaught exception reaching a top-level handler."""
    logger.error(f"run failed: {event.get('error_type')}: {event.get('message')}")


def announce_cancelled(event: "Mapping[str, Any]", logger: "logging.Logger") -> None:
    """Render a ``cancelled`` stream event (the run ended CANCELLED without
    ever raising -- see :func:`announce_failed`)."""
    logger.warning(f"run cancelled: {event.get('run_id')}")


def announce_paused(
    approval_request: Any,
    event: "Mapping[str, Any]",
    logger: "logging.Logger",
) -> None:
    """Render a ``paused`` stream event.

    Prints the fields the spec requires -- ``tool``, ``arguments``, ``reason``,
    ``run_id``, ``approval_id`` -- plus the cross-process command that resumes
    the run. ``approval_request`` is the request fetched by the caller via
    ``RuntimeClient.get_approval`` (best-effort: only ``run_id``/``approval_id``
    travel on the stream event); ``None`` degrades to the ids alone."""
    run_id = event.get("run_id")
    approval_id = event.get("approval_id")
    tool_name: "str | None" = None
    reason: "str | None" = None
    arguments: "dict[str, Any]" = {}
    if approval_id is not None and approval_request is not None:
        tool_name = approval_request.tool_name
        reason = approval_request.reason
        arguments = dict(approval_request.arguments)
    logger.warning("run paused waiting for approval")
    logger.info(f"tool: {tool_name or '?'}")
    logger.info(f"arguments: {arguments}")
    if reason:
        logger.info(f"reason: {reason}")
    logger.info(f"run_id: {run_id}")
    logger.info(f"approval_id: {approval_id}")
    logger.info(f"resume with: lt ai continue {run_id} --approve")
