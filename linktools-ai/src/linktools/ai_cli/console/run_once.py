#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``lt ai run`` business logic.

Runs one agent task against a freshly-built client, streams the events to the
console renderer, and returns the exit codes: ``0`` on completion, ``4`` when
the run pauses for approval (with run_id/approval_id printed), ``130`` on Ctrl+C
(after cancelling the run through the runtime, not just the process)."""

import asyncio

from linktools.core import environ

from ..client import RunRequest, RuntimeClient, build_runtime_client, new_run_id
from .renderer import announce_paused, print_event


def run_once(
    *,
    prompt: "str | None",
    agent: "str | None",
    session: "str | None",
    base_url: "str | None",
    model: "str | None",
    api_key: "str | None",
    json_output: bool,
    client: "RuntimeClient | None" = None,
) -> int:
    """Run one task. ``client`` is injectable for tests; when omitted a local
    client is built from the model flags/env."""
    return asyncio.run(
        _run_once_async(
            prompt=prompt,
            agent=agent,
            session=session,
            base_url=base_url,
            model=model,
            api_key=api_key,
            json_output=json_output,
            client=client,
        )
    )


async def _run_once_async(
    *,
    prompt: "str | None",
    agent: "str | None",
    session: "str | None",
    base_url: "str | None",
    model: "str | None",
    api_key: "str | None",
    json_output: bool,
    client: "RuntimeClient | None",
) -> int:
    logger = environ.logger
    if prompt is None:
        from linktools.cli import CommandError

        raise CommandError("a prompt is required")

    own_client = client is None
    if own_client:
        # --json is non-interactive; --yes (the framework's global no-input flag)
        # is too — both must suppress prompts.
        from linktools.rich import is_no_input

        client = build_runtime_client(
            model=model,
            base_url=base_url,
            api_key=api_key,
            with_model=True,
            interactive=(not json_output) and not is_no_input(),
        )

    run_id = new_run_id()
    request = RunRequest(
        prompt=prompt,
        session_id=session or "main",
        agent_id=agent,
        run_id=run_id,
    )
    try:
        async for event in client.run_stream(request):
            if event.get("type") == "paused":
                if json_output:
                    # Emit the pause as a structured event; the exit code (4)
                    # is the machine-readable signal a CI parser checks.
                    print_event(event, json_output=True, logger=logger)
                else:
                    approval = await _fetch_approval(client, event.get("approval_id"))
                    announce_paused(approval, event, logger)
                return 4
            print_event(event, json_output=json_output, logger=logger)
        if not json_output:
            print()
        return 0
    except asyncio.CancelledError:
        # Ctrl+C mid-run: cancel the Run through the runtime so it actually
        # stops executing, then surface the interrupt exit code.
        await client.cancel(run_id)
        logger.warning("run cancelled")
        return 130


async def _fetch_approval(client: "RuntimeClient", approval_id: "str | None"):
    """Read approval detail through the client, not Storage."""
    if approval_id is None:
        return None
    return await client.get_approval(approval_id)
