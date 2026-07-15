#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``lt ai continue`` business logic.

One entry point that absorbs the former ``approve``/``reject``/``resume``
commands. It inspects the run's status and dispatches:

* ``WAITING_APPROVAL`` + ``--approve``  -> approve the pending request, resume
* ``WAITING_APPROVAL`` + ``--reject``   -> reject the pending request, cancel
* ``WAITING_APPROVAL`` + ``--resume``   -> resume (already approved)
* ``WAITING_APPROVAL`` + (no flag)      -> interactive approve/reject/later
* ``RUNNING``                           -> report still running
* terminal                              -> show status, do not re-execute
* anything else                         -> resume

Resume re-drives the original agent spec from the persisted snapshot, so only
the run id is needed. A second pause while resuming again exits 4."""

import asyncio

from linktools.ai.run.models import RunStatus
from linktools.cli import CommandError
from linktools.core import environ

from ..client import RuntimeClient, build_runtime_client
from .renderer import announce_paused, print_event

_APPROVE = "approve"
_REJECT = "reject"
_LATER = "later"

_TERMINAL = (
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.CANCELLING,
)


def continue_run(
    run_id: str,
    *,
    approve: bool = False,
    reject: bool = False,
    resume: bool = False,
    base_url: "str | None" = None,
    model: "str | None" = None,
    api_key: "str | None" = None,
    client: "RuntimeClient | None" = None,
) -> int:
    with_model = not reject
    return asyncio.run(
        _continue_async(
            run_id=run_id,
            approve=approve,
            reject=reject,
            resume=resume,
            base_url=base_url,
            model=model,
            api_key=api_key,
            client=client,
            with_model=with_model,
        )
    )


async def _continue_async(
    *,
    run_id: str,
    approve: bool,
    reject: bool,
    resume: bool,
    base_url: "str | None" = None,
    model: "str | None" = None,
    api_key: "str | None" = None,
    client: "RuntimeClient | None" = None,
    with_model: bool = True,
) -> int:
    logger = environ.logger
    if client is None:
        client = build_runtime_client(
            with_model=with_model,
            base_url=base_url,
            model=model,
            api_key=api_key,
        )

    record = await client.get_run(run_id)
    if record is None:
        raise CommandError(f"run not found: {run_id}")
    status = record.status

    if status == RunStatus.WAITING_APPROVAL:
        if approve:
            return await _approve_and_resume(client, run_id, logger)
        if reject:
            return await _reject_and_cancel(client, run_id, logger)
        if resume:
            return await _resume(client, run_id, logger)
        return await _interactive(client, run_id, logger)

    if status == RunStatus.RUNNING:
        logger.info(f"run {run_id} still running")
        return 0

    if status in _TERMINAL:
        logger.info(f"run {run_id} already {status.value}")
        return 0

    # Any other (recoverable) state: resume.
    return await _resume(client, run_id, logger)


async def _resume(client: "RuntimeClient", run_id: str, logger) -> int:
    try:
        async for event in client.resume_stream(run_id):
            if event.get("type") == "paused":
                # Paused again mid-resume: surface and stop. Read the approval
                # detail through the client, not Storage.
                approval_id = event.get("approval_id")
                approval = (
                    await client.get_approval(approval_id) if approval_id else None
                )
                announce_paused(approval, event, logger)
                return 4
            print_event(event, json_output=False, logger=logger)
        print()
        return 0
    except asyncio.CancelledError:
        await client.cancel(run_id)
        logger.warning("resume cancelled")
        return 130


async def _approve_and_resume(client: "RuntimeClient", run_id: str, logger) -> int:
    approval = await _find_pending_approval(client, run_id)
    await client.approve(approval.id)
    logger.info(f"approved {approval.id}")
    return await _resume(client, run_id, logger)


async def _reject_and_cancel(client: "RuntimeClient", run_id: str, logger) -> int:
    approval = await _find_pending_approval(client, run_id)
    await client.reject(approval.id)
    await client.cancel(run_id)
    logger.info(f"rejected {approval.id} and cancelled {run_id}")
    return 0


async def _interactive(client: "RuntimeClient", run_id: str, logger) -> int:
    """No flag + WAITING_APPROVAL: offer approve/reject/later."""
    choice = await asyncio.to_thread(_prompt_choice)
    if choice == _APPROVE:
        return await _approve_and_resume(client, run_id, logger)
    if choice == _REJECT:
        return await _reject_and_cancel(client, run_id, logger)
    logger.info(f"resume later: lt ai continue {run_id} --approve")
    return 0


async def _find_pending_approval(client: "RuntimeClient", run_id: str):
    """The pending approval request for ``run_id``. ``list_approvals`` already
    returns only pending requests; pick the one whose run matches."""
    for approval in await client.list_approvals():
        if getattr(approval, "run_id", None) == run_id:
            return approval
    raise CommandError(f"no pending approval for run {run_id}")


def _prompt_choice() -> str:
    try:
        raw = input("approve/reject/later [later]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return _LATER
    if raw.startswith("a"):
        return _APPROVE
    if raw.startswith("r"):
        return _REJECT
    return _LATER
