#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`lt ai approvals`: list pending approval requests across all runs."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.ai.agent.approval import ApprovalStatus
from linktools.cli import BaseCommand

from .assembly import project_storage

if TYPE_CHECKING:
    from argparse import Namespace

    from linktools.cli import CommandParser


class Command(BaseCommand):
    """list pending approval requests"""

    def init_arguments(self, parser: "CommandParser") -> None:
        pass

    def run(self, args: "Namespace") -> "int | None":
        storage = project_storage()
        return asyncio.run(self._list_async(storage))

    async def _list_async(self, storage) -> "int | None":
        requests = await _list_pending_approvals(storage)
        if not requests:
            self.logger.info("no pending approvals")
            return 0
        for req in requests:
            self.logger.info(f"{req.id}\trun={req.run_id}\ttool={req.tool_name}")
        return 0


async def _list_pending_approvals(storage) -> list:
    """Enumerate pending approval requests under the project's storage.

    `ApprovalStore.list_pending(run_id)` is scoped to a single run; there is no
    whole-store listing API. We glob ``<root>/approvals/requests/*.json``,
    rehydrate each request via the store's `get()`, and keep only pending ones.
    """
    root = Path(storage.root) / "approvals" / "requests"
    requests = []
    for request_path in sorted(root.glob("*.json")):
        approval_id = request_path.stem
        request = await storage.approvals.get(approval_id)
        if request is not None and request.status == ApprovalStatus.PENDING:
            requests.append(request)
    return requests


command = Command()
