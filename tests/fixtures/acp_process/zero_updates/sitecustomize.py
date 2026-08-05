#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Suppress execution events in the child process for the negative smoke gate."""

from linktools.ai.execution.live_events import ExecutionEventHub


async def _publish_without_delivery(self, execution_id, event):
    return None


ExecutionEventHub.publish = _publish_without_delivery
