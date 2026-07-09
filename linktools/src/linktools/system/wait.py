#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Wait helpers built on monotonic Timeout (spec §14.5 SYS-004)."""

import subprocess
import threading
from typing import TYPE_CHECKING

from ..decorator import timeoutable

if TYPE_CHECKING:
    from typing import Any


@timeoutable
def wait_event(event: "threading.Event", timeout: "Any") -> bool:
    """Wait for ``event`` to be set, polling with the remaining timeout budget."""
    interval = 1
    while True:
        t = timeout.remaining
        if t is None:
            t = interval
        elif t <= 0:
            return False
        if event.wait(min(t, interval)):
            return True


@timeoutable
def wait_thread(thread: "threading.Thread", timeout: "Any") -> bool:
    """Wait for ``thread`` to terminate; return True if it did, False on timeout."""
    interval = 1
    while True:
        t = timeout.remaining
        if t is None:
            t = interval
        elif t <= 0:
            return False
        try:
            thread.join(min(t, interval))
        except Exception:
            pass
        if not thread.is_alive():
            return True


@timeoutable
def wait_process(process: "subprocess.Popen", timeout: "Any") -> "int | None":
    """Wait for ``process`` to exit; return its exit code or None on timeout."""
    interval = 1
    while True:
        t = timeout.remaining
        if t is None:
            t = interval
        elif t <= 0:
            return None
        try:
            return process.wait(min(t, interval))
        except subprocess.TimeoutExpired:
            pass
