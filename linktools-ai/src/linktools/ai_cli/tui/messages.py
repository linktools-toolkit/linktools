#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UI messages posted from a Textual Worker to the ChatScreen.

Streaming work runs inside a ``@work`` worker; the worker must not touch widgets
directly (it can be cancelled mid-iteration). It posts these immutable messages
and the screen's ``on_*`` handlers update the widgets on the UI thread."""

from textual.message import Message


class RunEventMessage(Message):
    """One Runtime dict-event streamed from ``RuntimeClient.run_stream``."""

    def __init__(self, event) -> None:
        self.event = event
        super().__init__()


class RunFinishedMessage(Message):
    """A run completed normally."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__()


class RunFailedMessage(Message):
    """A run failed with an exception (a cancel is NOT a failure)."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__()
