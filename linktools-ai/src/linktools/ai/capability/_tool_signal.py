#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinkTools-owned signals raised by model-visible function tools."""

_MAX_MESSAGE_LENGTH = 2048


def _validate_message(message: str) -> str:
    if (
        not isinstance(message, str)
        or not message.strip()
        or len(message) > _MAX_MESSAGE_LENGTH
    ):
        raise ValueError("tool call signal message must be a non-empty short string")
    return message


class ToolCallRetry(Exception):
    """Ask the model to revise and retry the current tool call."""

    def __init__(self, message: str) -> None:
        self.message = _validate_message(message)
        super().__init__(self.message)


class ToolCallFailed(Exception):
    """Report the current tool call as failed and let the model adapt."""

    def __init__(self, message: str) -> None:
        self.message = _validate_message(message)
        super().__init__(self.message)


__all__ = ["ToolCallRetry", "ToolCallFailed"]
