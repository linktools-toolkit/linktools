#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Small ACP execution adapter helpers."""

from ..execution.domain import RunStatus


class AcpExecutionAdapter:
    @staticmethod
    def stop_reason(status: RunStatus) -> str:
        if status is RunStatus.COMPLETED:
            return "end_turn"
        if status is RunStatus.CANCELLED:
            return "cancelled"
        if status is RunStatus.FAILED:
            raise RuntimeError("failed executions use JSON-RPC internal error")
        raise RuntimeError(f"execution is not terminal: {status.value}")


__all__ = ["AcpExecutionAdapter"]
