#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Structured execution over docker/compose processes.

Uniformly captures stdout/stderr, returncode, duration and timeout for a
process created by ``manager.runtime.create_process``/``create_docker_process``/
``create_docker_compose_process`` (with ``capture_output=True``), and parses
JSON output. This is the shared foundation Actual Status, Plan preflight,
Doctor and Lock all build on: it wraps existing process execution, it is not
a second runtime.

``ProcessResult`` (in ``linktools.runtime``) is not reused as this module's
result type: its ``stdout``/``stderr`` are the raw, unconsumed ``Popen``
streams (or ``None``), not decoded text -- unsuited as a business-facing
result for cntr's callers.
"""
import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from typing import Any
    from linktools.runtime import Process
    from linktools.types import TimeoutType
    from ..manager import ContainerManager


class StructuredCommandError(ContainerError):
    pass


class StructuredCommandTimeout(StructuredCommandError):
    pass


class StructuredCommandOutputError(StructuredCommandError):
    pass


_TRUNCATE_AT = 4000

_REDACTED = "***"
# Flags whose following token is a KEY=VALUE pair that may embed a secret,
# e.g. --build-arg http_proxy=http://user:pass@host -- the value, not the
# whole token, needs redacting. Shared with execution/report.py so a
# command is never displayed unredacted through either path (spec sections
# 7/46: "敏感参数脱敏").
_VALUE_BEARING_FLAGS = ("--build-arg",)


def redact_command(args: "tuple[str, ...] | None") -> "tuple[str, ...] | None":
    if not args:
        return args
    redacted = []
    redact_next = False
    for token in args:
        if redact_next:
            key, sep, _value = token.partition("=")
            redacted.append(f"{key}={_REDACTED}" if sep else _REDACTED)
            redact_next = False
        else:
            redacted.append(token)
        if token in _VALUE_BEARING_FLAGS:
            redact_next = True
    return tuple(redacted)


def _truncate(text: "str | None") -> str:
    if not text:
        return ""
    if len(text) <= _TRUNCATE_AT:
        return text
    return text[:_TRUNCATE_AT] + "...(truncated)"


def _decode(chunk) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    return chunk.decode(errors="ignore")


@dataclass(frozen=True)
class CommandResult:
    args: "tuple[str, ...]"
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


class StructuredCommandRunner:
    """Execute a ``Process`` and collect a ``CommandResult``.

    ``process`` must already be created with ``capture_output=True`` by one of
    the ``RuntimeProcessFactory`` methods -- this runner only consumes and
    times it, it never spawns a process itself.
    """

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def execute(
            self,
            process: "Process",
            timeout: "TimeoutType" = None,
            check: bool = True,
            error_type: "type | None" = None,
    ) -> "CommandResult":
        # Redacted immediately: CommandResult.args and every exception message
        # built from `args` below must never carry a secret (e.g. a
        # --build-arg proxy URL with embedded credentials) -- spec sections
        # 7/81 "敏感参数脱敏"/"命令脱敏".
        args = redact_command(tuple(str(a) for a in (getattr(process, "args", None) or ())))
        started = time.monotonic()
        stdout_chunks: "list[str]" = []
        stderr_chunks: "list[str]" = []
        try:
            for out, err in process.fetch(timeout=timeout):
                if out:
                    stdout_chunks.append(_decode(out))
                if err:
                    stderr_chunks.append(_decode(err))
        finally:
            # Always reap the process tree, timed out or not -- mirrors
            # Process.exec()'s own finally-block cleanup.
            process.recursive_kill()
        duration = time.monotonic() - started

        returncode = process.returncode
        timed_out = returncode is None

        result = CommandResult(
            args=args,
            returncode=returncode if returncode is not None else -1,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            duration=duration,
            timed_out=timed_out,
        )

        if timed_out:
            raise StructuredCommandTimeout(
                f"Command timed out after {duration:.1f}s: {' '.join(args)}"
            )
        if check and not result.succeeded:
            error_cls = error_type or StructuredCommandError
            raise error_cls(
                f"Command failed (exit {result.returncode}): {' '.join(args)}\n"
                f"stderr: {_truncate(result.stderr)}"
            )
        return result

    def execute_text(
            self,
            process: "Process",
            timeout: "TimeoutType" = None,
            check: bool = True,
    ) -> "CommandResult":
        return self.execute(process, timeout=timeout, check=check)

    def execute_json(
            self,
            process: "Process",
            timeout: "TimeoutType" = None,
            check: bool = True,
    ) -> "Any":
        result = self.execute(process, timeout=timeout, check=check)
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise StructuredCommandOutputError(
                f"Command produced invalid JSON: {' '.join(result.args)}\n"
                f"error: {exc}\n"
                f"stdout: {_truncate(result.stdout)}\n"
                f"stderr: {_truncate(result.stderr)}"
            ) from exc
