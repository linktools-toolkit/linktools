#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SecurityCapability: destructive shell command blacklist.

This is one of two independent defenses; the other (path-escape confinement)
lives in `execution/local.py` as an unconditional invariant, not a hook, and
is not controlled by this capability or by `enable_security_preset`.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import SkipToolExecution

if TYPE_CHECKING:
    from pydantic_ai import RunContext
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.tools import ToolDefinition

_DESTRUCTIVE_PATTERNS: "tuple[re.Pattern[str], ...]" = (
    re.compile(r"rm\s+(-\w*r\w*f\w*|-\w*f\w*r\w*)\s+/(\s|\*|$)"),  # rm -rf /, rm -rf /*
    re.compile(r"\bdd\b.*\bof=/dev/"),  # dd ... of=/dev/sdX — overwrite a block device
    re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),  # classic fork bomb
    re.compile(r"\bmkfs\."),  # reformat a filesystem
    re.compile(r">\s*/dev/(sd|nvme|hd)[a-z0-9]"),  # redirect output onto a raw block device
)

_BASH_TOOL_NAMES = frozenset({"bash"})


@dataclass
class SecurityCapability(AbstractCapability[None]):
    """`before_tool_execute` hook: blocks `bash` calls matching a known-destructive pattern."""

    async def before_tool_execute(
        self,
        ctx: "RunContext[Any]",
        *,
        call: "ToolCallPart",
        tool_def: "ToolDefinition",
        args: Any,
    ) -> Any:
        if tool_def.name in _BASH_TOOL_NAMES:
            command = str((args or {}).get("command", ""))
            for pattern in _DESTRUCTIVE_PATTERNS:
                if pattern.search(command):
                    raise SkipToolExecution({
                        "error": f"blocked: command matches a known-destructive pattern ({pattern.pattern})",
                    })
        return args
