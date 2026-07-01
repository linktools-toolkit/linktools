#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Event hook mechanism for plugins and observability."""

import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    AGENT_START = "agent_start"            # agent invocation started
    AGENT_END = "agent_end"                # agent invocation ended
    LLM_CALL_START = "llm_call_start"      # LLM call started
    MCP_CALL_START = "mcp_call_start"      # MCP tool call started
    SUBAGENT_START = "subagent_start"      # sub-capability call started
    SUBAGENT_END = "subagent_end"          # sub-capability call ended
    POST_LLM_CALL = "post_llm_call"       # after each LLM API call
    POST_MCP_CALL = "post_mcp_call"        # after each MCP tool call


class HookRegistry:
    """Lightweight hook registration and dispatch center.

    Exceptions inside fire() are logged as warnings and do not interrupt the main pipeline.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., None]]] = {}

    def on(self, hook_event: HookEvent | str, handler: Callable[..., None]) -> None:
        key = hook_event.value if isinstance(hook_event, HookEvent) else str(hook_event)
        self._handlers.setdefault(key, []).append(handler)

    def fire(self, hook_event: HookEvent | str, **kwargs: Any) -> None:
        key = hook_event.value if isinstance(hook_event, HookEvent) else str(hook_event)
        for handler in self._handlers.get(key, []):
            try:
                handler(**kwargs)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' handler %s raised: %s",
                    key,
                    getattr(handler, "__qualname__", repr(handler)),
                    exc,
                )
