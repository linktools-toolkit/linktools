#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared v1 execution-owner fields for persistence fixtures."""

from collections.abc import Callable
from typing import Any

from pydantic_ai import Tool

from linktools.ai.capability import tool_semantic_metadata
from linktools.ai.runtime._tool_boundary import ManagedToolDescriptor
from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import StoredPayload


def execution_owner_fields(prompt: str = "prompt") -> dict[str, object]:
    return {
        "principal_id": "principal",
        "principal_kind": "service",
        "stored_user_input": StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
        ),
    }


def semantic_tool(
    function: Callable[..., Any],
    descriptor: ManagedToolDescriptor,
) -> Tool[Any]:
    """Build a test leaf with the metadata required by the final boundary."""
    path_fields = list(descriptor.workspace_path_fields) or None
    return Tool(
        function,
        metadata=tool_semantic_metadata(
            effect=descriptor.effect,
            tool_class=descriptor.tool_class,
            path_fields=path_fields,
        ),
    )
