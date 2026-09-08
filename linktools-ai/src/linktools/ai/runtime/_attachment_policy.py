#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attachment-only extension of Runtime trusted tool classification."""

from typing import Any

import linktools.ai.runtime._capabilities as capabilities_runtime

_installed = False
_original_trusted_tool_capability: Any = None


def _trusted_tool_capability(name: str, tool_class: str) -> str | None:
    if name == "read_attachment":
        return (
            capabilities_runtime._WORKSPACE_SANDBOX_CAPABILITY_ID
            if tool_class == "filesystem.read"
            else None
        )
    if _original_trusted_tool_capability is None:
        return None
    return _original_trusted_tool_capability(name, tool_class)


def install_attachment_tool_policy() -> None:
    """Install the read_attachment trusted-tool classification once."""
    global _installed
    global _original_trusted_tool_capability
    if _installed:
        return
    _original_trusted_tool_capability = capabilities_runtime._trusted_tool_capability
    capabilities_runtime._trusted_tool_capability = _trusted_tool_capability
    _installed = True


__all__ = ["install_attachment_tool_policy"]
