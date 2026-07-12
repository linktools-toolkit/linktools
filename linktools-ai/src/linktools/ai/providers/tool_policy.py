#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolPolicyMetadataSource: source-agnostic surface for tool policy metadata. The
Runtime consumes the metadata map to enforce Permission/Risk/Approval rules;
the map can originate from a YAML ToolRegistry, a DB, or any business source."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from collections.abc import Mapping

if TYPE_CHECKING:
    from ..policy.rule import ToolPolicyMetadata


@runtime_checkable
class ToolPolicyMetadataSource(Protocol):
    """Provides a tool-name -> ToolPolicyMetadata map from any source."""

    async def get_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]": ...
