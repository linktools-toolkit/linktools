#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool specification index and policy metadata projection."""

from __future__ import annotations

from ..spec import SpecSource
from ..spec.index import SpecIndex
from ..spec.parsing import SpecLoader
from ..spec.source import SpecLoaderSource
from ..governance.policy.rule import ToolPolicyMetadata
from .codec import ToolSpecCodec
from .spec import ToolSpec


class ToolSpecIndex(SpecIndex[ToolSpec]):

    def __init__(
        self,
        source: SpecSource,
        *,
        codec: "ToolSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        super().__init__(
            source,
            codec or ToolSpecCodec(),
            suffix=suffix,
            source_name=source_name,
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "ToolSpecIndex":
        return cls(SpecLoaderSource(loader), suffix=suffix)

    async def list_ids(self) -> "tuple[str, ...]":
        return await self._cache.list_ids()

    async def get(self, tool_id: str) -> ToolSpec:
        return await self._cache.get(tool_id)

    async def get_metadata_map(self) -> "Mapping[str, ToolPolicyMetadata]":
        """Return {tool_name: ToolPolicyMetadata} for every loaded tool -- the
        bridge the PermissionRule/RiskRule/ApprovalRule consume."""
        ids = await self.list_ids()
        result: "dict[str, ToolPolicyMetadata]" = {}
        for tool_id in ids:
            spec = await self.get(tool_id)
            result[spec.name] = ToolPolicyMetadata(
                permissions=spec.permissions,
                risk=spec.risk,
                side_effect=spec.side_effect,
                approval=spec.approval,
                idempotent=spec.idempotent,
                timeout_seconds=spec.timeout_seconds,
                schema_version=spec.schema_version,
                enabled=spec.enabled,
                max_retries=spec.max_retries,
                idempotency_strategy=spec.idempotency_strategy,
                idempotency_key_field=spec.idempotency_key_field,
                metadata=spec.metadata,
            )
        return result


__all__: "list[str]" = ["ToolSpecIndex"]
