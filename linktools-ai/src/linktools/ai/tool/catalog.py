#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ToolCatalog: the tool domain Catalog over the generic Catalog contracts.

Same pattern as the other domain catalogs (items are ``{name}.yaml``), plus the
tool-specific ``get_metadata_map`` -- the bridge the policy rule modules
(PermissionRule / RiskRule / ApprovalRule) consume."""

from __future__ import annotations

from ..catalog import CatalogSource, RevisionCache
from ..catalog.parsing import SpecLoader
from ..catalog.source import SpecLoaderSource
from ..governance.policy.rule import ToolPolicyMetadata
from .codec import ToolSpecCodec
from .spec import ToolSpec


class ToolCatalog:
    """The tool domain Catalog: list/get ToolSpec over a CatalogSource with
    revision-cached, single-flight refresh and strict decode."""

    def __init__(
        self,
        source: CatalogSource,
        *,
        codec: "ToolSpecCodec | None" = None,
        suffix: str = ".yaml",
        source_name: "str | None" = None,
    ) -> None:
        self._source = source
        self._codec = codec or ToolSpecCodec()
        self._suffix = suffix
        self._cache: RevisionCache[ToolSpec] = RevisionCache(
            source, self._codec, suffix=suffix, source_name=source_name
        )

    @classmethod
    def from_specloader(
        cls, loader: SpecLoader, *, suffix: str = ".yaml"
    ) -> "ToolCatalog":
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


__all__: "list[str]" = ["ToolCatalog"]
