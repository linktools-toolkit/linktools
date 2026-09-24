#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read raw Rule Markdown from a captured Asset source."""

from ..asset import AssetStoreReader
from ..core import DEFAULT_DISCOVERY_POLICY
from ..errors import AIError, ErrorCode


class AssetRuleResourceSource:
    """Load version-pinned Rule Markdown without interpreting its contents."""

    def __init__(self, source_id: str, reader: AssetStoreReader) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("rule source id must be non-empty")
        if not isinstance(reader, AssetStoreReader):
            raise TypeError("reader must provide read-only AssetStore operations")
        self._id = source_id
        self._reader = reader

    @property
    def id(self) -> str:
        return self._id

    async def load(self) -> tuple[tuple[str, str], ...]:
        """Return sorted `(asset_id, markdown)` pairs from one source revision."""
        revision = await self._reader.current_revision()
        infos = tuple(
            sorted(
                (
                    info
                    for info in await self._reader.metadata_snapshot()
                    if info.key.kind == "rule"
                    and info.key.id.endswith(".md")
                    and not DEFAULT_DISCOVERY_POLICY.ignores(info.key.id)
                ),
                key=lambda info: info.key.id,
            )
        )
        if not infos:
            if await self._reader.current_revision() != revision:
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            return ()

        refs = await self._reader.resolve_versions(
            tuple(info.key for info in infos)
        )
        if len(refs) != len(infos) or any(
            not ref.matches_info(info)
            for info, ref in zip(infos, refs, strict=True)
        ):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        values = await self._reader.read_versions(refs)
        if len(values) != len(infos):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if await self._reader.current_revision() != revision:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)

        rules: list[tuple[str, str]] = []
        for info, value in zip(infos, values, strict=True):
            try:
                content = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise AIError(
                    ErrorCode.ASSET_CODEC_UNKNOWN,
                    safe_details={
                        "source_id": self._id,
                        "asset_id": info.key.id,
                    },
                ) from error
            rules.append((info.key.id[:-3], content))
        return tuple(rules)


__all__ = ["AssetRuleResourceSource"]
