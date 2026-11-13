#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instruction storage projections with Asset-owned Rule bytes."""

import hashlib
from collections.abc import Mapping

from ..asset import AssetStoreReader, AssetVersionRef, RetainedAssetReader
from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import RepositoryInstructions, RuleInstructionAdapter


class RepositoryInstructionCodec:
    """Borrow the host's Asset owners without creating another history store."""

    def __init__(self, asset_sources: Mapping[str, AssetStoreReader] | None = None) -> None:
        self._asset_sources = dict(asset_sources or {})
        self._adapter = RuleInstructionAdapter()

    async def encode(self, value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Omit Rule bodies only after verifying the retained version and parser."""
        instructions = RepositoryInstructions.from_payload(value)
        documents = []
        for raw, document in zip(value["documents"], instructions.documents, strict=True):
            stored = dict(raw)
            if "content_digest" in stored:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            reader = self._asset_sources.get(document.asset_source_id)
            if document.asset is not None and isinstance(reader, RetainedAssetReader):
                try:
                    data, = await reader.require_retained_versions((document.asset,))
                except AIError as error:
                    if (error.code is not ErrorCode.CAPABILITY_REQUIRED_MISSING
                            or error.safe_details.get("reason") != "asset_history_not_retained"):
                        raise
                else:
                    parsed = self._adapter.decode_markdown(
                        data, asset_source_id=document.asset_source_id, asset=document.asset,
                    )
                    if parsed != document:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    stored.pop("content")
                    stored["content_digest"] = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            documents.append(stored)
        return {**value, "documents": documents}

    async def decode(self, value: object) -> dict[str, JsonValue]:
        """Restore original Rule bytes; never resolve a mutable Asset head."""
        try:
            if not isinstance(value, Mapping) or not isinstance(value.get("documents"), list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            documents = []
            sources: dict[str, list[tuple[int, AssetVersionRef]]] = {}
            for raw in value["documents"]:
                if not isinstance(raw, Mapping):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                document = dict(raw)
                if "content_digest" in document:
                    if "content" in document:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    asset = AssetVersionRef.from_payload(document["asset"])
                    source_id = document["asset_source_id"]
                    if not isinstance(source_id, str) or not source_id.strip():
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    reader = self._asset_sources.get(source_id)
                    if not isinstance(reader, RetainedAssetReader):
                        raise AIError(ErrorCode.CAPABILITY_REQUIRED_MISSING, safe_details={
                            "reason": "repository_instruction_history_unavailable", "asset_source_id": source_id,
                        })
                    sources.setdefault(source_id, []).append((len(documents), asset))
                documents.append(document)
            for source_id, entries in sources.items():
                reader = self._asset_sources[source_id]
                bodies = await reader.require_retained_versions(tuple(asset for _, asset in entries))
                for (index, asset), data in zip(entries, bodies, strict=True):
                    document = documents[index]
                    parsed = self._adapter.decode_markdown(data, asset_source_id=source_id, asset=asset)
                    if parsed.source != document["source"] or parsed.scope != document["scope"]:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    # The fixed Asset version owns the bytes. The stored digest
                    # remains an admission-comparison key, not a second read check.
                    document.pop("content_digest")
                    document["content"] = parsed.content
            result = {**value, "documents": documents}
            RepositoryInstructions.from_payload(result)
            return result
        except AIError as error:
            if error.code is ErrorCode.OUTPUT_CONTRACT_INVALID:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = ["RepositoryInstructionCodec"]
