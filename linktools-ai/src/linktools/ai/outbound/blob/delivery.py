#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Controlled artifact download delivery."""

from datetime import datetime, timezone

from ...domain.execution import ArtifactDownload
from ...foundation.digest import hmac_digest


class BlobDelivery:
    def __init__(self, object_store: object, signing_key: bytes) -> None:
        self._object_store = object_store
        self._signing_key = signing_key

    async def prepare_download(self, artifact_id: str, expires_at: datetime) -> object:
        current = datetime.now(timezone.utc)
        checked_expiry = expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
        if checked_expiry <= current:
            raise ValueError("artifact download grant has expired")
        token = hmac_digest(self._signing_key, f"{artifact_id}\0{checked_expiry.isoformat()}".encode("utf-8"))
        return ArtifactDownload(artifact_id=artifact_id, uri=f"lt://artifact/{token}", expires_at=checked_expiry)

    async def open_stream(self, artifact_id: str) -> object:
        return await self._object_store.get(artifact_id)


__all__ = ["BlobDelivery"]
