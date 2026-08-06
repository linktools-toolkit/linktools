#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""S3-compatible ObjectStore boundary."""


class S3ObjectStore:
    def __init__(self, client: object, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    async def put(self, key: str, content: bytes) -> object:
        return await self._client.put_object(Bucket=self._bucket, Key=key, Body=content)

    async def get(self, key: str) -> bytes:
        response = await self._client.get_object(Bucket=self._bucket, Key=key)
        return await response["Body"].read()

    async def head(self, key: str) -> object:
        return await self._client.head_object(Bucket=self._bucket, Key=key)

    async def delete(self, key: str) -> None:
        await self._client.delete_object(Bucket=self._bucket, Key=key)


__all__ = ["S3ObjectStore"]
