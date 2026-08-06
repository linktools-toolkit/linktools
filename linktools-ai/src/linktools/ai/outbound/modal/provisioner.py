#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Modal Sandbox provisioning adapter."""


class ModalSandboxProvisioner:
    def __init__(self, client: object) -> None:
        self._client = client

    async def create(self, request: object) -> object:
        return await self._client.create(request)

    async def inspect(self, lease_id: str) -> object:
        return await self._client.inspect(lease_id)

    async def destroy(self, lease_id: str) -> None:
        await self._client.destroy(lease_id)


__all__ = ["ModalSandboxProvisioner"]
