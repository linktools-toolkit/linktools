#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bounded Modal command execution adapter."""


class ModalCommandExecutor:
    def __init__(self, sandbox: object) -> None:
        self._sandbox = sandbox

    async def execute(self, command: object) -> object:
        return await self._sandbox.exec(command)


__all__ = ["ModalCommandExecutor"]
