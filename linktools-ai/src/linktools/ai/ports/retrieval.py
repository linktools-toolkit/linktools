#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Retriever protocol with provenance-bearing results."""

from typing import Protocol


class Retriever(Protocol):
    async def retrieve(self, context: object) -> "tuple[object, ...]": ...


__all__ = ["Retriever"]
