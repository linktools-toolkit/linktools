#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Retrieval protocol and provenance-bearing public values."""

from typing import Protocol

from ..domain.retrieval import RetrievalContext, RetrievalResult, RetrievalScope, RetrievalTrust


class Retriever(Protocol):
    async def retrieve(self, context: RetrievalContext) -> "tuple[RetrievalResult, ...]": ...


__all__ = ["RetrievalContext", "RetrievalResult", "RetrievalScope", "RetrievalTrust", "Retriever"]
