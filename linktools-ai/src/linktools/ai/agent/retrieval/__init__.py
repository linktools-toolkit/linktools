#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retrieval: the retrieval subsystem. Provides the Retriever
Protocol and a KnowledgeContext that renders retrieved documents/memories into a
prompt section. Pure data + thin adapters -- no I/O of its own."""

from .context import KnowledgeContext, format_memory, format_untrusted_context
from .document import Document
from .retriever import MemoryRetriever, Retriever
from .scope import RetrievalScope
from .trust import ContextItem, ContextTrustLevel

__all__ = [
    "ContextItem",
    "ContextTrustLevel",
    "Document",
    "KnowledgeContext",
    "MemoryRetriever",
    "RetrievalScope",
    "Retriever",
    "format_memory",
    "format_untrusted_context",
]
