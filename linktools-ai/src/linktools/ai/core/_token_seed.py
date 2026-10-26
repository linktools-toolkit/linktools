#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Namespace-scoped opaque-token integrity without authorization semantics."""

import hashlib


def token_seed(namespace: str) -> bytes:
    """Return a deterministic namespace-scoped integrity seed for opaque tokens.

    This seed is not an authorization secret. Runtime services must bind access
    to an authorization policy or durable authorization receipt independently
    of this signature.
    """
    return hashlib.sha256(f"runtime-token-v1:{namespace}".encode()).digest()


__all__ = ["token_seed"]
