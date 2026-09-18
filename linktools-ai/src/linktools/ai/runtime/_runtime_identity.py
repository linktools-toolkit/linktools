#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable identities shared by Runtime composition and history."""

import hashlib


def grant_key(namespace: str) -> bytes:
    """Return the namespace-scoped key used for local capability grants."""
    return hashlib.sha256(f"workspace:{namespace}".encode()).digest()


__all__ = ["grant_key"]
