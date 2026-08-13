#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Asset-owned opaque Object key derivation."""

import re
from dataclasses import dataclass

from ..storage import namespace_key

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class AssetObjectKeyFactory:
    namespace: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("namespace must be a non-empty string")

    @property
    def namespace_key(self) -> str:
        return namespace_key(self.namespace)

    def key(self, digest: str) -> str:
        if _DIGEST.fullmatch(digest) is None:
            raise ValueError("Asset object digest is invalid")
        return f"v1/asset/{self.namespace_key}/{digest}"


__all__ = ["AssetObjectKeyFactory"]
