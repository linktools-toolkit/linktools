#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bundle and manifest signing helpers."""

import hmac

from ..foundation.digest import hmac_digest


class BundleSigner:
    def sign(self, content: bytes, key: bytes) -> str:
        return hmac_digest(key, content)

    def verify(self, content: bytes, signature: str, key: bytes) -> bool:
        return hmac.compare_digest(self.sign(content, key), signature)


__all__ = ["BundleSigner"]
