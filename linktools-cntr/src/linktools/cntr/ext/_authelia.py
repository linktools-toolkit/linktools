#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy native access rules and callbacks for the existing Authelia client."""
from typing import TYPE_CHECKING
from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import urlsplit

from linktools.decorator import cached_property
from linktools.types import MISSING
from ._base import Integration
from ..errors import ContainerError

if TYPE_CHECKING:
    from typing import Any, Sequence
    from ._nginx import Nginx


class Authelia(Integration):
    consumer = "authelia"

    def __init__(self) -> None:
        self.site: "Nginx | tuple[Integration, ...] | None" = MISSING
        self._rule = MISSING
        self._redirect_uris = ()

    @classmethod
    def access(cls, rule: "Mapping[str, Any]" = MISSING, *,
               site: "Nginx | tuple[Integration, ...] | None" = None, **fields: "Any") -> "Authelia":
        """Declare native access-control fields for a producer's site."""
        if rule is not MISSING and fields:
            raise TypeError("Authelia.access accepts a rule mapping or keyword fields, not both")
        self = cls()
        self.site = site
        self._rule = fields if rule is MISSING else rule
        return self

    @cached_property
    def rule(self) -> "Mapping[str, Any]":
        if not isinstance(self._rule, Mapping):
            raise ContainerError("Authelia access rule must be a mapping")
        return MappingProxyType(dict(self._rule))

    @classmethod
    def oidc(cls, redirect_uris: "Sequence[str]") -> "Authelia":
        """Contribute exact callback URLs without creating clients or credentials."""
        self = cls()
        self._redirect_uris = tuple(redirect_uris) if type(redirect_uris) in (tuple, list) else redirect_uris
        return self

    @cached_property
    def redirect_uris(self) -> "tuple[str, ...]":
        if isinstance(self._redirect_uris, (str, bytes)):
            raise ContainerError("Authelia redirect_uris must be a sequence of URLs")
        result = []
        for value in self._redirect_uris:
            if not isinstance(value, str):
                raise ContainerError("Authelia redirect URI must be a string")
            value = str(value)
            if not value:
                continue
            parsed = urlsplit(value)
            if (not parsed.scheme or "#" in value or "{{" in value or "}}" in value
                    or parsed.scheme in ("http", "https") and not parsed.netloc):
                raise ContainerError("Authelia redirect URI must be absolute without a fragment")
            if value not in result:
                result.append(value)
        return tuple(result)
