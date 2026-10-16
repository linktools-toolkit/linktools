#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lazy native access rules and callbacks for the existing Authelia client."""
from copy import copy
from typing import TYPE_CHECKING
from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import urlsplit

from linktools.decorator import cached_property
from linktools.types import MISSING
from linktools.runtime import lazy_load
from ._base import Integration
from ._urls import is_complete_url
from ..errors import ContainerError

if TYPE_CHECKING:
    from typing import Any, Sequence
    from linktools.core import Config, ConfigValue
    from ._nginx import Nginx


class Authelia(Integration):
    consumer = "authelia"

    def __init__(self) -> None:
        self.site: "Nginx | tuple[Integration, ...] | None" = MISSING
        self._rule = MISSING
        self._redirect_uris = ()
        self._base_url = MISSING

    @classmethod
    def access(cls, rule: "Mapping[str, Any] | ConfigValue" = MISSING, *,
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

    def resolve_rule(self, config: "Config") -> "Mapping[str, Any]":
        """Resolve native rule fields in the producer's config without caching."""
        rule = config.resolve(self._rule)
        if not isinstance(rule, Mapping):
            raise ContainerError("Authelia access rule must be a mapping")
        return MappingProxyType(config.resolve(dict(rule)))

    @classmethod
    def oidc(cls, redirect_uris: "Sequence[str | ConfigValue] | ConfigValue") -> "Authelia":
        """Contribute exact callback URLs without creating clients or credentials."""
        self = cls()
        self._redirect_uris = tuple(redirect_uris) if type(redirect_uris) in (tuple, list) else redirect_uris
        return self

    def with_base_url(self, base_url: "str | ConfigValue") -> "Authelia":
        """Bind callback formatting to the endpoint's lazy public URL."""
        result = copy(self)
        result._base_url = base_url
        result.__dict__.pop("redirect_uris", None)
        return result

    @cached_property
    def redirect_uris(self) -> "tuple[str, ...]":
        return self._resolve_redirect_uris()

    def resolve_redirect_uris(self, config: "Config") -> "tuple[str, ...]":
        """Resolve callbacks in the producer's config before URL validation."""
        return self._resolve_redirect_uris(config)

    def _resolve_redirect_uris(self, config: "Config | None" = None) -> "tuple[str, ...]":
        values = self._redirect_uris if config is None else config.resolve(self._redirect_uris)
        if isinstance(values, (str, bytes)):
            raise ContainerError("Authelia redirect_uris must be a sequence of URLs")
        result = []
        for value in values:
            if config is not None:
                value = config.resolve(value)
            if not isinstance(value, str):
                raise ContainerError("Authelia redirect URI must be a string")
            value = str(value)
            if not value:
                continue
            base_url = "" if self._base_url is MISSING else self._base_url
            if config is not None:
                base_url = lazy_load(config.resolve, base_url)
            value = value.format(base_url=base_url)
            if not is_complete_url(value):
                continue
            if not urlsplit(value).hostname:
                continue
            if "#" in value:
                raise ContainerError("Authelia redirect URI must be absolute without a fragment")
            if value not in result:
                result.append(value)
        return tuple(result)
