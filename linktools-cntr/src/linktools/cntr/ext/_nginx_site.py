#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-local resolution of nginx integration declarations."""
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from linktools import utils
from linktools.decorator import cached_property
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any, Optional
    from linktools.core import Config, ConfigLazy
    from ..container import BaseContainer
    from ..manager import ContainerManager
    from ._nginx import Nginx
    from ._base import Integration


class ResolvedSite:
    """A command-local lazy view shared by navigation and configuration consumers."""

    @classmethod
    def collect(cls, manager: "ContainerManager") -> "Mapping[Nginx, ResolvedSite]":
        """Resolve original declaration objects without evaluating lazy fields."""
        from collections import OrderedDict
        from ._nginx import Nginx
        from ..errors import ContainerError

        result = OrderedDict()
        for producer_name, declarations in manager.integration_snapshot.items():
            producer = manager.containers[producer_name]
            for declaration in declarations:
                if declaration.consumer != "nginx":
                    continue
                if not isinstance(declaration, Nginx):
                    raise ContainerError("Invalid nginx site in %s: expected Nginx" % producer_name)
                if declaration in result:
                    raise ContainerError("Repeated nginx declaration in " + producer_name)
                result[declaration] = cls(producer, declaration, len(result))
        return MappingProxyType(result)

    @classmethod
    def find(cls, producer: "BaseContainer",
             reference: "Nginx | tuple[Integration, ...] | None" = None) -> "ConfigLazy":
        """Declare a lazy lookup of an original site or endpoint.

        Omission requires one declared site. Lookup and validation happen on each
        Config.resolve call against the producer's command-local snapshot.
        """
        from linktools.core import ConfigLazy

        def resolve(config: "Config") -> "ResolvedSite":
            from ._nginx import Nginx
            from ..errors import ContainerError

            target = reference
            sites = producer.manager.containers["nginx"].sites
            if target is None:
                matches = [site for site in sites.values() if site.producer is producer]
            else:
                if isinstance(target, tuple):
                    declarations = [item for item in target if isinstance(item, Nginx)]
                    if len(declarations) != 1:
                        raise ContainerError("Endpoint reference must contain exactly one nginx site")
                    target = declarations[0]
                if not isinstance(target, Nginx):
                    raise ContainerError("Expected an original nginx site or endpoint declaration")
                site = sites.get(target)
                matches = [site] if site is not None and site.producer is producer else []
            if len(matches) != 1:
                raise ContainerError("Expected one declared nginx site in %s; reference the original endpoint" % producer.name)
            return matches[0]

        return ConfigLazy(resolve)

    def __init__(self, producer: "BaseContainer", declaration: "Nginx", position: int) -> None:
        self.producer = producer
        self.var_name = "s_" + str(position)
        self._declaration = declaration

    def _error(self, message: str) -> None:
        from ..errors import ContainerError
        raise ContainerError("Nginx site in %s: %s" % (self.producer.name, message))

    def _text(self, value: "Any", field: str, optional: bool = False) -> "Optional[str]":
        if value is None and optional:
            return None
        if value is MISSING or not isinstance(value, str):
            self._error(field + " must be a string")
        return str(value)

    @cached_property
    def server_name(self) -> str:
        return self._text(self.producer.env_config.resolve(self._declaration.server_name), "server_name")

    @cached_property
    def enabled(self) -> bool:
        return "nginx" in self.producer.manager.integration_snapshot and bool(self.server_name)

    @cached_property
    def default_server(self) -> bool:
        return bool(self.producer.env_config.resolve(self._declaration.default_server)) if self.enabled else False

    @cached_property
    def literal_domain(self) -> "Optional[str]":
        value = self.server_name
        return value if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value) else None

    def _switch(self, name: str) -> bool:
        if not self.enabled:
            return False
        requested = self.producer.env_config.resolve(getattr(self._declaration, name))
        if requested is not None and not isinstance(requested, bool):
            self._error(name + " must be None or a boolean")
        if requested is not None and not bool(requested):
            return False
        global_value = self.producer.get_config("NGINX_%s_ENABLE" % name.upper(), type=bool)
        if requested is not None and bool(requested) and not global_value:
            self._error(name + " explicitly requires a disabled global capability")
        result = global_value if requested is None else bool(requested)
        provider = {"waf": "safeline", "auth": "authelia"}.get(name)
        if result and provider and provider not in self.producer.manager.integration_snapshot:
            self._error(name + " requires installed " + provider)
        return bool(result)

    @cached_property
    def https(self) -> bool:
        return self._switch("https")

    @cached_property
    def waf(self) -> bool:
        return self._switch("waf")

    @cached_property
    def auth(self) -> bool:
        value = self._switch("auth")
        if value and not self.https:
            self._error("Authelia authentication requires HTTPS")
        return value

    @cached_property
    def public_url(self) -> str:
        return self.get_url()

    def get_url(self, default: str = MISSING) -> str:
        """Resolve a public URL, optionally leaving a non-concrete identity unset."""
        if not self.enabled:
            return ""
        explicit = self.producer.env_config.resolve(self._declaration.public_url)
        if explicit is not None:
            return self._text(explicit, "public_url")
        if self.literal_domain is None:
            if default is not MISSING:
                return default
            self._error("requires an explicit public URL")
        https = self.https
        port = self.producer.get_config("NGINX_HTTPS_PORT" if https else "NGINX_HTTP_PORT")
        return utils.make_url("https" if https else "http", self.literal_domain, port)

    @cached_property
    def template(self) -> "Optional[str]":
        if not self.enabled:
            return None
        value = self.producer.env_config.resolve(self._declaration.template)
        if value is MISSING:
            self._error("template must be a path or None")
        return None if value is None else str(value)

    @cached_property
    def proxy(self) -> "Optional[str]":
        return self._text(self.producer.env_config.resolve(self._declaration.proxy), "proxy", optional=True) if self.enabled else None

    def _sequence(self, field: str, active: bool = True) -> "tuple[str, ...]":
        if not self.enabled or not active:
            return ()
        values = self.producer.env_config.resolve(getattr(self._declaration, field))
        if isinstance(values, (str, bytes)):
            self._error(field + " must be a sequence of strings")
        try:
            return tuple(self._text(self.producer.env_config.resolve(value), field) for value in values)
        except TypeError:
            self._error(field + " must be a sequence of strings")

    def _mapping(self, field: str, resolve: bool = True) -> "Mapping":
        if not self.enabled:
            return MappingProxyType({})
        value = getattr(self._declaration, field)
        if resolve:
            value = self.producer.env_config.resolve(value)
        if not isinstance(value, Mapping):
            self._error(field + " must be a mapping")
        return MappingProxyType(self.producer.env_config.resolve(dict(value)) if resolve else dict(value))

    @cached_property
    def waf_bypass(self) -> "tuple[str, ...]":
        return self._sequence("waf_bypass", self.waf)

    @cached_property
    def auth_bypass(self) -> "tuple[str, ...]":
        return self._sequence("auth_bypass", self.auth)

    @cached_property
    def auth_headers(self) -> "Mapping[str, str]":
        if not self.auth:
            return MappingProxyType({})
        return MappingProxyType({self._text(k, "auth_headers name"): self._text(v, "auth_headers value")
                                 for k, v in self._mapping("auth_headers").items()})

    @cached_property
    def cert_domains(self) -> "tuple[str, ...]":
        return self._sequence("cert_domains")

    @cached_property
    def template_vars(self) -> "Mapping":
        return self._mapping("template_vars", resolve=False)

    def resolve(self) -> "ResolvedSite":
        if not self.enabled:
            return self
        if not self.template and not self.proxy:
            self._error("default proxy template requires a nonempty proxy")
        for field in ("default_server", "https", "waf", "auth", "waf_bypass", "auth_bypass", "auth_headers",
                      "cert_domains", "template_vars"):
            getattr(self, field)
        return self
