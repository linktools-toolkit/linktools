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
    from ..container import BaseContainer
    from ._flare import FlareLink
    from ._nginx import NginxSite


class ResolvedSite:
    """A command-local lazy view shared by navigation and configuration consumers."""

    def __init__(self, producer: "BaseContainer", local_id: str, declaration: "NginxSite") -> None:
        self.producer = producer
        self.local_id = local_id
        self.identity = (producer.name, local_id)
        self.file_id = "s_" + producer.name.encode("utf-8").hex() + "_" + local_id.encode("utf-8").hex()
        self.var_name = self.file_id
        self._declaration = declaration

    def _error(self, message: str) -> None:
        from ..container import ContainerError
        raise ContainerError("Nginx site %s/%s: %s" % (self.producer.name, self.local_id, message))

    def _text(self, value: "Any", field: str, optional: bool = False) -> "Optional[str]":
        if value is None and optional:
            return None
        if value is MISSING or not isinstance(value, str):
            self._error(field + " must be a string")
        return str(value)

    @cached_property
    def expose(self) -> "Optional[FlareLink]":
        from ._flare import FlareLink
        from linktools.runtime import lazy_load

        value = self._declaration.expose
        if value is None:
            return None
        if not isinstance(value, FlareLink):
            self._error("expose must be a FlareLink or None")
        return value.with_default_url(lazy_load(lambda: self.url))

    @cached_property
    def server_name(self) -> str:
        return self._text(self._declaration.server_name, "server_name")

    @cached_property
    def enabled(self) -> bool:
        return "nginx" in self.producer.manager.integration_snapshot and bool(self.server_name)

    @cached_property
    def default(self) -> bool:
        return bool(self._declaration.default) if self.enabled else False

    @cached_property
    def literal_domain(self) -> "Optional[str]":
        value = self.server_name
        return value if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value) else None

    def _switch(self, name: str) -> bool:
        if not self.enabled:
            return False
        requested = getattr(self._declaration, name)
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
    def url(self) -> str:
        if not self.enabled:
            return ""
        explicit = self._declaration.url
        if explicit is not None:
            return self._text(explicit, "url")
        if self.literal_domain is None:
            self._error("requires an explicit public URL")
        https = self.https
        port = self.producer.get_config("NGINX_HTTPS_PORT" if https else "NGINX_HTTP_PORT")
        return utils.make_url("https" if https else "http", self.literal_domain, port)

    @cached_property
    def template(self) -> "Optional[str]":
        if not self.enabled:
            return None
        value = self._declaration.template
        if value is MISSING:
            self._error("template must be a path or None")
        return None if value is None else str(value)

    @cached_property
    def proxy(self) -> "Optional[str]":
        return self._text(self._declaration.proxy, "proxy", optional=True) if self.enabled else None

    def _sequence(self, field: str, active: bool = True) -> "tuple[str, ...]":
        if not self.enabled or not active:
            return ()
        values = getattr(self._declaration, field)
        if isinstance(values, (str, bytes)):
            self._error(field + " must be a sequence of strings")
        try:
            return tuple(self._text(value, field) for value in values)
        except TypeError:
            self._error(field + " must be a sequence of strings")

    def _mapping(self, field: str) -> "Mapping":
        if not self.enabled:
            return MappingProxyType({})
        value = getattr(self._declaration, field)
        if not isinstance(value, Mapping):
            self._error(field + " must be a mapping")
        return MappingProxyType(dict(value))

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
    def auth_rule(self) -> "Optional[Mapping]":
        if not self.auth or self._declaration.auth_rule is None:
            return None
        rule = dict(self._mapping("auth_rule"))
        if "domain" not in rule and "domain_regex" not in rule:
            if self.literal_domain is None:
                self._error("auth_rule requires domain or domain_regex for nonliteral server_name")
            rule["domain"] = self.literal_domain
        return MappingProxyType(rule)

    @cached_property
    def oidc_redirects(self) -> "tuple[str, ...]":
        values = self._sequence("oidc_redirects")
        if values and "authelia" not in self.producer.manager.integration_snapshot:
            self._error("OIDC redirects require installed authelia")
        result = []
        for value in values:
            parsed = urlsplit(value)
            if "#" in value or value.startswith("//"):
                self._error("invalid OIDC redirect URI")
            if parsed.scheme:
                resolved = value
            elif value == "" or value.startswith("/"):
                base = self.url
                base_parts = urlsplit(base)
                if (not base_parts.scheme or not base_parts.netloc or "{{" in base
                        or "}}" in base or base_parts.fragment):
                    self._error("relative OIDC redirect requires a concrete public URL")
                resolved = base if value == "" else urlunsplit((base_parts.scheme, base_parts.netloc,
                                                                parsed.path, parsed.query, ""))
            else:
                self._error("OIDC redirect must be absolute, empty, or an absolute path")
            if resolved not in result:
                result.append(resolved)
        return tuple(result)

    @cached_property
    def cert_domains(self) -> "tuple[str, ...]":
        return self._sequence("cert_domains")

    @cached_property
    def vars(self) -> "Mapping":
        return self._mapping("vars")

    def resolve(self) -> "ResolvedSite":
        if not self.enabled:
            return self
        if not self.template and not self.proxy:
            self._error("default proxy template requires a nonempty proxy")
        for field in ("default", "https", "waf", "auth", "waf_bypass", "auth_bypass", "auth_headers",
                      "auth_rule", "oidc_redirects", "cert_domains", "vars"):
            getattr(self, field)
        if self.producer.name == "authelia" and not self.https:
            self._error("Authelia public site requires HTTPS")
        return self
