#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx declarations, lazy domain configuration, and site resolution."""
import re
from typing import TYPE_CHECKING

from ._base import Integration
from ..errors import ContainerError

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Sequence
    from linktools.core import ConfigResolver, ConfigValue, LazyProvider
    from linktools.types import PathType
    from ..container import BaseContainer


class Nginx(Integration):
    """One nginx site, referenced by its original declaration object."""

    consumer = "nginx"

    @classmethod
    def site(
            cls,
            server_name: "str | ConfigValue",
            proxy: "str | ConfigValue | None" = None,
            template: "PathType | ConfigValue | None" = None,
            https: "bool | ConfigValue | None" = None,
            waf: "bool | ConfigValue | None" = None, waf_bypass: "Sequence[str | ConfigValue] | ConfigValue" = (),
            auth: "bool | ConfigValue | None" = None, auth_bypass: "Sequence[str | ConfigValue] | ConfigValue" = (),
            auth_headers: "Mapping[str, str | ConfigValue] | ConfigValue | None" = None,
            public_url: "str | ConfigValue | None" = None,
            cert_domains: "Sequence[str | ConfigValue] | ConfigValue" = (),
            template_vars: "Optional[Mapping[str, Any]]" = None,
            *,
            default_server: "bool | ConfigValue" = False,
    ) -> "Nginx":
        self = cls()
        self.server_name = server_name
        self.default_server = default_server
        self.proxy = proxy
        self.template = template
        self.https = https
        self.waf = waf
        self.auth = auth
        self.waf_bypass = tuple(waf_bypass) if type(waf_bypass) in (tuple, list) else waf_bypass
        self.auth_bypass = tuple(auth_bypass) if type(auth_bypass) in (tuple, list) else auth_bypass
        self.auth_headers = {} if auth_headers is None else auth_headers
        self.public_url = public_url
        self.cert_domains = tuple(cert_domains) if type(cert_domains) in (tuple, list) else cert_domains
        self.template_vars = {} if template_vars is None else template_vars
        return self

    @classmethod
    def validate_https_domain(cls, value: str) -> None:
        """Reject routing placeholders and non-DNS names before TLS configuration."""
        if not cls.is_dns_domain(value):
            raise ContainerError("Configure NGINX_ROOT_DOMAIN with a DNS domain for HTTPS.")

    @classmethod
    def certificate_domain(cls, value: str) -> "Optional[str]":
        """Resolve an optional certificate name, allowing a leading wildcard."""
        if not value:
            return None
        name = value[2:] if value.startswith("*.") else value
        if not cls.is_dns_domain(name):
            raise ContainerError("Invalid nginx certificate domain: {!r}".format(value))
        return value

    @classmethod
    def is_dns_domain(cls, value: object) -> bool:
        """Distinguish DNS names from defaults and other HTTP routing names."""
        labels = value.split(".") if isinstance(value, str) and len(value) <= 253 else ()
        return (len(labels) > 1 and not labels[-1].isdigit() and
                all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in labels))

    @classmethod
    def domain(cls, container: "BaseContainer", name: "Optional[str]" = None) -> "LazyProvider":
        """Resolve a container's domain from the installed nginx configuration."""
        from linktools.core import LazyProvider

        def get_domain(config: "ConfigResolver") -> str:
            if not container.containers["nginx"].enable:
                return ""
            root = config.get("NGINX_ROOT_DOMAIN")
            if not config.get("NGINX_WILDCARD_DOMAIN", type=bool) or not cls.is_dns_domain(root):
                return root
            if name is None:
                return f"{container.name}.{root}"
            if name.strip() == "":
                return root
            return f"{name}.{root}"

        return LazyProvider(get_domain)

