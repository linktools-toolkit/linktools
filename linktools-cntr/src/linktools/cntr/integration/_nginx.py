#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nginx declarations, lazy domain configuration, and site resolution."""
from typing import TYPE_CHECKING

from ..container import ContainerError
from ._base import Integration
from ._nginx_site import ResolvedSite

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Sequence
    from linktools.core import ConfigResolver, LazyProvider
    from ..container import BaseContainer
    from ..manager import ContainerManager
    from ._flare import FlareLink


class NginxSite(Integration):
    """One nginx site owned by a container and identified by its local ID."""

    consumer = "nginx"
    requires_local_id = True

    def __init__(
            self, server_name: "Any", proxy: "Optional[Any]" = None,
            template: "Optional[Any]" = None, https: "Optional[bool]" = None,
            waf: "Optional[bool]" = None, auth: "Optional[bool]" = None,
            waf_bypass: "Sequence[str]" = (), auth_bypass: "Sequence[str]" = (),
            auth_headers: "Optional[Mapping[str, Any]]" = None,
            auth_rule: "Optional[Mapping[str, Any]]" = None,
            oidc_redirects: "Sequence[str]" = (), url: "Optional[Any]" = None,
            cert_domains: "Sequence[str]" = (),
            vars: "Optional[Mapping[str, Any]]" = None,
            expose: "Optional[FlareLink]" = None,
            *, local_id: str = "web",
    ) -> None:
        self.local_id = local_id
        self.expose = expose
        self.server_name = server_name
        self.proxy = proxy
        self.template = template
        self.https = https
        self.waf = waf
        self.auth = auth
        self.waf_bypass = tuple(waf_bypass) if type(waf_bypass) in (tuple, list) else waf_bypass
        self.auth_bypass = tuple(auth_bypass) if type(auth_bypass) in (tuple, list) else auth_bypass
        self.auth_headers = {} if auth_headers is None else auth_headers
        self.auth_rule = auth_rule
        self.oidc_redirects = tuple(oidc_redirects) if type(oidc_redirects) in (tuple, list) else oidc_redirects
        self.url = url
        self.cert_domains = tuple(cert_domains) if type(cert_domains) in (tuple, list) else cert_domains
        self.vars = {} if vars is None else vars


class Nginx:
    """Nginx declarations, lazy domain configuration, and site resolution."""

    @classmethod
    def domain(cls, container: "BaseContainer", name: "Optional[str]" = None) -> "LazyProvider":
        """Resolve a container's domain from the installed nginx configuration."""
        from linktools.core import LazyProvider

        def get_domain(config: "ConfigResolver") -> str:
            if not container.containers["nginx"].enable:
                return ""
            if not config.get("NGINX_WILDCARD_DOMAIN", type=bool):
                return config.get("NGINX_ROOT_DOMAIN")
            root = config.get("NGINX_ROOT_DOMAIN")
            if root in ("_", "localhost"):
                return root
            if name is None:
                return f"{container.name}.{root}"
            if name.strip() == "":
                return root
            return f"{name}.{root}"

        return LazyProvider(get_domain)

    @classmethod
    def resolve_sites(cls, manager: "ContainerManager") -> "Mapping[tuple[str, str], ResolvedSite]":
        from collections import OrderedDict
        from types import MappingProxyType

        result = OrderedDict()
        # Preserve declared identities even when the optional nginx consumer is absent.
        for producer_name, declarations in manager.integration_snapshot.items():
            producer = manager.containers[producer_name]
            for declaration in declarations:
                if declaration.consumer != "nginx":
                    continue
                local_id = declaration.local_id
                if not isinstance(declaration, NginxSite):
                    raise ContainerError("Invalid nginx site %s/%s: expected NginxSite" % (producer_name, local_id))
                result[(producer_name, local_id)] = ResolvedSite(producer, local_id, declaration)
        return MappingProxyType(result)

    @classmethod
    def site(
            cls, server_name: "Any", proxy: "Optional[Any]" = None,
            template: "Optional[Any]" = None, https: "Optional[bool]" = None,
            waf: "Optional[bool]" = None, auth: "Optional[bool]" = None,
            waf_bypass: "Sequence[str]" = (), auth_bypass: "Sequence[str]" = (),
            auth_headers: "Optional[Mapping[str, Any]]" = None,
            auth_rule: "Optional[Mapping[str, Any]]" = None,
            oidc_redirects: "Sequence[str]" = (), url: "Optional[Any]" = None,
            cert_domains: "Sequence[str]" = (),
            vars: "Optional[Mapping[str, Any]]" = None,
            expose: "Optional[FlareLink]" = None,
            *, local_id: str = "web",
    ) -> NginxSite:
        """Declare a site with a stable producer-local identity."""
        return NginxSite(
            server_name=server_name, proxy=proxy, template=template,
            https=https, waf=waf, auth=auth,
            waf_bypass=waf_bypass, auth_bypass=auth_bypass,
            auth_headers=auth_headers, auth_rule=auth_rule,
            oidc_redirects=oidc_redirects, url=url, cert_domains=cert_domains,
            vars=vars, expose=expose, local_id=local_id,
        )
