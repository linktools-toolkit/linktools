#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public nginx integration declaration."""
from typing import Any, Mapping, Optional, Sequence


class NginxSite:
    """One nginx site owned by a container and identified by its local ID."""

    def __init__(
            self, server_name: Any, proxy: "Optional[Any]" = None,
            template: "Optional[Any]" = None, https: "Optional[bool]" = None,
            waf: "Optional[bool]" = None, auth: "Optional[bool]" = None,
            waf_bypass: "Sequence[str]" = (), auth_bypass: "Sequence[str]" = (),
            auth_headers: "Optional[Mapping[str, Any]]" = None,
            auth_rule: "Optional[Mapping[str, Any]]" = None,
            oidc_redirects: "Sequence[str]" = (), url: "Optional[Any]" = None,
            cert_domains: "Sequence[str]" = (),
            vars: "Optional[Mapping[str, Any]]" = None,
    ) -> None:
        self.server_name = server_name
        self.proxy = proxy
        self.template = template
        self.https = https
        self.waf = waf
        self.auth = auth
        self.waf_bypass = tuple(waf_bypass)
        self.auth_bypass = tuple(auth_bypass)
        self.auth_headers = dict(auth_headers or {})
        self.auth_rule = dict(auth_rule) if auth_rule is not None else None
        self.oidc_redirects = tuple(oidc_redirects)
        self.url = url
        self.cert_domains = tuple(cert_domains)
        self.vars = dict(vars or {})
