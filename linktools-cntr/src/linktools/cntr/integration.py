#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure declarations shared by container integration producers and consumers."""
from typing import TYPE_CHECKING, Iterable, Mapping, Union

from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any, Optional, Sequence


class Integration:
    """Nominal marker for a consumer-specific integration declaration."""


Integrations = Mapping[str, Union[Mapping[str, Integration], Iterable[Integration]]]


class ExposeCategory:

    def __init__(self, name: str, desc: str) -> None:
        self.name = name
        self.desc = desc

    def __call__(self, name: str, icon: str, desc: str, url: "str | None" = MISSING) -> "ExposeLink":
        return ExposeLink(self, name, icon, desc or name, url)


class ExposeLink(Integration):
    public = ExposeCategory("public", "Public")
    private = ExposeCategory("private", "Private")
    container = ExposeCategory("container", "Internal")
    other = ExposeCategory("other", "Tools")

    def __init__(self, category: "ExposeCategory", name: str, icon: str, desc: str, url: "str | None" = MISSING) -> None:
        self.category = category
        self.name = name
        self.icon = icon
        self.desc = desc
        self._url = url

    @property
    def url(self) -> "str | None":
        if self._url is MISSING or not self._url:
            return None
        return str(self._url)

    def with_default_url(self, url: str) -> "ExposeLink":
        """Bind only an omitted URL, preserving explicit empty/None values."""
        if self._url is not MISSING:
            return self
        return ExposeLink(self.category, self.name, self.icon, self.desc, url)

    @property
    def is_valid(self) -> bool:
        return not not self.url


class NginxSite(Integration):
    """One nginx site owned by a container and identified by its local ID."""

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
            expose: "Optional[ExposeLink]" = None,
    ) -> None:
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
