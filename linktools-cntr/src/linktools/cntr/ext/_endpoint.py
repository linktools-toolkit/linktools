#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose a primary site with explicit consumer declarations."""
from copy import copy
from typing import TYPE_CHECKING

from linktools.types import MISSING

from ._base import Integration
from ._authelia import Authelia
from ._flare import Flare
from ._nginx import Nginx
from ._urls import load_nginx_url

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Sequence, Tuple
    from linktools.types import PathType
    from ..container import BaseContainer
    from ._base import Integrations


def endpoint(
        producer: "BaseContainer", *, domain: str,
        proxy: "Optional[str]" = None, template: "Optional[PathType]" = None,
        https: "Optional[bool]" = None,
        waf: "Optional[bool]" = None, waf_bypass: "Sequence[str]" = (),
        auth: "Optional[bool]" = None, auth_bypass: "Sequence[str]" = (),
        auth_headers: "Optional[Mapping[str, str]]" = None,
        cert_domains: "Sequence[str]" = (),
        template_vars: "Optional[Mapping[str, Any]]" = None,
        default_server: bool = False,
        extensions: "Integrations" = (),
) -> "Tuple[Integration, ...]":
    """Declare a site and explicitly associated extensions.

    Applications with an omitted URL inherit the site's lazy public URL.
    Other URLs and all display metadata belong to the extension declarations.
    Explicit relative URL references are resolved lazily against this site.
    """
    site = Nginx.site(
        domain, proxy=proxy, template=template, https=https,
        waf=waf, waf_bypass=waf_bypass, auth=auth, auth_bypass=auth_bypass,
        auth_headers=auth_headers, cert_domains=cert_domains,
        template_vars=template_vars, default_server=default_server,
    )
    base_url = load_nginx_url(producer, site)
    result = [site]
    for declaration in Integration.iter_declarations(extensions):
        if isinstance(declaration, Flare):
            declaration = declaration.with_base_url(base_url)
        elif isinstance(declaration, Authelia):
            if declaration.site is None:
                declaration = copy(declaration)
                declaration.site = site
            elif declaration.site is MISSING:
                declaration = declaration.with_base_url(base_url)
        result.append(declaration)
    return tuple(result)
