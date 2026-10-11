#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose a primary site with explicit consumer declarations."""
from typing import TYPE_CHECKING

from ._base import Integration
from ._flare import Flare
from ._nginx import Nginx
from ._urls import load_nginx_url

if TYPE_CHECKING:
    from typing import Any, Mapping, Optional, Sequence, Tuple, Union
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
        public_url: "Optional[str]" = None, cert_domains: "Sequence[str]" = (),
        template_vars: "Optional[Mapping[str, Any]]" = None,
        default_server: bool = False, extensions: "Integrations" = (),
) -> "Tuple[Union[Mapping[str, Integration], Integration], ...]":
    """Declare the primary ``web`` site and explicitly associated extensions.

    Applications with an omitted URL inherit the site's lazy public URL.
    Other URLs and all display metadata belong to the extension declarations.
    """
    result = [{"web": Nginx.site(
        domain, proxy=proxy, template=template, https=https,
        waf=waf, waf_bypass=waf_bypass, auth=auth, auth_bypass=auth_bypass,
        auth_headers=auth_headers, public_url=public_url, cert_domains=cert_domains,
        template_vars=template_vars, default_server=default_server,
    )}]
    for key, declaration in Integration.iter_declarations(extensions):
        if isinstance(declaration, Flare):
            declaration = declaration.with_default_url(load_nginx_url(producer, "web"))
        result.append({key: declaration} if key is not None else declaration)
    return tuple(result)
