#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small, side-effect-free recipes for common service access declarations."""
from typing import TYPE_CHECKING

from ._authelia import Authelia
from ._flare import Flare
from ._nginx import Nginx
from ._urls import load_nginx_url, load_port_url

if TYPE_CHECKING:
    from typing import Mapping, Optional, Sequence, Union
    from ..container import BaseContainer
    from ._base import Integration


def endpoint(
    producer: "BaseContainer", key: str, *, name: str, icon: str, desc: str,
    domain: str, proxy: str, auth: "Optional[bool]" = None,
    auth_bypass: "Sequence[str]" = (), waf_bypass: "Sequence[str]" = (),
    path: str = "", direct_port: "Optional[str]" = None,
    oidc_paths: "Sequence[str]" = (), oidc_enabled: bool = True,
) -> "tuple[Union[Mapping[str, Nginx], Integration], ...]":
    """Declare a proxy site, navigation and optional site-relative OIDC callbacks.

    ``path`` selects the public and direct-port navigation page. Callback paths
    are relative to the site root, independently of that navigation page.
    All configuration values remain lazy until their consumers resolve them.
    """
    result = [
        {key: Nginx.site(server_name=domain, proxy=proxy, auth=auth,
                        auth_bypass=auth_bypass, waf_bypass=waf_bypass)},
        Flare.public(name, icon, desc, load_nginx_url(producer, key, path)),
    ]
    if direct_port is not None:
        result.append(Flare.container(name, icon, load_port_url(
            producer, direct_port, path, https=False), desc=desc))
    if oidc_paths:
        result.append(Authelia.oidc(tuple(load_nginx_url(producer, key, part)
            for part in oidc_paths), enabled=oidc_enabled))
    return tuple(result)
