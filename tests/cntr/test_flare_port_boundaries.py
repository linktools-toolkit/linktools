#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct navigation preserves valid ports independently of optional sites."""
from linktools.cntr.ext import ResolvedSite
from linktools.core import ConfigRef
import pytest

from linktools.cntr.ext import Flare, Nginx, load_nginx_url, load_port_url
from test_declared_integrations import _render_navigation, _site_navigation_manager


@pytest.mark.parametrize("nginx", [False, True])
@pytest.mark.parametrize("port", [0, 65535, 65536])
def test_direct_port_boundary_survives_disabled_public_navigation(nginx: bool, port: int) -> None:
    manager, reads = _site_navigation_manager(
        {"web": Nginx.site("")},
        lambda producer: [
            Flare.public("Public", "web", url=load_nginx_url(ResolvedSite.find(producer, None))),
            Flare.container("Direct", "web", load_port_url(ConfigRef("PORT", type=int, default=0), "ui", https=False)),
        ],
        nginx=nginx,
    )
    producer = manager.containers["app"]

    def get_config(key: str, **kwargs: object) -> "int | str":
        reads.append(key)
        assert key in ("PORT", "HOST")
        return port if key == "PORT" else "host.test"

    producer.get_config = get_config
    producer.env_config.get = get_config
    assert reads == []
    result = _render_navigation(manager)
    assert result["apps.yml"]["links"] == []
    expected = [{"name": "Direct", "icon": "web", "link": "http://host.test:65535/ui",
                 "category": "container"}] if port == 65535 else []
    assert result["bookmarks.yml"]["links"] == expected
    assert reads == (["PORT", "HOST"] if port == 65535 else ["PORT"])
