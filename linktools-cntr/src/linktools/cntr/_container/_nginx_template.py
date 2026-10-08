#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict, single-pass rendering for native nginx location templates."""
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, PrefixLoader, StrictUndefined, TemplateError

if TYPE_CHECKING:
    from typing import Any
    from ..container import BaseContainer


def render_nginx_template(
        container: "BaseContainer", nginx: "BaseContainer",
        source: "Any", site: "Any",
) -> str:
    """Render one location template with unambiguous local/nginx namespaces."""
    source = Path(source).absolute()
    nginx_root = Path(nginx.get_source_path("templates")).absolute()
    environment = Environment(
        loader=PrefixLoader({
            "local": FileSystemLoader(str(source.parent)),
            "nginx": FileSystemLoader(str(nginx_root)),
        }),
        undefined=StrictUndefined,
        autoescape=False,
    )
    try:
        template_name = "nginx/" + source.relative_to(nginx_root).as_posix()
    except ValueError:
        template_name = "local/" + source.name
    try:
        return environment.get_template(template_name).render(
            site=site,
            container=container,
            nginx=nginx,
            config=container.env_config,
            vars=site.vars,
        )
    except TemplateError as exc:
        from ..container import ContainerTemplateError
        raise ContainerTemplateError(
            f"Invalid nginx template {source} for {container.name}/{getattr(site, 'local_id', '?')}: {exc}"
        ) from exc
