#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jinja rendering for a container's Compose/Dockerfile/config templates."""
import os
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, TemplateError

from linktools import utils
from linktools.runtime import lazy_load
from ...capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType
    from ..container import BaseContainer


def render_template(container: "BaseContainer", source: "PathType", destination: "PathType" = None, **kwargs: "Any"):
    config = container.env_config

    def mkdir(path: "PathType") -> str:
        path = config.cast(path, type="path")
        container.add_start_hook(("mkdir", str(path)),
                                 lambda: os.makedirs(path, mode=0o755, exist_ok=True),
                                 name="mkdir", order=100, source="builtin")
        return path

    def chown(path: "PathType", user: str = None, recursive: bool = False) -> str:
        path = config.cast(path, type="path")
        if user:
            container.add_start_hook(("chown", str(path), str(user), bool(recursive)),
                                     lambda: container.runtime.chown(path, user, recursive=recursive),
                                     name="chown", order=200, source="builtin")
        return path

    def chmod(path: "PathType", mode: int = 0o755, recursive: bool = False) -> str:
        path = config.cast(path, type="path")
        container.add_start_hook(("chmod", str(path), int(mode), bool(recursive)),
                                 lambda: container.runtime.chmod(path, mode, recursive=recursive),
                                 name="chmod", order=300, source="builtin")
        return path

    context = {
        key: container.get_config_later(key)
        for key in config.keys()
    }

    context.update(
        DEBUG=container.manager.debug,

        SOURCE_PATH=lazy_load(container.get_source_path),
        APP_PATH=lazy_load(container.get_app_path),
        APP_DATA_PATH=lazy_load(container.get_app_data_path),

        manager=container.manager,
        container=container,
        containers=container.containers,
        config=config,
        user=container.user,
        docker_user=container.get_config_later("DOCKER_USER"),

        utils=utils,
        mkdir=mkdir,
        chown=chown,
        chmod=chmod,
    )

    context.update(kwargs)

    # Local templates intentionally take precedence over shared snippets.
    environment = Environment(
        loader=FileSystemLoader([
            Path(source).parent,
            __cap_cntr__.get_asset_path("containers-snippets")
        ])
    )
    environment.filters.update(
        mkdir=mkdir,
        chown=chown,
        chmod=chmod,
    )

    try:
        container.logger.debug(f"{container} render template {source} to {destination or 'memory'}")
        template = environment.from_string(utils.read_file(source, text=True))
        result = template.render(context)
        if destination:
            utils.write_file(destination, result)
        return result

    except TemplateError as e:
        from ..container import ContainerTemplateError
        raise ContainerTemplateError(e)
