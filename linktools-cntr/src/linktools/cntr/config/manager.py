#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import TYPE_CHECKING

from linktools.core import (
    AliasProvider, ConfigField, LazyProvider, PromptProvider,
)
from linktools.system import get_gid, get_lan_ip, get_uid

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


def build_manager_configs(manager: "ContainerManager") -> "dict[str, Any]":
    """Build the default ContainerManager-level configuration fields."""
    return dict(
        HOST=ConfigField.chain(
            PromptProvider(),
            LazyProvider(lambda r: get_lan_ip()),
        ),
        DOCKER_HOST="/var/run/docker.sock",

        COMPOSE_PROJECT_NAME=manager.name,
        SERVICE_RESTART_POLICY="unless-stopped",
        SERVICE_LOG_DRIVER="json-file",
        SERVICE_LOG_MAX_SIZE="10m",

        DOCKER_USER=ConfigField.chain(
            PromptProvider(cached=True),
            default=os.environ.get("SUDO_USER", manager.user).replace(" ", ""),
        ),
        DOCKER_UID=ConfigField(provider=LazyProvider(
            lambda r: get_uid(r.get("DOCKER_USER", type=str)),
        )),
        DOCKER_GID=ConfigField(provider=LazyProvider(
            lambda r: get_gid(r.get("DOCKER_USER", type=str)),
        )),
        DOCKER_TYPE=ConfigField.chain(
            AliasProvider("CONTAINER_TYPE"),
            PromptProvider(choices=["docker", "docker-rootless"], cached=True),
            default="docker",
        ) if manager.system == "linux" and os.getuid() != 0 else ConfigField(default="docker"),

        DOCKER_APP_PATH=ConfigField.chain(
            PromptProvider(cached=True), cast="path", default=str(manager.data_path.joinpath("app")),
        ),
        DOCKER_APP_DATA_PATH=ConfigField(cast="path", provider=AliasProvider("DOCKER_APP_PATH")),
        DOCKER_USER_DATA_PATH=ConfigField.chain(
            PromptProvider(cached=True), cast="path", default=str(manager.data_path.joinpath("user_data")),
        ),
        DOCKER_DOWNLOAD_PATH=ConfigField.chain(
            PromptProvider(cached=True), cast="path", default=str(manager.data_path.joinpath("download")),
        ),
    )
