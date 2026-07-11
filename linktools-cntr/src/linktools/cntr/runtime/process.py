#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""docker / docker-compose process creation, and host file permission
operations (chown/chmod) for paths bind-mounted into containers."""
import os
import shutil
from typing import TYPE_CHECKING

from linktools.runtime import popen
from linktools.system import get_gid, get_system, get_uid

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any
    from linktools.runtime import Process
    from linktools.types import PathType
    from ..container import BaseContainer
    from ..manager import ContainerManager


def _is_chown_supported(system: str = None) -> bool:
    """Docker Desktop's VM-backed bind mounts (macOS/Windows) don't reflect
    host-side ownership/permission changes inside the container; only Linux
    bind mounts do."""
    return (system or get_system()) == "linux"


_DEFAULT_DOCKER_HOST = "/var/run/docker.sock"


def _docker_host_args(host: "str | None") -> "list[str]":
    """Explicit connection args for a configured DOCKER_HOST.

    Only emitted when DOCKER_HOST was actually changed away from the built-in
    default -- otherwise docker's own default connection resolution (e.g.
    docker-rootless's active `docker context`) is left alone, since the
    built-in default string doesn't itself describe where that resolves to.
    """
    if not host or host == _DEFAULT_DOCKER_HOST:
        return []
    if "://" not in host:
        host = f"unix://{host}"
    return ["-H", host]


class RuntimeProcessFactory:
    """Create docker/compose subprocesses behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def create_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs,
    ) -> "Process":
        # Private kwarg, not part of the public facade signature: read-only
        # background queries (list/status, doctor) pass sudo_non_interactive=True
        # so a missing sudo policy fails fast instead of blocking on a
        # password prompt; up/restart/down keep the interactive default.
        sudo_non_interactive = kwargs.pop("sudo_non_interactive", False)
        if privilege:
            if self.manager.system in ("darwin", "linux") and self.manager.uid != 0:
                proxy_keys = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
                preserve_keys = [*[e.lower() for e in proxy_keys], *[e.upper() for e in proxy_keys]]
                preserve_env = [key for key in preserve_keys if key in os.environ]
                sudo_args = ["sudo"]
                if sudo_non_interactive:
                    sudo_args.append("-n")
                if preserve_env:
                    sudo_args.append(f"--preserve-env={','.join(preserve_env)}")
                sudo_args.extend(args)
                return popen(*sudo_args, **kwargs)
        return popen(*args, **kwargs)

    def create_docker_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs,
    ) -> "Process":
        commands = []
        host = self.manager.env_config.get("DOCKER_HOST", type=str, default=None)
        if self.manager.container_type in ("docker", "docker-rootless"):
            commands.extend(["docker"])
            commands.extend(_docker_host_args(host))
            if privilege is None:
                privilege = self.manager.container_type == "docker"
        elif self.manager.container_type == "podman":
            raise ContainerError(
                "Podman is no longer supported. Use DOCKER_TYPE=docker or docker-rootless."
            )
        else:
            raise ContainerError(f"Invalid container type: {self.manager.container_type}")
        return self.create_process(*commands, *args, privilege=privilege, **kwargs)

    def create_docker_compose_process(
            self,
            containers: "Iterable[BaseContainer]",
            *args: str,
            privilege: bool = None,
            **kwargs: "Any",
    ) -> "Process":
        options = []
        for container in containers:
            path = container.get_docker_compose_file()
            if path and os.path.exists(path):
                options.extend(["--file", path])
        if not options:
            # Without any --file, docker compose falls back to searching
            # the current working directory for a compose file --
            # if the targeted containers produced none, that search could hit
            # a completely unrelated project instead of failing loudly.
            raise ContainerError("No Docker Compose file was generated for the targeted containers")
        options.extend(["--project-name", self.manager.project_name])
        return self.create_docker_process("compose", *options, *args, privilege=privilege, **kwargs)

    def chown(self, path: "PathType", user: str, recursive: bool = False) -> None:
        manager = self.manager
        path = manager.env_config.cast(path, type="path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if not _is_chown_supported(manager.system):
            manager.logger.debug(f"Skip chown of {path} on {manager.system}")
            return
        if not shutil.which("chown"):
            manager.logger.debug("Command `chown` not found")
            return
        args = ["chown"]
        if recursive:
            args.append("-R")
        uid, gid = get_uid(user), get_gid(user)
        args.extend([f"{uid}:{gid}", str(path)])
        try:
            stat = os.stat(path)
            # A RuntimeProcessFactory subclass overriding create_process still
            # governs privilege here, since this calls back into `self`, not
            # a separate manager-level wrapper.
            self.create_process(
                *args,
                privilege=manager.uid != stat.st_uid or manager.uid != uid
            ).check_call()
        except Exception as e:
            manager.logger.warning(f"Failed to chown of {path}")
            raise e

    def chmod(self, path: "PathType", mode: int = 0o755, recursive: bool = False) -> None:
        manager = self.manager
        path = manager.env_config.cast(path, type="path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if not _is_chown_supported(manager.system):
            manager.logger.debug(f"Skip chmod of {path} on {manager.system}")
            return
        if not shutil.which("chmod"):
            manager.logger.debug("Command `chmod` not found")
            return
        args = ["chmod"]
        if recursive:
            args.append("-R")
        args.extend([oct(mode)[2:], str(path)])
        try:
            stat = os.stat(path)
            self.create_process(
                *args,
                privilege=manager.uid != stat.st_uid
            ).check_call()
        except Exception as e:
            manager.logger.warning(f"Failed to chmod of {path}")
            raise e
