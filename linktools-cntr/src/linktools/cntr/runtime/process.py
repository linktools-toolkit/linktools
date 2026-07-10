#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""docker / podman / docker-compose process creation, and host file
permission operations (chown/chmod) for paths bind-mounted into containers."""
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


class RuntimeProcessFactory:
    """Create docker/podman/compose subprocesses behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def create_process(
            self,
            *args,
            privilege: bool = None,
            **kwargs,
    ) -> "Process":
        if privilege:
            if self.manager.system in ("darwin", "linux") and self.manager.uid != 0:
                proxy_keys = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
                preserve_keys = [*[e.lower() for e in proxy_keys], *[e.upper() for e in proxy_keys]]
                preserve_env = [key for key in preserve_keys if key in os.environ]
                sudo_args = ["sudo"]
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
        if self.manager.container_type in ("docker", "docker-rootless"):
            commands.extend(["docker"])
            if privilege is None:
                privilege = self.manager.container_type == "docker"
        elif self.manager.container_type == "podman":
            commands.extend(["podman"])
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
            # Route through manager.create_process (not self.create_process) so a
            # ContainerManager subclass overriding it still governs privilege here.
            manager.create_process(
                *args,
                privilege=manager.uid != stat.st_uid or manager.uid != uid
            ).check_call()
        except Exception as e:
            manager.logger.warning(f"Failed to chown of {path}")
            raise e

    def chmod(self, path: "PathType", mode: int = 0o755, recursive: bool = False) -> None:
        manager = self.manager
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
            manager.create_process(
                *args,
                privilege=manager.uid != stat.st_uid
            ).check_call()
        except Exception as e:
            manager.logger.warning(f"Failed to chmod of {path}")
            raise e
