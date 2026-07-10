#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""docker / podman / docker-compose process creation (refactor spec Phase 4).

Extracted verbatim from ContainerManager so the manager delegates to it. The
docker / docker-rootless / podman branching, privilege (sudo) escalation, proxy
env preservation, compose --file ordering and project name are all unchanged.
"""
import os
from typing import TYPE_CHECKING

from linktools.runtime import popen

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any
    from linktools.runtime import Process
    from ..container import BaseContainer
    from ..manager import ContainerManager


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
