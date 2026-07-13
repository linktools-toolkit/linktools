#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Container-level command implementations (up/restart/down/config/shell/logs/
mount/umount). The @subcommand-decorated wrapper methods stay on BaseContainer
(command discovery inspects the concrete class namespace) and call into this
module for their bodies."""
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import yaml

from linktools import utils
from linktools.rich import choose, confirm
from ..runtime.compose import ComposeOptions

if TYPE_CHECKING:
    from ..container import BaseContainer


def up(container: "BaseContainer", build: bool = True, pull: bool = False):
    context = container._make_exec_context(["up", pull and "pull", build and "build"])
    services = container.compose_runner.collect_services(context)
    # exec never emitted default --pull flags -> emit_default_pull=False.
    options = ComposeOptions(build=build, pull=pull, services=services, emit_default_pull=False)

    with container.lifecycle.notify_start(context):
        if build:
            container.compose_runner.build(context, options)
        container.compose_runner.up(context, options)
        # Recorded immediately after up succeeds, still inside this `with`
        # (before notify_start's on_started/AFTER_START hooks) -- see
        # operations.ComposeOperations.up's identical comment for why.
        container.running_state.mark_started(context)


def restart(container: "BaseContainer", build: bool = True, pull: bool = False):
    context = container._make_exec_context(["restart", pull and "pull", build and "build"])
    services = container.compose_runner.collect_services(context)
    options = ComposeOptions(build=build, pull=pull, services=services, emit_default_pull=False)

    with container.lifecycle.notify_stop(context):
        container.compose_runner.stop(context, services)
        # If build/up below then fails, persisted state must reflect that
        # the target is actually stopped.
        container.running_state.mark_stopped(context)

    with container.lifecycle.notify_start(context):
        if build:
            container.compose_runner.build(context, options)
        container.compose_runner.up(context, options)
        container.running_state.mark_started(context)


def down(container: "BaseContainer"):
    context = container._make_exec_context("down")
    services = container.compose_runner.collect_services(context)

    with container.lifecycle.notify_stop(context):
        container.compose_runner.down(context, services)
        container.running_state.mark_stopped(context)


def config(container: "BaseContainer"):
    context = container._make_exec_context("config")
    services = container.compose_runner.collect_services(context)
    return container.compose_runner.config(context, services)


def shell(container: "BaseContainer", command: str = None, privileged: bool = False,
          user: str = None, service_name: str = None):
    service = container.choose_service(service_name)

    options = []
    if privileged:
        options.append("--privileged")
    if user:
        options.append("--user")
        options.append(user)

    if not command:
        commands = []
        for sh in ["/bin/zsh", "/bin/fish", "/bin/bash", "/bin/ash", "/bin/sh"]:
            shell_command = [
                "if" if len(commands) == 0 else "elif", "[", "-x", sh, "]", ";",
                "then", sh, ";",
            ]
            commands.extend(shell_command)
        commands.extend(["else", "sh", ";"])
        commands.append("fi")
        commands = ("sh", "-c", utils.list2cmdline(commands))
    else:
        commands = utils.cmdline2list(command)

    return container.runtime.create_docker_process(
        "exec", "-it", *options, service.get("container_name"), *commands
    ).call()


def logs(container: "BaseContainer", follow: bool = True, tail: str = None, timestamps: bool = True,
         since: str = None, until: str = None, service_name: str = None):
    service = container.choose_service(service_name)

    options = []
    if follow:
        options.append("--follow")
    if timestamps:
        options.append("--timestamps")
    if tail:
        options.append("--tail")
        options.append(tail)
    if since:
        options.append("--since")
        options.append(since)
    if until:
        options.append("--until")
        options.append(until)
    return container.runtime.create_docker_process(
        "logs", *options, service.get("container_name")
    ).call()


def mount(container: "BaseContainer", source: str = None, target: str = None,
          permission: str = "rw", service_name: str = None):
    if not source or not target:
        if not source and not target:
            with container.settings.transaction() as settings:
                result = {}
                mount_paths = settings.get("mount_paths") or {}
                for service in container.services.values():
                    container_paths = mount_paths.get(service.get("container_name"), {})
                    if container_paths:
                        result[service.get("container_name")] = list(container_paths.values())
                if not result:
                    container.logger.info("Not found any mount path")
                    return
                container.logger.info(yaml.dump(result))
            return
        if not source:
            container.logger.error("Argument error: source is empty")
        if not target:
            container.logger.error("Argument error: target is empty")
        return

    source_path = Path(os.path.expanduser(source)).absolute()
    target_path = PurePosixPath(target).as_posix()
    if not os.path.exists(source_path):
        container.logger.error(f"{source_path} not exists.")
        return

    service = container.choose_service(service_name)
    with container.settings.transaction() as settings:
        mount_paths = settings.get("mount_paths") or {}
        containers_paths = mount_paths.setdefault(service.get("container_name"), {})
        container_path = f"{source_path}:{target_path}:{permission}"
        if target_path in containers_paths:
            if not confirm(f"{target_path} is mounted: {containers_paths.get(target_path)}, overwrite it?"):
                container.logger.info(f"cancel")
                return
        containers_paths[target_path] = container_path
        settings.set("mount_paths", mount_paths)
        container.logger.info(f"add {container_path}")


def umount(container: "BaseContainer", service_name: str = None):
    service = container.choose_service(service_name)
    with container.settings.transaction() as settings:
        mount_paths = settings.get("mount_paths") or {}
        containers_paths = mount_paths.setdefault(service.get("container_name"), {})
        if not containers_paths:
            container.logger.error("Not found any mount path")
            return
        dest_path = choose(
            "Choose mount path",
            choices=containers_paths
        )
        mount_path = containers_paths.pop(dest_path)
        settings.set("mount_paths", mount_paths)
        container.logger.info(f"remove {mount_path}")
