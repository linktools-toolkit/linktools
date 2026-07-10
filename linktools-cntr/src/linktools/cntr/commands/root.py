#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from subprocess import SubprocessError
from typing import TYPE_CHECKING

from dulwich.errors import GitProtocolError

from linktools.cli import (
    BaseCommandGroup, CommandGroupRef, SubCommandWrapper, subcommand, subcommand_argument,
)
from linktools.cli.argparse import BooleanOptionalAction, LazyChoices
from linktools.errors import ConfigError, GitError
from ..container import ContainerError
from ..context import EventContext
from ..doctor import Doctor
from ..runtime.compose import ComposeOptions
from . import _shared
from .config import ConfigCommand
from .exec_ import ExecCommand
from .repo import RepoCommand

if TYPE_CHECKING:
    from typing import Any


class Command(BaseCommandGroup):
    """
    Deploy and manage Docker/Podman containers with ease
    """

    @property
    def name(self) -> str:
        return "cntr"

    @property
    def parent(self) -> "CommandGroupRef | str | None":
        return CommandGroupRef(
            id="common",
            name="ct",
            description="Common scripts",
        )

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [
            ContainerError, ConfigError, GitError, SubprocessError, GitProtocolError, OSError, AssertionError,
        ]

    def init_subcommands(self) -> "Any":
        return [
            self,
            SubCommandWrapper(ExecCommand()),
            SubCommandWrapper(ConfigCommand()),
            SubCommandWrapper(RepoCommand()),
        ]

    @subcommand("list", help="list all containers")
    @subcommand_argument("--detail", action="store_true", help="show container detail info")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_list(self, names: "list[str]" = None, detail: bool = False):
        install_containers = _shared.manager.get_installed_containers(resolve=False)
        all_install_containers = _shared.manager.resolve_depend_containers(install_containers)
        # Prefer live state, fall back to persisted when Docker is unavailable
        # so `list` never crashes.
        running_names = set(_shared.manager.running_state.get_effective(install_containers))
        for container in sorted(_shared.manager.containers.values(), key=lambda o: o.order):
            if names and container.name not in names:
                continue
            installed = container in all_install_containers
            added = container in install_containers
            running = container.name in running_names
            if not installed and running:
                style, symbol, label = "yellow bold", "[-]", "pending remove"
            elif not installed:
                style, symbol, label = "dim", "[ ]", None
            elif added and running:
                style, symbol, label = "green bold", "[*]", "added"
            elif added:
                style, symbol, label = "cyan bold", "[+]", "pending install"
            elif running:
                style, symbol, label = "green dim", "[-]", "dependency"
            else:
                style, symbol, label = "cyan dim", "[+]", "pending install, dependency"
            suffix = f" \\[{label}]" if label else ""
            message = f"[{style}]{symbol} {container.name}{suffix}[/]"
            if detail:
                message += f"{os.linesep}    [dim]Enable: {container.enable}[/]"
                message += f"{os.linesep}    [dim]Order: {container.order}[/]"
                message += f"{os.linesep}    [dim]Path: {container.root_path}[/]"
                message += f"{os.linesep}    [dim]Description: {container.description}[/]"
                message += f"{os.linesep}    [dim]Dependencies: \\[{', '.join(container.dependencies)}][/]"
                message += f"{os.linesep}    [dim]Configs: \\[{', '.join(container.configs.keys())}][/]"
            self.logger.info(message, extra={"markup": True})

    @subcommand("add", help="add containers to installed list")
    @subcommand_argument("names", metavar="CONTAINER", nargs="+", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_add(self, names: "list[str]"):
        containers = _shared.manager.add_installed_containers(*names)
        assert containers, "No container added"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Add {', '.join(result)} success")

    @subcommand("remove", help="remove containers from installed list")
    @subcommand_argument("-f", "--force", help="Force remove")
    @subcommand_argument("names", metavar="CONTAINER", nargs="+", help="container name",
                         choices=LazyChoices(_shared.iter_container_names))
    def on_command_remove(self, names: "list[str]", force: bool = False):
        containers = _shared.manager.remove_installed_containers(*names, force=force)
        assert containers, "No container removed"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Remove {', '.join(result)} success")

    @subcommand("up", help="deploy installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_up(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        context = self._make_context(["up", pull and "pull", build and "build"], names)
        services = _shared.manager.compose_runner.collect_services(context)
        options = ComposeOptions(
            build=build,
            pull=pull,
            remove_orphans=context.is_full_containers,
            services=services,
            emit_default_pull=True,
        )

        with _shared.manager.notify_start(context):
            if build:
                _shared.manager.compose_runner.build(context, options)
            _shared.manager.compose_runner.up(context, options)

        with _shared.manager.notify_remove(context):
            pass

        # Record running state only after a successful up.
        _shared.manager.running_state.mark_started(context)

    @subcommand("restart", help="restart installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_restart(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        context = self._make_context(["restart", pull and "pull", build and "build"], names)
        services = _shared.manager.compose_runner.collect_services(context)
        # restart omits the --pull=false / --pull missing defaults that `up`
        # emits, and (unlike `up`/`exec up`/`exec restart`) never includes
        # proxy --build-args.
        options = ComposeOptions(
            build=build,
            pull=pull,
            remove_orphans=context.is_full_containers,
            services=services,
            emit_default_pull=False,
            include_proxy_build_args=False,
        )

        with _shared.manager.notify_stop(context):
            _shared.manager.compose_runner.stop(context, services)

        with _shared.manager.notify_start(context):
            if build:
                _shared.manager.compose_runner.build(context, options)
            _shared.manager.compose_runner.up(context, options)

        with _shared.manager.notify_remove(context):
            pass

        # restart ends with the targets running.
        _shared.manager.running_state.mark_started(context)

    @subcommand("down", help="stop installed containers")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_down(self, names: "list[str]" = None):
        context = self._make_context("down", names)
        services = _shared.manager.compose_runner.collect_services(context)

        with _shared.manager.notify_stop(context):
            _shared.manager.compose_runner.down(context, services)

        with _shared.manager.notify_remove(context):
            pass

        # Record stopped state only after a successful down.
        _shared.manager.running_state.mark_stopped(context)

    @subcommand("doctor", help="read-only environment and security checks (changes nothing)")
    def on_command_doctor(self):
        findings = Doctor(_shared.manager).run()
        for finding in findings:
            self.logger.info(f"[{finding.severity}] {finding.message}")
        self.logger.info("[INFO] No change has been made. Suggestions are kept for compatibility.")

    def _make_context(self, commands, names):
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = _shared.manager.prepare_installed_containers()
        if not names:
            context.target_containers = context.containers
            context.is_full_containers = True
        else:
            context.target_containers = [c for c in context.containers if c.name in names]
            context.is_full_containers = False
        return context
