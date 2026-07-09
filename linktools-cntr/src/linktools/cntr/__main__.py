#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : container.py 
@time    : 2024/3/21
@site    : https://github.com/ice-black-tea
@software: PyCharm 

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
import os
from subprocess import SubprocessError
from typing import TYPE_CHECKING

import yaml
from dulwich.errors import GitProtocolError

from linktools.cli import BaseCommand, subcommand, SubCommandWrapper, subcommand_argument, SubCommandGroup, BaseCommandGroup, CommandParser, CommandGroupRef
from linktools.cli.argparse import KeyValueAction, BooleanOptionalAction, ArgParseComplete, LazyChoices
from linktools.core import environ
from linktools.core import ConfigField
from linktools.rich import confirm, choose
from linktools.errors import ConfigError, GitError
from .container import ContainerError
from .context import EventContext
from .manager import ContainerManager

if TYPE_CHECKING:
    from typing import Any
    from argparse import Namespace
    from linktools.cli import SubCommand

manager = ContainerManager(environ)


def _iter_container_names():
    return [container.name for container in manager.containers.values()]


def _iter_installed_container_names():
    return [container.name for container in manager.get_installed_containers()]


class RepoCommand(BaseCommandGroup):
    """
    manage container repository
    """

    @property
    def name(self):
        return "repo"

    @subcommand("list", help="list repositories")
    def on_command_list(self):
        repos = manager.get_all_repos()
        for key, value in repos.items():
            data = {key: value}
            self.logger.info(
                yaml.dump(data, sort_keys=False).strip()
            )

    @subcommand("add", help="add repository")
    @subcommand_argument("url", help="repository url")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force add")
    def on_command_add(self, url: str, branch: str = None, force: bool = False):
        manager.add_repo(url, branch=branch, force=force)

    @subcommand("update", help="update repositories")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force update")
    def on_command_update(self, branch: str = None, force: bool = False):
        manager.update_repos(branch=branch, reset=force)

    @subcommand("remove", help="remove repository")
    @subcommand_argument("url", nargs="?", help="repository url")
    def on_command_remove(self, url: str = None):
        repos = list(manager.get_all_repos().keys())
        if not repos:
            raise ContainerError("No repository found")

        if url is None:
            repo = choose("Choose repository you want to remove", repos)
            if not confirm(f"Remove repository `{repo}`?", default=False):
                raise ContainerError("Canceled")
            manager.remove_repo(repo)

        elif url in repos:
            if not confirm(f"Remove repository `{url}`?", default=False):
                raise ContainerError("Canceled")
            manager.remove_repo(url)

        else:
            raise ContainerError(f"Repository `{url}` not found.")


class ConfigCommand(BaseCommand):
    """
    manage container configs
    """

    @property
    def name(self):
        return "config"

    def init_arguments(self, parser: "CommandParser") -> None:
        self.add_subcommands(parser)

    def run(self, args: "Namespace") -> "int | None":
        subcommand = self.parse_subcommand(args)
        if subcommand:
            return subcommand.run(args)
        containers = manager.prepare_installed_containers()
        return manager.create_docker_compose_process(
            containers,
            "config",
            privilege=False,
        ).check_call()

    @subcommand("set", help="set container configs")
    @subcommand_argument("configs", action=KeyValueAction, nargs="+", help="container config key=value")
    def on_command_set(self, configs: "dict[str, str]"):
        for key, value in configs.items():
            manager.env_config.persist(key, value)
        for key in sorted(configs.keys()):
            value = manager.env_config.get(key)
            self.logger.info(f"{key}: {value}")

    @subcommand("unset", help="remove container configs")
    @subcommand_argument("configs", action=KeyValueAction, metavar="KEY", nargs="+", help="container config keys")
    def on_command_remove(self, configs: "dict[str, str]"):
        for key in configs.keys():
            manager.env_config.remove(key)
        self.logger.info(f"Unset {', '.join(configs.keys())} success")

    @subcommand("list", help="list container configs")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_iter_installed_container_names))
    @subcommand_argument("-d", "--with-dependencies", action="store_true", default=False,
                         help="include configs from dependency containers")
    def on_command_list(self, names: "list[str]", with_dependencies: bool = False):
        containers = manager.prepare_installed_containers()
        target_containers = [c for c in containers if c.name in names] if names else containers
        if with_dependencies and names:
            target_containers = manager.resolve_depend_containers(target_containers)

        keys = set()
        for container in target_containers:
            keys.update(container.configs.keys())
        for container in target_containers:
            keys.update(container.extend_configs.keys())
        if not names:
            keys.update([key for key, value in manager.configs.items() if not isinstance(value, ConfigField)])
            keys.update(manager.env_config.keys())
        for key in sorted(keys):
            value = manager.env_config.get(key)
            self.logger.info(f"{key}={value}")

    @subcommand("edit", help="edit the config file in an editor")
    @subcommand_argument("--editor", help="editor to use to edit the file")
    def on_command_edit(self, editor: str):
        return manager.create_process(editor, str(manager.environ.paths.config / "settings.json")).call()

    @subcommand("reload", help="reload container configs")
    def on_command_reload(self):
        manager.env_config.reload()
        manager.prepare_installed_containers()

    @subcommand("migrate", help="migrate old config data to new format (v2 §3.3/§13.2)")
    @subcommand_argument("--dry-run", action="store_true", default=False,
                         help="inspect only, do not modify anything")
    @subcommand_argument("--backup", action="store_true", default=False,
                         help="backup old config before migrating")
    def on_command_migrate(self, dry_run: bool, backup: bool):
        from linktools.core import ConfigMigration
        old_path = str(manager.environ.paths.config / "settings.json")
        mig = ConfigMigration(manager.environ.config_store, self.logger)
        info = mig.inspect(old_path)
        if info["count"] == 0:
            self.logger.info("No old config data found to migrate.")
            return
        self.logger.info(f"Found {info['count']} keys in old config: {old_path}")
        if dry_run:
            for key in info["keys"]:
                self.logger.info(f"  {key}")
            return
        backup_path = None
        if backup:
            backup_path = mig.backup(old_path)
            self.logger.info(f"Backed up to {backup_path}")
        report = mig.migrate(old_path)
        self.logger.info(
            f"Migrated {len(report['migrated'])}, skipped {len(report['skipped'])}, "
            f"legacy {len(report['legacy'])}")
        if mig.verify():
            self.logger.info("Verification passed.")
        else:
            self.logger.error("Verification FAILED!")
            if backup_path:
                self.logger.warning("Rolling back...")
                mig.rollback(backup_path, old_path)


class ExecCommand(BaseCommand):
    """
    exec container command
    """

    @property
    def name(self):
        return "exec"

    @property
    def config(self):
        return manager.env_config

    @property
    def _subparser(self) -> "CommandParser":
        parser = CommandParser()

        subcommands: "list[SubCommand]" = []
        for container in manager.get_installed_containers():
            subcommand_group = SubCommandGroup(container.name, container.description)
            subcommands.append(subcommand_group)
            subcommands.extend(self.walk_subcommands(container, parent_id=subcommand_group.id))
        self.add_subcommands(parser, target=subcommands)

        return parser

    def init_arguments(self, parser: "CommandParser") -> None:

        class Completer(ArgParseComplete.Completer):
            get_parser = lambda _: self._subparser
            get_args = lambda _, args, **kw: \
                [args.exec_name, *args.exec_args] \
                if args.exec_name \
                else None

        parser.add_argument("exec_name", nargs="?", metavar="CONTAINER", help="container name",
                            choices=LazyChoices(_iter_installed_container_names))
        action = parser.add_argument("exec_args", nargs="...", metavar="ARGS", help="container exec args")
        action.completer = Completer()

    def run(self, args: "Namespace") -> "int | None":
        args = self._subparser.parse_args([args.exec_name, *args.exec_args] if args.exec_name else [])
        subcommand = self.parse_subcommand(args)
        if not subcommand or isinstance(subcommand, SubCommandGroup):
            return self.print_subcommands(args, root=subcommand, max_level=2)
        manager.prepare_installed_containers()
        return subcommand.run(args)


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
                         choices=LazyChoices(_iter_container_names))
    def on_command_list(self, names: "list[str]" = None, detail: bool = False):
        install_containers = manager.get_installed_containers(resolve=False)
        all_install_containers = manager.resolve_depend_containers(install_containers)
        running_containers = manager.get_running_containers()
        for container in sorted(manager.containers.values(), key=lambda o: o.order):
            if names and container.name not in names:
                continue
            installed = container in all_install_containers
            added = container in install_containers
            running = container in running_containers
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
                         choices=LazyChoices(_iter_container_names))
    def on_command_add(self, names: "list[str]"):
        containers = manager.add_installed_containers(*names)
        assert containers, "No container added"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Add {', '.join(result)} success")

    @subcommand("remove", help="remove containers from installed list")
    @subcommand_argument("-f", "--force", help="Force remove")
    @subcommand_argument("names", metavar="CONTAINER", nargs="+", help="container name",
                         choices=LazyChoices(_iter_container_names))
    def on_command_remove(self, names: "list[str]", force: bool = False):
        containers = manager.remove_installed_containers(*names, force=force)
        assert containers, "No container removed"
        result = sorted(list([container.name for container in containers]))
        self.logger.info(f"Remove {', '.join(result)} success")

    @subcommand("up", help="deploy installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_iter_installed_container_names))
    def on_command_up(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        context = self._make_context(["up", pull and "pull", build and "build"], names)

        build_options = []
        up_options = ["--detach", "--no-build"]
        if pull:
            build_options.extend(["--pull"])
            up_options.extend(["--pull", "always"])
        else:
            build_options.extend(["--pull=false"])
            up_options.extend(["--pull", "missing"])
        if context.is_full_containers:
            up_options.extend(["--remove-orphans"])

        for key in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
            if key in os.environ:
                build_options.extend(["--build-arg", f"{key}={os.environ[key]}"])
            key = key.upper()
            if key in os.environ:
                build_options.extend(["--build-arg", f"{key}={os.environ[key]}"])

        services = []
        if not context.is_full_containers:
            for container in context.target_containers:
                services.extend(container.services.keys())
            if not services:
                raise ContainerError(
                    f"No service found in container "
                    f"`{','.join([c.name for c in context.target_containers])}`"
                )

        with manager.notify_start(context):
            if build:
                manager.create_docker_compose_process(
                    context.containers,
                    "build", *build_options, *services,
                ).check_call()
            manager.create_docker_compose_process(
                context.containers,
                "up", *up_options, *services
            ).check_call()

        with manager.notify_remove(context):
            pass

    @subcommand("restart", help="restart installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_iter_installed_container_names))
    def on_command_restart(self, names: "list[str]" = None, build: bool = True, pull: str = False):
        context = self._make_context(["restart", pull and "pull", build and "build"], names)

        build_options = []
        up_options = ["--detach", "--no-build"]
        if pull:
            build_options.extend(["--pull"])
            up_options.extend(["--pull", "always"])
        if context.is_full_containers:
            up_options.extend(["--remove-orphans"])

        services = []
        if not context.is_full_containers:
            for container in context.target_containers:
                services.extend(container.services.keys())
            if not services:
                raise ContainerError(
                    f"No service found in container "
                    f"`{','.join([c.name for c in context.target_containers])}`"
                )

        with manager.notify_stop(context):
            manager.create_docker_compose_process(
                context.containers,
                "stop", *services
            ).check_call()

        with manager.notify_start(context):
            if build:
                manager.create_docker_compose_process(
                    context.containers,
                    "build", *build_options, *services,
                ).check_call()
            manager.create_docker_compose_process(
                context.containers,
                "up", *up_options, *services
            ).check_call()

        with manager.notify_remove(context):
            pass

    @subcommand("down", help="stop installed containers")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_iter_installed_container_names))
    def on_command_down(self, names: "list[str]" = None):
        context = self._make_context("down", names)

        services = []
        if not context.is_full_containers:
            for container in context.target_containers:
                services.extend(container.services.keys())
            if not services:
                raise ContainerError(
                    f"No service found in container "
                    f"`{','.join([c.name for c in context.target_containers])}`"
                )

        with manager.notify_stop(context):
            manager.create_docker_compose_process(
                context.containers,
                "down", *services
            ).check_call()

        with manager.notify_remove(context):
            pass

    def _make_context(self, commands, names):
        context = EventContext()
        context.commands = [commands] if isinstance(commands, str) else list(filter(None, commands))
        context.containers = manager.prepare_installed_containers()
        if not names:
            context.target_containers = context.containers
            context.is_full_containers = True
        else:
            context.target_containers = [c for c in context.containers if c.name in names]
            context.is_full_containers = False
        return context


command = Command()
if __name__ == '__main__':
    command.main()
