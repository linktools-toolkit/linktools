#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandParser, subcommand, subcommand_argument
from linktools.cli.argparse import KeyValueAction, LazyChoices
from linktools.core import ConfigField
from . import _shared

if TYPE_CHECKING:
    from argparse import Namespace


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
        containers = _shared.manager.prepare_installed_containers()
        return _shared.manager.create_docker_compose_process(
            containers,
            "config",
            privilege=False,
        ).check_call()

    @subcommand("set", help="set container configs")
    @subcommand_argument("configs", action=KeyValueAction, nargs="+", help="container config key=value")
    def on_command_set(self, configs: "dict[str, str]"):
        for key, value in configs.items():
            _shared.manager.env_config.persist(key, value)
        for key in sorted(configs.keys()):
            value = _shared.manager.env_config.get(key)
            self.logger.info(f"{key}: {value}")

    @subcommand("unset", help="remove container configs")
    @subcommand_argument("configs", action=KeyValueAction, metavar="KEY", nargs="+", help="container config keys")
    def on_command_remove(self, configs: "dict[str, str]"):
        for key in configs.keys():
            _shared.manager.env_config.remove(key)
        self.logger.info(f"Unset {', '.join(configs.keys())} success")

    @subcommand("list", help="list container configs")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    @subcommand_argument("-d", "--with-dependencies", action="store_true", default=False,
                         help="include configs from dependency containers")
    @subcommand_argument("--show-secret", action="store_true", default=False,
                         help="show secret values in plain text instead of the logger's automatic ***-redaction")
    def on_command_list(self, names: "list[str]", with_dependencies: bool = False, show_secret: bool = False):
        containers = _shared.manager.prepare_installed_containers()
        target_containers = [c for c in containers if c.name in names] if names else containers
        if with_dependencies and names:
            target_containers = _shared.manager.resolve_depend_containers(target_containers)

        keys = set()
        for container in target_containers:
            keys.update(container.configs.keys())
        for container in target_containers:
            keys.update(container.extend_configs.keys())
        if not names:
            keys.update([key for key, value in _shared.manager.configs.items() if not isinstance(value, ConfigField)])
            # Only keys someone has actually set (persisted_keys()), not every
            # schema-declared field name (keys()) -- otherwise a manager-level
            # field that's never actually been configured (e.g.
            # DOCKER_DOWNLOAD_PATH) gets force-resolved just because it's
            # *possible* to set, prompting for it even though nothing needs it.
            keys.update(_shared.manager.env_config.persisted_keys())
        for key in sorted(keys):
            value = _shared.manager.env_config.get(key)
            if show_secret:
                # self.logger.info goes through the logging redaction filter,
                # which masks anything that looks like a secret/password/token
                # (by design -- never leak one into a log file/CI output by
                # accident). --show-secret is an explicit, opt-in request to
                # see the real value, so print it directly instead.
                print(f"{key}={value}")
            else:
                self.logger.info(f"{key}={value}")

    @subcommand("edit", help="edit the config file in an editor")
    @subcommand_argument("--editor", help="editor to use to edit the file")
    def on_command_edit(self, editor: str):
        return _shared.manager.create_process(editor, str(_shared.manager.environ.paths.config / "settings.json")).call()

    @subcommand("reload", help="reload container configs")
    def on_command_reload(self):
        _shared.manager.env_config.reload()
        _shared.manager.prepare_installed_containers()
