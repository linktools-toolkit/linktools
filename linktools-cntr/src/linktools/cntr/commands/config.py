#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.cli.argparse import KeyValueAction, LazyChoices
from linktools.core import ConfigField
from ..container import ContainerError
from . import _shared
from ._order import CONFIG_COMMAND_ORDER


class ConfigCommand(BaseCommandGroup):
    """
    manage container configs
    """

    @property
    def name(self):
        return "config"

    def init_arguments(self, parser: "CommandParser") -> None:
        self.add_subcommands(parser=parser, sort=True)

    @subcommand("set", order=CONFIG_COMMAND_ORDER["set"], help="set container configs")
    @subcommand_argument("configs", action=KeyValueAction, nargs="+", help="container config key=value")
    def on_command_set(self, configs: "dict[str, str]"):
        for key, value in configs.items():
            _shared.manager.env_config.persist(key, value)
        for key in sorted(configs.keys()):
            value = _shared.manager.env_config.get(key)
            self.logger.info(f"{key}: {value}")

    @subcommand("unset", order=CONFIG_COMMAND_ORDER["unset"], help="remove container configs")
    @subcommand_argument("configs", action=KeyValueAction, metavar="KEY", nargs="+", help="container config keys")
    def on_command_remove(self, configs: "dict[str, str]"):
        for key in configs.keys():
            _shared.manager.env_config.remove(key)
        self.logger.info(f"Unset {', '.join(configs.keys())} success")

    @subcommand("list", order=CONFIG_COMMAND_ORDER["list"], help="list container configs")
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
            target_containers = _shared.manager.resolver.resolve_dependencies(target_containers)

        entries = []
        seen_entries = set()

        def add_entries(config, keys):
            for key in keys:
                entry = (id(config), key)
                if entry not in seen_entries:
                    seen_entries.add(entry)
                    entries.append((key, config))

        for container in target_containers:
            add_entries(container.env_config, container.configs.keys())
        for container in target_containers:
            add_entries(container.env_config, container.extend_configs.keys())
        if not names:
            add_entries(
                _shared.manager.env_config,
                [key for key, value in _shared.manager.configs.items() if not isinstance(value, ConfigField)],
            )
            # Only keys someone has actually set (persisted_keys()), not every
            # schema-declared field name (keys()) -- otherwise a manager-level
            # field that's never actually been configured (e.g.
            # DOCKER_DOWNLOAD_PATH) gets force-resolved just because it's
            # *possible* to set, prompting for it even though nothing needs it.
            add_entries(_shared.manager.env_config, _shared.manager.env_config.persisted_keys())
        for key, config in sorted(entries, key=lambda entry: (entry[0], id(entry[1]))):
            value = config.get(key)
            if show_secret:
                # self.logger.info goes through the logging redaction filter,
                # which masks anything that looks like a secret/password/token
                # (by design -- never leak one into a log file/CI output by
                # accident). --show-secret is an explicit, opt-in request to
                # see the real value, so print it directly instead.
                print(f"{key}={value}")
            else:
                self.logger.info(f"{key}={value}")

    @subcommand("get", order=CONFIG_COMMAND_ORDER["get"], help="read one or more resolved config values")
    @subcommand_argument("keys", metavar="KEY", nargs="+", help="config key(s)")
    @subcommand_argument("--show-secret", action="store_true", default=False,
                         help="show secret values in plain text instead of the logger's automatic ***-redaction")
    def on_command_get(self, keys: "list[str]", show_secret: bool = False):
        for key in keys:
            value = _shared.manager.env_config.get(key)
            if show_secret:
                print(f"{key}={value}")
            else:
                self.logger.info(f"{key}={value}")

    @subcommand("explain", order=CONFIG_COMMAND_ORDER["explain"],
               help="show a value's resolved source, default, persisted state and sensitivity")
    @subcommand_argument("key", help="config key")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_explain(self, key: str, as_json: bool = False):
        info = _shared.manager.env_config.explain(key)
        if as_json:
            import json
            print(json.dumps(info, indent=2, sort_keys=True, default=str))
        else:
            for field_name in sorted(info.keys()):
                self.logger.info(f"{field_name}: {info[field_name]}")

    @subcommand("validate", order=CONFIG_COMMAND_ORDER["validate"],
               help="validate persisted config values' types (never runs docker compose config)")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_validate(self, as_json: bool = False):
        # Only re-validates already-persisted values (persisted_keys()), the
        # same safe enumeration `config list` uses -- iterating every
        # schema-declared field (keys()) would force-resolve, and possibly
        # interactively prompt for, fields nobody has configured yet.
        manager = _shared.manager
        errors = []
        for key in manager.env_config.persisted_keys():
            try:
                manager.env_config.get(key)
            except Exception as exc:  # noqa: BLE001 - collect every bad key, not just the first
                errors.append(dict(key=key, error=str(exc)))

        if as_json:
            import json
            print(json.dumps(dict(valid=not errors, errors=errors), indent=2, sort_keys=True))
        elif not errors:
            self.logger.info("All persisted config values are valid.")
        else:
            for entry in errors:
                self.logger.info(f"[INVALID] {entry['key']}: {entry['error']}")

        if errors:
            raise ContainerError(f"{len(errors)} persisted config value(s) failed validation")

    @subcommand("edit", order=CONFIG_COMMAND_ORDER["edit"], help="edit the config file in an editor")
    @subcommand_argument("--editor", help="editor to use to edit the file")
    def on_command_edit(self, editor: str):
        return _shared.manager.runtime.create_process(
            editor, str(_shared.manager.environ.paths.config / "settings.json")
        ).call()

    @subcommand("reload", order=CONFIG_COMMAND_ORDER["reload"], help="reload container configs")
    def on_command_reload(self):
        _shared.manager.env_config.reload()
        _shared.manager.prepare_installed_containers()
