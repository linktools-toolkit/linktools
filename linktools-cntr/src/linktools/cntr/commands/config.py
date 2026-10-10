#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.cli.argparse import KeyValueAction, LazyChoices
from linktools.core import ConfigField, redact_config_value
from linktools.errors import ConfigNotFoundError
from ..container import ContainerError
from . import _shared
from ._order import CONFIG_COMMAND_ORDER


class ConfigCommand(BaseCommandGroup):
    """
    manage container configs
    """

    @property
    def name(self) -> str:
        return "config"

    def init_arguments(self, parser: "CommandParser") -> None:
        self.add_subcommands(parser=parser, sort=True)

    @subcommand("set", order=CONFIG_COMMAND_ORDER["set"], help="set container configs")
    @subcommand_argument("configs", action=KeyValueAction, nargs="+", help="container config key=value")
    def on_command_set(self, configs: "dict[str, str]") -> None:
        _shared.manager.load_installed_config_metadata()
        config = _shared.manager.env_config
        # Validate the entire batch before the single atomic store write.
        for key, value in configs.items():
            config.validate_value(key, value)
        config.persist_many(configs)

        for key in sorted(configs):
            shown = redact_config_value(config.schema.get(key), config.get(key))
            self.logger.info(f"{key}: {shown}")

    @subcommand("unset", order=CONFIG_COMMAND_ORDER["unset"], help="remove container configs")
    @subcommand_argument("configs", action=KeyValueAction, metavar="KEY", nargs="+", help="container config keys")
    def on_command_remove(self, configs: "dict[str, str]") -> None:
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
    def on_command_list(self, names: "list[str]", with_dependencies: bool = False, show_secret: bool = False) -> None:
        manager = _shared.manager
        config = manager.env_config
        containers = manager.load_installed_config_metadata()
        target_containers = [c for c in containers if c.name in names] if names else containers
        if with_dependencies and names:
            target_containers = manager.resolver.resolve_dependencies(target_containers)

        keys = set()
        for container in target_containers:
            keys.update(container.configs)
            keys.update(container.extend_configs)
        if not names:
            keys.update(key for key, value in manager.configs.items() if not isinstance(value, ConfigField))
            # Unconfigured manager fields may prompt, so only include persisted extras.
            keys.update(config.persisted_keys())

        for key in sorted(keys):
            try:
                value = config.get(key)
            except ConfigNotFoundError:
                continue
            if show_secret:
                # Explicit disclosure bypasses the logger's secret-name redaction.
                print(f"{key}={value}")
            else:
                shown = redact_config_value(config.schema.get(key), value)
                self.logger.info(f"{key}={shown}")

    @subcommand("get", order=CONFIG_COMMAND_ORDER["get"], help="read one or more resolved config values")
    @subcommand_argument("keys", metavar="KEY", nargs="+", help="config key(s)")
    @subcommand_argument("--show-secret", action="store_true", default=False,
                         help="show secret values in plain text instead of the logger's automatic ***-redaction")
    def on_command_get(self, keys: "list[str]", show_secret: bool = False) -> None:
        _shared.manager.load_installed_config_metadata()
        config = _shared.manager.env_config
        for key in keys:
            value = config.get(key)
            if show_secret:
                print(f"{key}={value}")
            else:
                shown = redact_config_value(config.schema.get(key), value)
                self.logger.info(f"{key}={shown}")

    @subcommand("explain", order=CONFIG_COMMAND_ORDER["explain"],
               help="show a value's resolved source, default, persisted state and sensitivity")
    @subcommand_argument("key", help="config key")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_explain(self, key: str, as_json: bool = False) -> None:
        _shared.manager.load_installed_config_metadata()
        info = _shared.manager.env_config.explain(key)
        if as_json:
            import json
            print(json.dumps(info, indent=2, sort_keys=True, default=str))
        else:
            for field_name in sorted(info):
                self.logger.info(f"{field_name}: {info[field_name]}")

    @subcommand("validate", order=CONFIG_COMMAND_ORDER["validate"],
               help="validate persisted config values' types (never runs docker compose config)")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_validate(self, as_json: bool = False) -> None:
        _shared.manager.load_installed_config_metadata()
        config = _shared.manager.env_config
        errors = []
        # Do not resolve schema-only fields that nobody has configured yet.
        for key in config.persisted_keys():
            try:
                config.get(key)
            except Exception as exc:  # noqa: BLE001 - report every invalid key
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
    def on_command_edit(self, editor: str) -> int:
        return _shared.manager.runtime.create_process(
            editor, str(_shared.manager.environ.paths.config / "settings.json")
        ).call()

    @subcommand("reload", order=CONFIG_COMMAND_ORDER["reload"], help="reload container configs")
    def on_command_reload(self) -> None:
        _shared.manager.env_config.reload()
        _shared.manager.load_installed_config_metadata()
