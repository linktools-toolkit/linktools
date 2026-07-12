#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import os
from collections import namedtuple

from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.cli.argparse import KeyValueAction, LazyChoices
from linktools.core import ConfigField, redact_config_value
from linktools.errors import ConfigNotFoundError
from ..container import ContainerError
from . import _shared
from ._order import CONFIG_COMMAND_ORDER

# A single displayable config entry: which Config object it resolves
# through (identity is what dedup keys off -- two containers can share one
# Config, e.g. same-repo siblings or the shared builtin Config), plus a
# stable, credential-free ``owner_id`` (used to detect genuine cross-owner
# ambiguity for a key) separate from the human-facing ``owner_label`` (which
# two DIFFERENT repositories may share, e.g. "common" cloned from
# team-a/common.git and team-b/common.git).
ConfigListEntry = namedtuple("ConfigListEntry", ["owner_id", "owner_label", "key", "config"])


def _owner_identity(container):
    """(owner_id, owner_label) for a container's config entries.

    ``owner_id`` is the repository's real, absolute root path (stable and
    never shown raw to the user) -- never a repo_name (two different repos
    can share one) and never a URL (may embed a Git credential). Falls back
    to the Config object's identity only when a context/root_path is
    genuinely unavailable (e.g. a container built directly in a test).
    """
    context = container.repository_context
    root_path = context.root_path if context is not None else None
    if root_path:
        owner_id = os.path.realpath(str(root_path))
    else:
        owner_id = "config:%s" % id(container.env_config)
    owner_label = (context.repo_name if context is not None else None) or container.name
    return owner_id, owner_label


def _short_owner_suffix(owner_id):
    """An 8-hex-char stable suffix derived from ``owner_id`` -- long enough
    to disambiguate in practice, short enough to stay readable, and never
    the raw (potentially filesystem-revealing) owner_id itself."""
    return hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:8]


def _shown_config_value(config, key, value, show_secret=False):
    """The single choke point every command output (set/get/list) routes a
    resolved value through before display: --show-secret is an explicit,
    opt-in bypass; otherwise a ``ConfigField(secret=True)`` value is always
    masked, regardless of whether the logger's own key-name-pattern
    redaction would have caught it (a field can be secret without its name
    looking like one)."""
    if show_secret:
        return value
    return redact_config_value(config.schema.get(key), value)


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
        config = _shared.manager.env_config
        for key, value in configs.items():
            config.persist(key, value)
        for key in sorted(configs.keys()):
            # Always redacted -- config set intentionally has no
            # --show-secret; the whole point of `set` is to confirm the
            # write happened, not to echo the value back.
            shown = _shown_config_value(config, key, config.get(key), show_secret=False)
            self.logger.info(f"{key}: {shown}")

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

        seen_entries = set()
        declared_keys = set()
        entries: "list[ConfigListEntry]" = []

        def add_container_entries(container, keys):
            config = container.env_config
            owner_id, owner_label = _owner_identity(container)
            for key in keys:
                declared_keys.add(key)
                # Dedup by (id(config), key), not key alone: different
                # repositories never share a Config object (only the
                # shared Environment/RuntimeOverride/Persistent triple, plus
                # each repo's own local-file layer), so the SAME key can
                # legitimately resolve to a DIFFERENT value per repository
                # and must be listed once per distinct Config, not collapsed
                # to a single, arbitrarily-first-seen entry.
                identity = (id(config), key)
                if identity in seen_entries:
                    continue
                seen_entries.add(identity)
                entries.append(ConfigListEntry(owner_id=owner_id, owner_label=owner_label, key=key, config=config))

        for container in target_containers:
            add_container_entries(container, container.configs.keys())
        for container in target_containers:
            add_container_entries(container, container.extend_configs.keys())
        if not names:
            for key, value in _shared.manager.configs.items():
                if isinstance(value, ConfigField):
                    continue
                declared_keys.add(key)
                identity = (id(_shared.manager.env_config), key)
                if identity in seen_entries:
                    continue
                seen_entries.add(identity)
                entries.append(ConfigListEntry(owner_id="manager", owner_label="manager",
                                               key=key, config=_shared.manager.env_config))
            # Only keys someone has actually set (persisted_keys()), not every
            # schema-declared field name (keys()) -- otherwise a manager-level
            # field that's never actually been configured (e.g.
            # DOCKER_DOWNLOAD_PATH) gets force-resolved just because it's
            # *possible* to set, prompting for it even though nothing needs it.
            for key in _shared.manager.env_config.persisted_keys():
                if key in declared_keys:
                    continue
                entries.append(ConfigListEntry(owner_id="manager", owner_label="manager",
                                               key=key, config=_shared.manager.env_config))

        # Owner is only shown when a key is genuinely ambiguous -- more than
        # one distinct REPOSITORY (owner_id) lists it, not merely more than
        # one Config object -- keeping `KEY=VALUE` stable for the common
        # single-owner case. Two different repositories that happen to
        # share a repo_name (e.g. both cloned as "common") get a stable
        # hash suffix instead of colliding under one ambiguous label.
        owner_ids_by_key = {}
        labels_by_key = {}
        for entry in entries:
            owner_ids_by_key.setdefault(entry.key, set()).add(entry.owner_id)
            labels_by_key.setdefault(entry.key, {}).setdefault(entry.owner_label, set()).add(entry.owner_id)

        for entry in sorted(entries, key=lambda e: (e.key, e.owner_label, e.owner_id)):
            try:
                value = entry.config.get(entry.key)
            except ConfigNotFoundError:
                # Declared (e.g. an optional secret extend_config) but never
                # actually configured -- nothing to list, and never worth
                # force-prompting for just to render a listing.
                continue

            if len(owner_ids_by_key[entry.key]) <= 1:
                label = entry.key
            elif len(labels_by_key[entry.key][entry.owner_label]) > 1:
                label = f"{entry.owner_label}@{_short_owner_suffix(entry.owner_id)}:{entry.key}"
            else:
                label = f"{entry.owner_label}:{entry.key}"

            if show_secret:
                # self.logger.info goes through the logging redaction filter,
                # which masks anything that looks like a secret/password/token
                # (by design -- never leak one into a log file/CI output by
                # accident). --show-secret is an explicit, opt-in request to
                # see the real value, so print it directly instead.
                print(f"{label}={value}")
            else:
                shown = _shown_config_value(entry.config, entry.key, value, show_secret=False)
                self.logger.info(f"{label}={shown}")

    @subcommand("get", order=CONFIG_COMMAND_ORDER["get"], help="read one or more resolved config values")
    @subcommand_argument("keys", metavar="KEY", nargs="+", help="config key(s)")
    @subcommand_argument("--show-secret", action="store_true", default=False,
                         help="show secret values in plain text instead of the logger's automatic ***-redaction")
    def on_command_get(self, keys: "list[str]", show_secret: bool = False):
        config = _shared.manager.env_config
        for key in keys:
            value = config.get(key)
            if show_secret:
                # Bypasses the logger's redaction filter entirely (an
                # explicit request to see the real value), never routes
                # back through it -- printing here, not logging.
                print(f"{key}={value}")
            else:
                shown = _shown_config_value(config, key, value, show_secret=False)
                self.logger.info(f"{key}={shown}")

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
