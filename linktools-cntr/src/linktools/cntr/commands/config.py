#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import os
from collections import namedtuple

from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.cli.argparse import KeyValueAction, LazyChoices
from linktools.core import ConfigField
from linktools.errors import ConfigNotFoundError
from linktools.types import MISSING
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

# One Config that declares a given key in its OWN schema, plus the field
# definition it declares for it -- ``set``/``get``/``explain``/``validate``
# resolve this per key (not just against Manager Config) because a
# third-party repository's ConfigField (secret=True, a custom cast/validator)
# only ever lives in that repository's own Config schema, never Manager's.
ConfigTarget = namedtuple("ConfigTarget", ["owner_id", "owner_label", "config", "field"])


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


def _installed_containers_or_empty():
    """``prepare_installed_containers()`` raises ``ContainerError`` when
    nothing is installed yet -- fine for ``list``, which has nothing
    meaningful to show either way, but ``set``/``get``/``explain``/
    ``validate`` must keep working against Manager Config alone before any
    repository is ever added (e.g. setting HOST as one of the very first
    commands run on a fresh install)."""
    try:
        return _shared.manager.prepare_installed_containers()
    except ContainerError:
        return []


def _iter_unique_configs(containers):
    """Yield (owner_id, owner_label, config) for Manager Config first, then
    every distinct per-repository Config -- same identity dedup as
    ``config list`` (two containers of the same repo share one Config; two
    different repos never do)."""
    manager_config = _shared.manager.env_config
    seen = {id(manager_config)}
    yield "manager", "manager", manager_config
    for container in containers:
        config = container.env_config
        if id(config) in seen:
            continue
        seen.add(id(config))
        owner_id, owner_label = _owner_identity(container)
        yield owner_id, owner_label, config


def _find_config_targets(key, containers):
    """Every Config -- Manager's and each distinct repository's -- that
    declares its OWN field for ``key``. A key can resolve fine through
    Manager Config's allow_unknown source chain while still being a secret
    ONLY a repository's own schema knows about, so callers must not assume
    "found via Manager" == "the only definition".

    ``build_repository_config`` seeds every repository's schema with a copy
    of Manager's OWN field objects (by reference) so a repo container can
    still resolve manager-owned keys like HOST -- that is not a repository
    declaring anything of its own, so a repo whose field for ``key`` is
    literally the SAME object as Manager's is never counted as a second
    target: otherwise every ordinary manager-only key would look ambiguous
    (a target per installed repo) the moment 2+ repos are installed.
    """
    manager_field = _shared.manager.env_config.schema.get(key)
    targets = []
    for owner_id, owner_label, config in _iter_unique_configs(containers):
        field = config.schema.get(key)
        if field is None:
            continue
        if owner_id != "manager" and field is manager_field:
            continue
        targets.append(ConfigTarget(owner_id=owner_id, owner_label=owner_label, config=config, field=field))
    return targets


def _redact_across_targets(targets, value, show_secret=False):
    """Mask ``value`` if ANY target declares the key secret=True -- a shared
    persisted key must not be shown in the clear just because one repo (or
    Manager) happens not to flag it, when another does."""
    if show_secret:
        return value
    if _is_secret_targets(targets):
        return "***"
    return value


def _is_secret_targets(targets):
    return any(target.field.secret for target in targets)


def _force_redact_explain(info):
    """Force a ``Config.explain()`` result to redact its resolved/raw/
    candidate values, regardless of whether ITS OWN field was secret.

    Used when a DIFFERENT target for the same shared persisted key declares
    secret=True -- ``Config.explain()`` only ever self-redacts against its
    own Config's field, so a repository whose own schema doesn't flag the
    key (e.g. a plain field colliding with another repo's secret one) would
    otherwise show a shared secret in the clear."""
    info = dict(info)
    info["secret"] = True
    if info.get("resolved_value") is not MISSING:
        info["resolved_value"] = "***"
    if info.get("raw_value") is not MISSING:
        info["raw_value"] = "***"
    info["all_candidates"] = [dict(candidate, raw="***") for candidate in info.get("all_candidates", [])]
    return info


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
        containers = _installed_containers_or_empty()
        for key, value in configs.items():
            config.persist(key, value)
        for key in sorted(configs.keys()):
            # Always redacted -- config set intentionally has no
            # --show-secret; the whole point of `set` is to confirm the
            # write happened, not to echo the value back. Secret-ness is
            # resolved against every repository's own schema, not just
            # Manager's -- see ConfigTarget.
            targets = _find_config_targets(key, containers)
            shown = _redact_across_targets(targets, config.get(key))
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

        # Redaction is decided per key across EVERY installed repository
        # (the full, unfiltered `containers`, not the possibly name-filtered
        # `target_containers` being rendered) -- same "any target says
        # secret" rule set/get/explain/validate use, so a persisted key
        # shared across repos is never shown in the clear just because the
        # entry being rendered here doesn't itself flag the field secret.
        secret_keys = {
            key for key in {entry.key for entry in entries}
            if _is_secret_targets(_find_config_targets(key, containers))
        }

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
                shown = "***" if entry.key in secret_keys else value
                self.logger.info(f"{label}={shown}")

    @subcommand("get", order=CONFIG_COMMAND_ORDER["get"], help="read one or more resolved config values")
    @subcommand_argument("keys", metavar="KEY", nargs="+", help="config key(s)")
    @subcommand_argument("--show-secret", action="store_true", default=False,
                         help="show secret values in plain text instead of the logger's automatic ***-redaction")
    def on_command_get(self, keys: "list[str]", show_secret: bool = False):
        manager_config = _shared.manager.env_config
        containers = _installed_containers_or_empty()
        for key in keys:
            targets = _find_config_targets(key, containers)
            if len(targets) <= 1:
                # Zero targets: genuinely unknown key, fall back to
                # Manager's allow_unknown source-chain read. One target:
                # resolve through THAT Config so its own cast/validator
                # apply, not just Manager's.
                config = targets[0].config if targets else manager_config
                value = config.get(key)
                if show_secret:
                    # Bypasses the logger's redaction filter entirely (an
                    # explicit request to see the real value), never routes
                    # back through it -- printing here, not logging.
                    print(f"{key}={value}")
                else:
                    shown = _redact_across_targets(targets, value)
                    self.logger.info(f"{key}={shown}")
            else:
                # Same key declared by more than one repository -- each may
                # resolve to a different value (own cast) and must be shown
                # with its owner, never silently collapsed to one.
                for target in targets:
                    value = target.config.get(key)
                    label = f"{target.owner_label}:{key}"
                    if show_secret:
                        print(f"{label}={value}")
                    else:
                        shown = _redact_across_targets(targets, value)
                        self.logger.info(f"{label}={shown}")

    @subcommand("explain", order=CONFIG_COMMAND_ORDER["explain"],
               help="show a value's resolved source, default, persisted state and sensitivity")
    @subcommand_argument("key", help="config key")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_explain(self, key: str, as_json: bool = False):
        containers = _installed_containers_or_empty()
        targets = _find_config_targets(key, containers)
        if len(targets) <= 1:
            config = targets[0].config if targets else _shared.manager.env_config
            info = config.explain(key)
        else:
            # Each repository's own explain() self-redacts only against ITS
            # OWN field -- if ANY target for this shared key declares
            # secret=True, every target's output is force-redacted here too,
            # so a repo whose own field isn't flagged secret never shows a
            # value another repo's field on the same persisted key protects.
            force_secret = _is_secret_targets(targets)
            info = {
                "key": key,
                "targets": [
                    {
                        "owner": target.owner_label,
                        "explain": (
                            _force_redact_explain(target.config.explain(key))
                            if force_secret else target.config.explain(key)
                        ),
                    }
                    for target in targets
                ],
            }
        if as_json:
            import json
            print(json.dumps(info, indent=2, sort_keys=True, default=str))
        elif "targets" in info:
            for target_entry in info["targets"]:
                self.logger.info(f"owner: {target_entry['owner']}")
                for field_name in sorted(target_entry["explain"].keys()):
                    self.logger.info(f"  {field_name}: {target_entry['explain'][field_name]}")
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
        containers = _installed_containers_or_empty()
        errors = []
        for key in manager.env_config.persisted_keys():
            targets = _find_config_targets(key, containers)
            if not targets:
                try:
                    manager.env_config.get(key)
                except Exception as exc:  # noqa: BLE001 - collect every bad key, not just the first
                    errors.append(dict(key=key, error=str(exc)))
                continue
            # Every repository's own cast/validator is applied independently
            # -- a key shared by two repositories with different rules
            # (spec: one requires int, the other a "tcp-" prefix) can pass
            # for one and fail for the other, and both outcomes must surface.
            for target in targets:
                try:
                    target.config.get(key)
                except Exception as exc:  # noqa: BLE001 - collect every bad target, not just the first
                    errors.append(dict(key=key, owner=target.owner_label, error=str(exc)))

        if as_json:
            import json
            print(json.dumps(dict(valid=not errors, errors=errors), indent=2, sort_keys=True))
        elif not errors:
            self.logger.info("All persisted config values are valid.")
        else:
            for entry in errors:
                owner = entry.get("owner")
                label = f"{entry['key']} ({owner})" if owner else entry["key"]
                self.logger.info(f"[INVALID] {label}: {entry['error']}")

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
