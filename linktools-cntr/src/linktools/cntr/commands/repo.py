#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml

from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.rich import choose, confirm, is_no_input
from ..container import ContainerError
from ..repo.service import safe_display_url
from . import _shared
from ._order import REPO_COMMAND_ORDER


class RepoCommand(BaseCommandGroup):
    """
    manage container repository
    """

    @property
    def name(self):
        return "repo"

    def init_arguments(self, parser: "CommandParser") -> None:
        self.add_subcommands(parser=parser, sort=True)

    @subcommand("list", order=REPO_COMMAND_ORDER["list"], help="list repositories")
    def on_command_list(self):
        repos = _shared.manager.repos.get_all()
        for key, value in repos.items():
            data = {safe_display_url(key): value}
            self.logger.info(
                yaml.dump(data, sort_keys=False).strip()
            )

    @subcommand("add", order=REPO_COMMAND_ORDER["add"], help="add repository")
    @subcommand_argument("url", help="repository url")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("--replace", action="store_true",
                         help="replace an already-added repository at this URL/path")
    def on_command_add(self, url: str, branch: str = None, replace: bool = False):
        # A repo may carry executable Python container definitions, so
        # interactive `add` asks for confirmation; the global --yes flag (or
        # any other non-interactive run) skips it via is_no_input() -- there
        # is no repo-add-specific flag for this, to avoid a second way to
        # spell the same thing. Independent of --replace: skipping this
        # prompt never implies replacing an existing repository, and
        # --replace never skips the prompt on its own.
        if not is_no_input():
            if not confirm(
                    "This repository may contain executable Python container definitions. "
                    "Only add repositories you trust. Continue?",
                    default=False):
                raise ContainerError("Canceled")
        _shared.manager.repos.add(url, branch=branch, replace=replace)

    @subcommand("status", order=REPO_COMMAND_ORDER["status"], help="show repository status (read-only)")
    def on_command_status(self):
        repos = _shared.manager.repos.get_all()
        if not repos:
            self.logger.info("No repository found")
            return
        for url, meta in repos.items():
            info = _shared.manager.repos.describe(url, meta)
            self.logger.info(yaml.dump({safe_display_url(url): info}, sort_keys=False).strip())

    @subcommand("validate", order=REPO_COMMAND_ORDER["validate"],
               help="validate repository local config and compatibility (read-only)")
    @subcommand_argument("url", nargs="?", help="repository url or local path; all repositories if omitted")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_validate(self, url: str = None, as_json: bool = False):
        if not _shared.manager.repos.get_all():
            self.logger.info("No repository found")
            return

        results, incompatible = _shared.manager.repos.validate(url)

        if as_json:
            import json
            # Machine-readable output goes straight to stdout, not the logger
            # (whose destination depends on TTY/rich state).
            print(json.dumps(results, indent=2, sort_keys=True, default=str))
        else:
            for u, info in results.items():
                self.logger.info(yaml.dump({u: info}, sort_keys=False).strip())

        if incompatible:
            raise ContainerError(f"Incompatible repositories: {', '.join(incompatible)}")

    @subcommand("update", order=REPO_COMMAND_ORDER["update"], help="update repositories")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force update")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_update(self, branch: str = None, force: bool = False, as_json: bool = False):
        results = _shared.manager.repos.update(branch=branch, reset=force)

        if as_json:
            import dataclasses
            import json
            print(json.dumps([dataclasses.asdict(r) for r in results], indent=2, sort_keys=True))
        else:
            for result in results:
                if not result.updated:
                    self.logger.warning(f"Repository `{result.url}` update failed: {result.error}")
                elif not result.compatible:
                    self.logger.warning(
                        f"Repository `{result.url}` updated to revision {result.revision}, "
                        f"but {result.error}"
                    )
                else:
                    self.logger.info(f"Repository `{result.url}` updated to revision {result.revision}")

        # Every repository is synced/reported regardless of another one's
        # outcome; only after all of them are done does an unmet/failed one
        # make the whole command exit non-zero.
        failed = sorted(r.url for r in results if not r.updated or not r.compatible)
        if failed:
            raise ContainerError(f"Repository update failed or incompatible: {', '.join(failed)}")

    @subcommand("remove", order=REPO_COMMAND_ORDER["remove"], help="remove repository")
    @subcommand_argument("url", nargs="?", help="repository url")
    def on_command_remove(self, url: str = None):
        repos = list(_shared.manager.repos.get_all().keys())
        if not repos:
            raise ContainerError("No repository found")

        if url is None:
            # choices is {real key: displayed label} -- choose() returns the
            # real key (needed to actually remove it), the user only ever
            # sees the credential-free label.
            repo = choose("Choose repository you want to remove",
                          {r: safe_display_url(r) for r in repos})
            if not confirm(f"Remove repository `{safe_display_url(repo)}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.repos.remove(repo)

        elif url in repos:
            if not confirm(f"Remove repository `{safe_display_url(url)}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.repos.remove(url)

        else:
            raise ContainerError(f"Repository `{url}` not found.")
