#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import yaml

from linktools.cli import BaseCommandGroup, CommandParser, subcommand, subcommand_argument
from linktools.rich import choose, confirm, is_no_input
from ..container import ContainerError
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
        repos = _shared.manager.repo_store.get_all()
        for key, value in repos.items():
            data = {key: value}
            self.logger.info(
                yaml.dump(data, sort_keys=False).strip()
            )

    @subcommand("add", order=REPO_COMMAND_ORDER["add"], help="add repository")
    @subcommand_argument("url", help="repository url")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force add (skip trust prompt)")
    def on_command_add(self, url: str, branch: str = None, force: bool = False):
        # A repo may carry executable Python container definitions, so
        # interactive `add` asks for confirmation unless --force.
        # Non-interactive runs don't block.
        if not force and not is_no_input():
            if not confirm(
                    "This repository may contain executable Python container definitions. "
                    "Only add repositories you trust. Continue?",
                    default=False):
                raise ContainerError("Canceled")
        _shared.manager.repo_store.add(url, branch=branch, force=force)

    @subcommand("status", order=REPO_COMMAND_ORDER["status"], help="show repository status (read-only)")
    def on_command_status(self):
        repos = _shared.manager.repo_store.get_all()
        if not repos:
            self.logger.info("No repository found")
            return
        from ..repo.status import describe_repository
        for url, meta in repos.items():
            info = describe_repository(_shared.manager, url, meta)
            self.logger.info(yaml.dump({url: info}, sort_keys=False).strip())

    @subcommand("validate", order=REPO_COMMAND_ORDER["validate"],
               help="validate repository local config and compatibility (read-only)")
    @subcommand_argument("url", nargs="?", help="repository url or local path; all repositories if omitted")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_validate(self, url: str = None, as_json: bool = False):
        repos = _shared.manager.repo_store.get_all()
        if url is not None:
            if url not in repos:
                raise ContainerError(f"Repository `{url}` not found.")
            targets = {url: repos[url]}
        else:
            targets = repos

        if not targets:
            self.logger.info("No repository found")
            return

        from ..repo.status import describe_repository
        results = {
            u: describe_repository(_shared.manager, u, m)
            for u, m in targets.items()
        }

        if as_json:
            import json
            # Machine-readable output goes straight to stdout, not the logger
            # (whose destination depends on TTY/rich state).
            print(json.dumps(results, indent=2, sort_keys=True, default=str))
        else:
            for u, info in results.items():
                self.logger.info(yaml.dump({u: info}, sort_keys=False).strip())

        incompatible = sorted(u for u, info in results.items() if info.get("compatible") is False)
        if incompatible:
            raise ContainerError(f"Incompatible repositories: {', '.join(incompatible)}")

    @subcommand("update", order=REPO_COMMAND_ORDER["update"], help="update repositories")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force update")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    def on_command_update(self, branch: str = None, force: bool = False, as_json: bool = False):
        results = _shared.manager.repo_store.update(branch=branch, reset=force)

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
        repos = list(_shared.manager.repo_store.get_all().keys())
        if not repos:
            raise ContainerError("No repository found")

        if url is None:
            repo = choose("Choose repository you want to remove", repos)
            if not confirm(f"Remove repository `{repo}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.repo_store.remove(repo)

        elif url in repos:
            if not confirm(f"Remove repository `{url}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.repo_store.remove(url)

        else:
            raise ContainerError(f"Repository `{url}` not found.")
