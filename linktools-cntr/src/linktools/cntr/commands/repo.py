#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import yaml

from linktools.cli import BaseCommandGroup, subcommand, subcommand_argument
from linktools.rich import choose, confirm, is_no_input
from ..container import ContainerError
from . import _shared


class RepoCommand(BaseCommandGroup):
    """
    manage container repository
    """

    @property
    def name(self):
        return "repo"

    @subcommand("list", help="list repositories")
    def on_command_list(self):
        repos = _shared.manager.get_all_repos()
        for key, value in repos.items():
            data = {key: value}
            self.logger.info(
                yaml.dump(data, sort_keys=False).strip()
            )

    @subcommand("add", help="add repository")
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
        _shared.manager.add_repo(url, branch=branch, force=force)

    @subcommand("status", help="show repository status (read-only)")
    def on_command_status(self):
        repos = _shared.manager.get_all_repos()
        if not repos:
            self.logger.info("No repository found")
            return
        from linktools.git import GitRepository
        from dulwich.errors import NotGitRepository
        for url, meta in repos.items():
            repo_type = meta.get("type", "unknown")
            repo_path = meta.get("repo_path")
            line = f"{url} ({repo_type}) -> {repo_path}"
            if repo_type == "git" and repo_path and os.path.exists(repo_path):
                try:
                    repo = GitRepository(_shared.manager.environ, repo_path)
                    line += f" [dirty={repo.is_dirty()}]"
                except NotGitRepository:
                    pass
                except Exception:
                    pass
            self.logger.info(line)

    @subcommand("update", help="update repositories")
    @subcommand_argument("-b", "--branch", help="branch name")
    @subcommand_argument("-f", "--force", help="force update")
    def on_command_update(self, branch: str = None, force: bool = False):
        _shared.manager.update_repos(branch=branch, reset=force)

    @subcommand("remove", help="remove repository")
    @subcommand_argument("url", nargs="?", help="repository url")
    def on_command_remove(self, url: str = None):
        repos = list(_shared.manager.get_all_repos().keys())
        if not repos:
            raise ContainerError("No repository found")

        if url is None:
            repo = choose("Choose repository you want to remove", repos)
            if not confirm(f"Remove repository `{repo}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.remove_repo(repo)

        elif url in repos:
            if not confirm(f"Remove repository `{url}`?", default=False):
                raise ContainerError("Canceled")
            _shared.manager.remove_repo(url)

        else:
            raise ContainerError(f"Repository `{url}` not found.")
