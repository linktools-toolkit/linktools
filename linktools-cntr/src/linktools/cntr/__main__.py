#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .commands._shared import iter_container_names, iter_installed_container_names, manager
from .commands.config import ConfigCommand
from .commands.exec_ import ExecCommand
from .commands.repo import RepoCommand
from .commands.root import Command

_iter_container_names = iter_container_names
_iter_installed_container_names = iter_installed_container_names

__all__ = ["manager", "RepoCommand", "ConfigCommand", "ExecCommand", "Command", "command"]

command = Command()
if __name__ == '__main__':
    command.main()
