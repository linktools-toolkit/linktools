#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : stub.py
@time    : 2024/8/6 16:34
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
import abc
import inspect
import pathlib
from argparse import Namespace
from typing import TYPE_CHECKING, Iterable, Optional
from urllib import parse

from . import BaseCommand, CommandParser, CommandError
from .. import utils

if TYPE_CHECKING:
    from ..types import PathType


class Updater(metaclass=abc.ABCMeta):

    def __init__(self):
        self._cwd: Optional[str] = None

    def update(
            self,
            name: str,
            index_url: str = None,
            extra_index_url: str = None,
            dependencies: "Iterable[str]" = None,
    ):
        pip_args = ["pip", "install"]
        pip_deps = f"[{','.join(dependencies)}]" if dependencies else ""
        pip_args.extend(self._append_package_args(name, pip_deps))

        if index_url:
            pip_args.append(f"--index-url={index_url}")
            url = parse.urlparse(index_url)
            if url.scheme == "http":
                pip_args.append("--trusted-host")
                pip_args.append(url.netloc)

        if extra_index_url:
            pip_args.append(f"--extra-index-url={extra_index_url}")
            url = parse.urlparse(extra_index_url)
            if url.scheme == "http":
                pip_args.append("--trusted-host")
                pip_args.append(url.netloc)

        utils.popen(
            utils.get_interpreter(), "-m", *pip_args,
            cwd=self._cwd,
        ).check_call()

    @abc.abstractmethod
    def _append_package_args(self, name: str, deps: str) -> "Iterable[str]":
        pass


class DevelopUpdater(Updater):

    def __init__(self, project_path: "PathType", max_depth: int = 2):
        super().__init__()
        self._project_path = project_path
        self._max_depth = max_depth

    def _append_package_args(self, name: str, deps: str) -> "Iterable[str]":
        self._cwd = self.get_project_url(self._project_path, self._max_depth)
        if not self._cwd:
            raise CommandError(
                f"{self._project_path} does not appear to be a Python project: "
                f"neither 'setup.py' nor 'pyproject.toml' found."
            )
        return ["--editable", f".{deps}"]

    @classmethod
    def get_project_url(cls, path: "PathType", max_depth: int) -> Optional[pathlib.Path]:
        path = pathlib.Path(path)
        for i in range(max(max_depth, 0) + 1):
            if path.is_dir():
                if (path / "pyproject.toml").exists() or (path / "setup.py").exists():
                    return path
            path = path.parent
        return None


class GitUpdater(Updater):

    def __init__(self, repository_url: str = None):
        super().__init__()
        self._repository_url = repository_url

    def _append_package_args(self, name: str, deps: str) -> "Iterable[str]":
        repository_url = self._repository_url
        if not repository_url:
            repository_url = self.get_repository_url(name)
        if not repository_url:
            raise CommandError(f"{name} has no repository url")
        return ["--ignore-installed", f"{name}{deps}@git+{repository_url.strip()}"]

    @classmethod
    def get_repository_url(cls, name: str):
        try:
            from importlib.metadata import distribution
        except ImportError:
            from importlib_metadata import distribution

        dist = distribution(name)
        for item in dist.metadata.get_all("Project-Url") or []:
            key, url = item.split(",", 1)
            if key.strip().lower() == "repository":
                return url.strip()
        return None


class PypiUpdater(Updater):

    def _append_package_args(self, name: str, deps: str) -> "Iterable[str]":
        return ["--upgrade", f"{name}{deps}"]


class UpdateCommand(BaseCommand):

    def __init__(
            self,
            name: str,
            updater: Updater,
            index_url: str = None,
            extra_index_url: str = None,
    ):
        super().__init__()
        self._update_name = name
        self._index_url = index_url
        self._extra_index_url = extra_index_url
        self._updater = updater
        self._module = self._get_caller_module(inspect.currentframe()) or self.__module__

    @classmethod
    def _get_caller_module(cls, frame):
        if not frame:
            return None
        frame = frame.f_back
        if not frame:
            return None
        module = inspect.getmodule(frame)
        if not module:
            return None
        return module.__name__

    @property
    def module(self) -> str:
        return self._module

    @property
    def update_name(self) -> str:
        return self._update_name

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("dependencies", metavar="DEPENDENCY", nargs='*', default=None)
        parser.add_argument("--index-url", metavar="INDEX_URL", default=self._index_url)
        parser.add_argument("--extra-index-url", metavar="EXTRA_INDEX_URL", default=self._extra_index_url)

    def run(self, args: Namespace) -> Optional[int]:
        if not self._updater:
            raise CommandError(f"No updater found")
        self._updater.update(
            name=self._update_name,
            dependencies=args.dependencies,
            index_url=args.index_url,
            extra_index_url=args.extra_index_url,
        )
