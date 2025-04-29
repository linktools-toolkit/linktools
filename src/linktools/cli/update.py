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

from typing import TYPE_CHECKING, Iterable
from urllib import parse

from . import iter_entry_point_commands
from .. import utils
from .._environ import environ
from ..metadata import __release__, __develop__, __ep_updater__

if TYPE_CHECKING:
    from typing import TypeVar

    T = TypeVar("T")


def get_repository_url(name: str):
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


def update(
        name: str,
        develop: bool,
        release: bool,
        *,
        project_path: str = None,
        repository_url: str = None,
        extra_index_url: str = None,
        dependencies: "Iterable[str]" = None
) -> None:
    pip_args = ["pip", "install"]
    pip_deps = f"[{','.join(dependencies)}]" if dependencies else ""
    pip_cwd = project_path

    if develop:
        pip_args.append("--editable")
        pip_args.append(f".{pip_deps}")

    elif not release:
        if not repository_url:
            repository_url = get_repository_url(name)
        if not repository_url:
            pip_args.append("--upgrade")
            pip_args.append(f"{name}{pip_deps}")
        else:
            pip_args.append("--ignore-installed")
            pip_args.append(f"{name}{pip_deps}@git+{repository_url.strip()}")

    else:
        pip_args.append("--upgrade")
        pip_args.append(f"{name}{pip_deps}")

    if extra_index_url:
        pip_args.append(f"--extra-index-url={extra_index_url}")
        url = parse.urlparse(extra_index_url)
        if url.scheme == "http":
            pip_args.append("--trusted-host")
            pip_args.append(url.netloc)

    utils.popen(
        utils.get_interpreter(), "-m", *pip_args,
        cwd=pip_cwd,
    ).check_call()


def update_all(dependencies: "Iterable[str]"):
    environ.logger.info("Update main packages ...")
    update(
        environ.name,
        develop=__develop__,
        release=__release__,
        project_path=environ.root_path.parent.parent,
        dependencies=dependencies,
    )

    for command_info in iter_entry_point_commands(__ep_updater__, onerror="warn"):
        environ.logger.info(f"Update package through {command_info.module} ...")
        command_info.command()
