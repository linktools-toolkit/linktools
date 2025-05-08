#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : setup.py
@time    : 2018/11/25
@site    :
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
import json
import os
import pathlib
import pkgutil
import re

import yaml
from jinja2 import Template
from setuptools import setup
from setuptools.command.editable_wheel import editable_wheel as _editable_wheel


class ConsoleScripts(list):

    def append_script(self, script_name, module_name):
        self.append(f"{script_name.replace('_', '-')} = {module_name}:command.main")
        return self

    def append_module(self, path, script_prefix, module_prefix):
        for _, module_name, _ in pkgutil.iter_modules([path]):
            if not module_name.startswith("_"):
                self.append_script(
                    f"{script_prefix}-{module_name.replace('_', '-')}",
                    f"{module_prefix}.{module_name}",
                )
        return self


class ContextVar(property):

    def __init__(self, key, default=None):
        super().__init__(
            fget=lambda o: o.get(key, default),
            fset=lambda o, v: o.__setitem__(key, v)
        )


class Context(dict):
    version: str = ContextVar("version")
    release: bool = ContextVar("release")
    develop: bool = ContextVar("develop")

    def __init__(self):
        super().__init__()
        self.develop = False
        self.release = os.environ.get("RELEASE", "false").lower() == "true"
        self.version = self.parse_version()

    @property
    def root_path(self):
        return pathlib.Path(os.path.dirname(__file__))

    @property
    def src_path(self):
        return self.root_path / "src" / "linktools"

    def parse_version(self):
        version = os.environ.get("VERSION", None) or self._read_file(self.root_path / ".version")
        if not version:
            version = "0.0.1"
        if version.startswith("v"):
            version = version[len("v"):]
        if not self.release:
            items = []
            for item in version.split("."):
                find = re.findall(r"^\d+", item)
                if find:
                    items.append(int(find[0]))
            version = ".".join(map(str, items))
            version = f"{version}.post100.dev0"
        return version

    def parse_requires(self):
        with open(self.root_path / "requirements.yml", "rt", encoding="utf-8") as fd:
            data = yaml.safe_load(fd)
            # install_requires = dependencies + dev-dependencies
            install_requires = data.get("dependencies")
            install_requires.extend(
                data.get("release-dependencies") if self.release else data.get("dev-dependencies"))
            # extras_require = optional-dependencies
            extras_require = data.get("optional-dependencies")
            all_requires = []
            for requires in extras_require.values():
                all_requires.extend(requires)
            extras_require["all"] = all_requires
        return install_requires, extras_require

    def parse_scripts(self):
        return ConsoleScripts().append_script(
            script_name="lt",
            module_name="linktools.__main__",
        ).append_module(
            self.src_path / "cli" / "commands" / "common",
            module_prefix="linktools.cli.commands.common",
            script_prefix="ct",
        ).append_module(
            self.src_path / "cli" / "commands" / "android",
            module_prefix="linktools.cli.commands.android",
            script_prefix="at",
        ).append_module(
            self.src_path / "cli" / "commands" / "ios",
            module_prefix="linktools.cli.commands.ios",
            script_prefix="it",
        )

    def rewrite_metadata(self):
        with open(self.src_path / "develop" / "metadata", "rt", encoding="utf-8") as fd_in, \
                open(self.src_path / "metadata.py", "wt", encoding="utf-8") as fd_out:
            fd_out.write(Template(fd_in.read()).render(**self))

    def rewrite_tools(self):
        with open(self.src_path / "develop" / "tools.yml", "rb") as fd_in, \
                open(self.src_path / "assets" / "tools.json", "wt") as fd_out:
            json.dump({
                key: value
                for key, value in yaml.safe_load(fd_in).items()
                if key[0].isupper()
            }, fd_out)

    def _read_file(self, path):
        try:
            with open(path, encoding="utf-8") as fd:
                return fd.read().strip()
        except:
            return None


if __name__ == '__main__':
    class EditableWheel(_editable_wheel):

        def run(self):
            context.develop = True
            context.rewrite_metadata()
            return super().run()


    context = Context()
    context.rewrite_metadata()
    context.rewrite_tools()
    install_requires, extras_require = context.parse_requires()
    console_scripts = context.parse_scripts()

    setup(
        version=context.version,
        install_requires=install_requires,
        extras_require=extras_require,
        entry_points={"console_scripts": console_scripts},
        cmdclass={"editable_wheel": EditableWheel},
    )
