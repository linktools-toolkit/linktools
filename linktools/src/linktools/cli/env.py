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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    import pathlib
    from collections.abc import Callable, Iterable

    from ..core import BaseEnviron
    from .command import SubCommand, CommandParser


def get_commands(environ: "BaseEnviron") -> "Iterable[SubCommand]":
    """Build environment management subcommands for an environment.

    Args:
        environ (BaseEnviron): The environ value.

    Returns:
        Iterable[SubCommand]: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    import os
    import re

    from .. import utils, metadata
    from ..core import environ as _environ
    from ..system import (
        get_interpreter, get_interpreter_ident,
        SUPPORTED_SHELLS, ShellScript, get_default_shell as _detect_default_shell,
        CommandStub,
    )
    from ..runtime import popen
    from .command import SubCommand, CommandError, iter_entry_points_capabilities

    commands: "list[SubCommand]" = []

    def register_command(name: str, description: str):
        def wrapper(cls: "Callable[[str, str], SubCommand]"):
            commands.append(cls(name, description))
            return None

        return wrapper

    def get_stub_path() -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            get_interpreter_ident(),
            f"env_v{environ.version}",
        )

    def get_alias_path() -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            get_interpreter_ident(),
            f"alias_v{environ.version}",
        )

    def remove_cache_files():
        stub_path = get_stub_path()
        if os.path.exists(stub_path):
            environ.logger.info(f"Remove stub path {stub_path} ...")
            utils.remove_file(stub_path)

        alias_path = get_alias_path()
        if os.path.exists(alias_path):
            environ.logger.info(f"Remove alias path {alias_path} ...")
            utils.remove_file(alias_path)

    def get_default_shell(environ: "BaseEnviron") -> str:
        return _detect_default_shell(system=environ.system)

    @register_command(name="shell", description="run shell command")
    class ShellCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-c", "--command", help="shell command", default=None)
            return parser

        def run(self, args: "argparse.Namespace"):
            shell = environ.get_tool("shell")
            if not shell.exists:
                raise NotImplementedError(f"Not found shell path")

            paths = os.environ.get("PATH", "").split(os.pathsep)
            stub_path = str(get_stub_path())
            if stub_path not in paths:
                paths.append(stub_path)
            stub_path = str(environ.tools.stub_path)
            if stub_path not in paths:
                paths.append(stub_path)

            env = dict(PATH=os.pathsep.join(paths))
            if args.command:
                return popen(args.command, shell=True, append_env=env).call()

            return shell.popen(append_env=env).call()

    @register_command(name="alias", description="generate shell alias script")
    class AliasCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=list(SUPPORTED_SHELLS), default=None)
            parser.add_argument("--reload", action="store_true", help="reload alias script", default=False)
            return parser

        def run(self, args: "argparse.Namespace"):
            shell = args.shell or get_default_shell(environ)
            alias_path = get_alias_path() / f"alias.{shell}"
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            if not args.reload and os.path.exists(alias_path):
                environ.logger.info(f"Found alias script: {alias_path}")
                print(utils.read_file(alias_path, text=True), flush=True)
                return 0

            from ..cli.argparse import ArgParseComplete
            from ..cli.command import iter_entry_point_commands

            stub_path = get_stub_path()
            stub_path.mkdir(parents=True, exist_ok=True)
            utils.clear_directory(stub_path)

            executables = []
            command_infos = {
                command_info.id: command_info
                for command_info in iter_entry_point_commands(metadata.__scripts_group__, onerror="warn")
            }
            for command_info in command_infos.values():
                if command_info.command:
                    temp = command_info
                    names = [command_info.command_name]
                    while temp.parent_id in command_infos:
                        temp = command_infos[temp.parent_id]
                        names.append(temp.command_name)
                    executable = "-".join(reversed(names))
                    CommandStub(stub_path, executable, system=environ.system).write(
                        [get_interpreter(), "-m", command_info.module]
                    )
                    environ.logger.info(f"Found alias: {executable} -> {command_info.module}")
                    executables.append(executable)

            script = ShellScript(shell)
            script.append_path([str(stub_path), str(environ.tools.stub_path)])

            completion = ArgParseComplete.shellcode(executables, shell=shell)
            if completion:
                environ.logger.info("Generate completion script ...")
                script.add_raw(completion)

            result = script.render()
            utils.write_file(alias_path, result)
            print(result, flush=True)

    @register_command(name="java", description="generate java environment script")
    class JavaCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=list(SUPPORTED_SHELLS))
            parser.add_argument("version", metavar="VERSION", nargs="?",
                                help="java version, such as 11.0.23 / 17.0.11 / 22.0.1")
            return parser

        def run(self, args: "argparse.Namespace"):
            java = environ.get_tool("java")
            if args.version:
                java = java.copy(version=args.version)

            shell = args.shell or get_default_shell(environ)
            home_path = java.get("home_path")

            # Structured intent only -- ShellScript owns quoting/$PATH syntax.
            # PATH gets the real bin path, never a "$JAVA_HOME/bin" expression.
            script = ShellScript(shell)
            script.define_command("java", java.make_cmdargs())
            script.set_env("JAVA_VERSION", java.get("version"))
            if home_path:
                script.set_env("JAVA_HOME", home_path)
                script.prepend_path([os.path.join(str(home_path), "bin")])

            result = script.render()
            print(result, flush=True)

    if environ is _environ:

        @register_command(name="update", description=f"update {environ.name} packages")
        class _UpdateCommand(SubCommand):

            def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
                parser = super().create_parser(type)
                parser.add_argument("packages", nargs='*', help=f"such as `{environ.name}[all]`")
                parser.add_argument("--no-build-isolation", action="store_true",
                                    help="Disable isolation when building a modern source distribution")
                return parser

            def run(self, args: "argparse.Namespace"):

                def unify_name(name):
                    return name.lower().replace("-", "_") if name else name

                def parse_package(package):
                    match = re.match(r"^([a-zA-Z0-9_-]+)(?:\[([a-zA-Z0-9_,-]+)\])?$", package)
                    if not match:
                        raise CommandError(f"Invalid package: {package}")
                    name, deps = match.group(1), match.group(2)
                    return unify_name(name), deps.split(",") if deps else []

                def get_package(name):
                    name = unify_name(name)
                    if not args.packages:  # Update all packages when no package arguments are provided.
                        return name, name, ""
                    if name in packages:
                        deps = f"[{','.join(packages[name])}]"
                        return f"{name}{deps}" if deps else name, name, deps
                    return None, None, None

                packages = {}
                if args.packages:
                    packages = {environ.name: set()}
                    for package in args.packages:
                        name, deps = parse_package(package)
                        if name:
                            packages.setdefault(name, set()).update(deps)

                pip_packages = list()
                pip_index_urls = set()
                pip_index_trusted_host = set()

                for module in iter_entry_points_capabilities(metadata.__capability_group__, onerror="warn"):
                    package, name, deps = get_package(module.name)
                    if package:
                        environ.logger.info(f"Update {package} ...")
                        updater = module.updater
                        pip_packages.extend(updater.get_packages(name, deps))
                        index_urls = updater.get_index_urls()
                        if index_urls:
                            for index_url in updater.get_index_urls():
                                from urllib import parse
                                pip_index_urls.add(index_url)
                                url = parse.urlparse(index_url)
                                if url.scheme == "http":
                                    pip_index_trusted_host.add(url.netloc)

                if not pip_packages:
                    raise CommandError("No package to update")

                pip_args = ["pip", "install"]
                for package in pip_packages:
                    pip_args.append(package)
                for index_url in pip_index_urls:
                    pip_args.extend(["--extra-index-url", index_url])
                for index_host in pip_index_trusted_host:
                    pip_args.extend(["--trusted-host", index_host])
                if args.no_build_isolation:
                    pip_args.append("--no-build-isolation")

                remove_cache_files()
                return popen(get_interpreter(), "-m", *pip_args).check_call()

    @register_command(name="clean", description="clean temporary files")
    class CleanCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("days", metavar="DAYS", nargs="?", type=int, default=7, help="expire days")
            return parser

        def run(self, args: "argparse.Namespace"):
            remove_cache_files()
            environ.clean_temp_files(expire_days=args.days)

    return commands


if __name__ == '__main__':
    import functools
    import logging
    from .command import BaseCommand, CommandParser, CommandMain


    class Command(BaseCommand):

        @property
        def main(self) -> "CommandMain":

            environ = self.environ

            class Main(CommandMain):

                def init_logging(self):
                    environ.logging.bootstrap()

            return Main(self)

        def init_base_arguments(self, parser: "CommandParser"):
            pass

        def init_global_arguments(self, parser: "CommandParser") -> None:
            pass

        def init_arguments(self, parser: "CommandParser") -> None:
            parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")

            command_parser = parser.add_subparsers(metavar="COMMAND", help="Command Help")
            command_parser.required = True

            for command in get_commands(self.environ):
                env_parser = command.create_parser(functools.partial(command_parser.add_parser, command=self))
                env_parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
                env_parser.set_defaults(func=command.run)

            tool_parser = command_parser.add_parser("tool")
            tool_parser.add_argument("name", help="tool Name").required = True
            tool_parser.add_argument("args", nargs="...", help="tool Args")
            tool_parser.set_defaults(func=self.on_tool)

        def run(self, args: "argparse.Namespace") -> "int | None":
            if args.verbose:
                self.logger.level = logging.DEBUG
            return args.func(args)

        def on_tool(self, args: "argparse.Namespace"):
            return self.environ.get_tool(args.name, cmdline=None) \
                .popen(*args.args) \
                .call()


    command = Command()
    command.main()
