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
    from typing import Optional, Callable, List, Iterable

    from .. import BaseEnviron
    from .command import SubCommand, CommandParser


def get_commands(environ: "BaseEnviron") -> "Iterable[SubCommand]":
    import argparse
    import os

    from .. import utils, environ as _environ
    from ..metadata import __release__, __develop__, __ep_updater__
    from .command import SubCommand, iter_entry_point_commands
    from .update import DevelopUpdater, GitUpdater, PypiUpdater

    commands: "List[SubCommand]" = []

    def register_command(name: str, description: str):
        def wrapper(cls: "Callable[[str, str], SubCommand]"):
            commands.append(cls(name, description))
            return None

        return wrapper

    def get_stub_path() -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            utils.get_interpreter_ident(),
            f"env_v{environ.version}",
        )

    def get_alias_path() -> "pathlib.Path":
        return environ.get_data_path(
            "scripts",
            utils.get_interpreter_ident(),
            f"alias_v{environ.version}",
        )

    def get_default_shell(environ: "BaseEnviron") -> "Optional[str]":
        return "bash" if environ.system != "windows" else "powershell"

    @register_command(name="shell", description="run shell command")
    class ShellCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-c", "--command", help="shell command", default=None)
            return parser

        def run(self, args: "argparse.Namespace"):
            shell = environ.tools["shell"]
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
                return utils.popen(args.command, shell=True, append_env=env).call()

            return shell.popen(append_env=env).call()

    @register_command(name="alias", description="generate shell alias script")
    class AliasCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"], default=None)
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

            from .. import metadata
            from ..cli import commands
            from ..cli.argparse import ArgParseComplete
            from ..cli.command import iter_module_commands, iter_entry_point_commands
            from .._tools import Tool

            stub_path = get_stub_path()
            stub_path.mkdir(parents=True, exist_ok=True)
            utils.clear_directory(stub_path)

            executables = []
            command_infos = {
                command_info.id: command_info
                for command_info in (
                    *iter_module_commands(commands, onerror="warn"),
                    *iter_entry_point_commands(metadata.__ep_scripts__, onerror="warn")
                )
            }
            for command_info in command_infos.values():
                if command_info.command:
                    temp = command_info
                    names = [command_info.command_name]
                    while temp.parent_id in command_infos:
                        temp = command_infos[temp.parent_id]
                        names.append(temp.command_name)
                    executable = "-".join(reversed(names))
                    executables.append(executable)
                    Tool.create_stub_file(
                        stub_path / Tool.get_stub_name(executable),
                        utils.list2cmdline([utils.get_interpreter(), "-m", command_info.module]),
                        system=environ.system
                    )
                    environ.logger.info(f"Found alias: {executable} -> {command_info.module}")

            lines = []
            completion = ArgParseComplete.shellcode(executables, shell=shell)
            if completion:
                environ.logger.info("Generate completion script ...")
                lines.append(completion)

            tools_path = environ.tools.stub_path
            if shell in ("bash", "zsh"):
                lines.append(f"export PATH=\"$PATH:{stub_path}:{tools_path}\"")
            elif shell in ("fish",):
                lines.append(f"set -x PATH \"$PATH\" \"{stub_path}\" \"{tools_path}\"")
            elif shell in ("tcsh",):
                lines.append(f"setenv PATH \"$PATH:{stub_path}:{tools_path}\"")
            elif shell in ("powershell",):
                lines.append(f"$env:PATH=\"$env:PATH;{stub_path}:{tools_path}\"")

            result = os.linesep.join(lines)
            utils.write_file(alias_path, result)
            print(result, flush=True)

    @register_command(name="completion", description="generate shell auto complete script (deprecated)")
    class CompletionCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"])
            return parser

        def run(self, args: "argparse.Namespace"):
            environ.logger.warning("Not support generate completion script, already integrated into alias subcommand")

    @register_command(name="java", description="generate java environment script")
    class JavaCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("-s", "--shell", help="output code for the specified shell",
                                choices=["bash", "zsh", "tcsh", "fish", "powershell"])
            parser.add_argument("version", metavar="VERSION", nargs="?",
                                help="java version, such as 11.0.23 / 17.0.11 / 22.0.1")
            return parser

        def run(self, args: "argparse.Namespace"):
            java = environ.tools["java"]
            if args.version:
                java = java.copy(version=args.version)

            cmdline = java.make_stub_cmdline("java")
            shell = args.shell or get_default_shell(environ)

            lines = []
            if shell in ("bash", "zsh"):
                lines.append(f"alias java='{cmdline}'")
                lines.append(f"export JAVA_VERSION='{java.get('version')}'")
                lines.append(f"export JAVA_HOME='{java.get('home_path')}'")
                lines.append(f"export PATH=\"$JAVA_HOME/bin:$PATH\"")
            elif shell in ("fish",):
                lines.append(f"alias java '{cmdline}'")
                lines.append(f"set -x JAVA_VERSION '{java.get('version')}'")
                lines.append(f"set -x JAVA_HOME '{java.get('home_path')}'")
                lines.append(f"set -x PATH \"$JAVA_HOME/bin\" \"$PATH\"")
            elif shell in ("tcsh",):
                lines.append(f"alias java '{cmdline}'")
                lines.append(f"setenv JAVA_VERSION '{java.get('version')}'")
                lines.append(f"setenv JAVA_HOME '{java.get('home_path')}'")
                lines.append(f"setenv PATH \"$JAVA_HOME/bin:$PATH\"")
            elif shell in ("powershell",):
                lines.append(f"function __tool_java__ {{ {cmdline} $args }}")
                lines.append(f"Set-Alias -Name java -Value __tool_java__")
                lines.append(f"$env:JAVA_VERSION='{java.get('version')}'")
                lines.append(f"$env:JAVA_HOME='{java.get('home_path')}'")
                lines.append(f"$env:PATH=\"$env:JAVA_HOME\\bin;$env:PATH\"")

            result = os.linesep.join(lines)
            print(result, flush=True)

    if environ is _environ:

        @register_command(name="update", description=f"update {_environ.name} packages")
        class UpdateCommand(SubCommand):

            def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
                parser = super().create_parser(type)
                parser.add_argument("dependencies", metavar="DEPENDENCY", nargs='*', default=None)
                return parser

            def run(self, args: "argparse.Namespace"):
                try:
                    environ.logger.info("Update main packages ...")
                    updater = utils.coalesce(*[
                        DevelopUpdater(environ.root_path) if __develop__ else None,
                        GitUpdater() if not __release__ else None,
                        PypiUpdater()
                    ])
                    updater.update(environ.name, dependencies=args.dependencies or [])

                    for command_info in iter_entry_point_commands(__ep_updater__, onerror="warn"):
                        environ.logger.info(f"Update package through {command_info.module} ...")
                        command_info.command([])
                except Exception as e:
                    _environ.logger.warning(f"Update {_environ.name} packages failed: {e}")

    @register_command(name="clean", description="clean temporary files")
    class CleanCommand(SubCommand):

        def create_parser(self, type: "Callable[..., CommandParser]") -> "CommandParser":
            parser = super().create_parser(type)
            parser.add_argument("days", metavar="DAYS", nargs="?", type=int, default=7, help="expire days")
            return parser

        def run(self, args: "argparse.Namespace"):
            environ.clean_temp_files(expire_days=args.days)

    return commands


if __name__ == '__main__':
    import functools
    import logging
    from .command import BaseCommand, CommandParser, CommandMain


    class Command(BaseCommand):

        @property
        def main(self) -> "CommandMain":

            class Main(CommandMain):

                def init_logging(self):
                    logging.basicConfig(level=logging.CRITICAL)

            return Main(self)

        def init_base_arguments(self, parser: "argparse.ArgumentParser"):
            pass

        def init_global_arguments(self, parser: "argparse.ArgumentParser") -> None:
            pass

        def init_arguments(self, parser: "argparse.ArgumentParser") -> None:
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

        def run(self, args: "argparse.Namespace") -> "Optional[int]":
            if args.verbose:
                self.logger.level = logging.DEBUG
            return args.func(args)

        def on_tool(self, args: "argparse.Namespace"):
            return self.environ.get_tool(args.name, cmdline=None) \
                .popen(*args.args) \
                .call()


    command = Command()
    command.main()
