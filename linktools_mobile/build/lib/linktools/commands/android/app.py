#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_app.py
@time    : 2019/01/02
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
from typing import Optional

from linktools import utils
from linktools.core import environ
from linktools.cli import CommandParser
from linktools.cli.argparse import BooleanOptionalAction
from linktools.mobile.cli import AndroidCommand, AndroidNamespace
from linktools.mobile.android import App, Permission, \
    Component, Activity, Service, Receiver, Provider, IntentFilter


class PrintLevel:
    min = 0
    useless = 100
    normal = 200
    dangerous_normal = 250
    dangerous = 300
    title = 400
    max = 1000


class PrintStream(PrintLevel):

    def __init__(self, max_level=PrintLevel.max, min_level=PrintLevel.min, file=None):
        self.max = max_level
        self.min = min_level
        self.file = file

    def print(self, text: str = "", indent: int = 0, level=PrintLevel.normal):
        if not self.min <= level <= self.max:
            pass
        elif level == PrintLevel.title:
            environ.logger.info(text, style="bold", indent=indent)
        elif level == PrintLevel.dangerous:
            environ.logger.info(text, style="red bold", indent=indent)
        elif level == PrintLevel.useless:
            environ.logger.info(text, style="strike", indent=indent)
        else:
            environ.logger.info(text, indent=indent)

    def print_line(self):
        environ.logger.info("")


class PrintStreamWrapper(PrintLevel):

    def __init__(self, stream: PrintStream, max_level: int = PrintLevel.max, min_level: int = PrintLevel.min):
        self.stream = stream
        self.max_level = max_level
        self.min_level = min_level

    def print(self, text: str = "", indent: int = 0, level=PrintLevel.normal):
        if level > self.max_level:
            level = self.max_level
        elif level < self.min_level:
            level = self.min_level
        self.stream.print(text, indent=indent, level=level)

    def print_line(self):
        self.stream.print_line()

    def create(self, max_level: int = PrintLevel.max, min_level: int = PrintLevel.min):
        if max_level > self.max_level:
            max_level = self.max_level
        elif min_level < self.min_level:
            min_level = self.min_level
        return PrintStreamWrapper(self.stream, max_level=max_level, min_level=min_level)


class AppPrinter:

    def __init__(self, stream: PrintStream, app: App):
        self.app = app
        self.max_level = PrintLevel.max if self.app.enabled else PrintLevel.useless
        self.min_level = PrintLevel.min
        self.stream = PrintStreamWrapper(stream, max_level=self.max_level, min_level=self.min_level)

    def print_app(self, indent: int = 0):
        self.stream.print("App [%s]" % self.app, indent=indent, level=self.stream.title)
        self.stream.print("name=%s" % self.app.app_name, indent=indent + 4, level=self.stream.normal)
        self.stream.print("userId=%s" % self.app.user_id, indent=indent + 4, level=self.stream.normal)
        self.stream.print("gids=%s" % self.app.gids, indent=indent + 4, level=self.stream.normal)
        self.stream.print("sourceDir=%s" % self.app.source_dir, indent=indent + 4, level=self.stream.normal)
        self.stream.print("dataDir=%s" % self.app.data_dir, indent=indent + 4, level=self.stream.normal)
        self.stream.print("nativeLibraryDir=%s" % self.app.native_library_dir, indent=indent + 4,
                          level=self.stream.normal)
        self.stream.print("versionCode=%s" % self.app.version_code, indent=indent + 4, level=self.stream.normal)
        self.stream.print("versionName=%s" % self.app.version_name, indent=indent + 4, level=self.stream.normal)
        self.stream.print("enabled=%s" % self.app.enabled, indent=indent + 4, level=self.stream.normal)
        self.stream.print("system=%s" % self.app.system, indent=indent + 4, level=self.stream.normal)
        self.stream.print("debuggable=%s" % self.app.debuggable, indent=indent + 4,
                          level=self.stream.dangerous if self.app.debuggable else self.stream.normal)
        self.stream.print("allowBackup=%s" % self.app.allow_backup, indent=indent + 4,
                          level=self.stream.dangerous if self.app.allow_backup else self.stream.normal)
        self.stream.print("targetSdkVersion=%d" % self.app.target_sdk_version, indent=indent + 4,
                          level=self.stream.normal)
        self.stream.print_line()

    def print_requested_permissions(self, indent: int = 4):
        if not utils.is_empty(self.app.requested_permissions):
            stream = self.stream.create(max_level=PrintLevel.normal)
            self.stream.print("RequestedPermissions:", indent=indent, level=self.stream.title)
            for permission in self.app.requested_permissions:
                self._print_permission(stream, permission, indent=indent + 4, identity="RequestedPermission")
            self.stream.print_line()

    def print_permissions(self, indent: int = 4):
        if not utils.is_empty(self.app.permissions):
            self.stream.print("Permissions:", indent=indent, level=self.stream.title)
            for permission in self.app.permissions:
                self._print_permission(self.stream, permission, indent=indent + 4, identity="Permission")
            self.stream.print_line()

    def print_activities(self, indent: int = 4):
        if not utils.is_empty(self.app.activities):
            self.stream.print("Activities:", indent=indent, level=self.stream.title)
            for activity in self.app.activities:
                self._print_component(self.stream, self.app, activity, indent=indent + 4, identity="Activity")
            self.stream.print_line()

    def print_services(self, indent: int = 4):
        if not utils.is_empty(self.app.services):
            self.stream.print("Services:", indent=indent, level=self.stream.title)
            for service in self.app.services:
                self._print_component(self.stream, self.app, service, indent=indent + 4, identity="Service")
            self.stream.print_line()

    def print_receivers(self, indent: int = 4):
        if not utils.is_empty(self.app.receivers):
            self.stream.print("Receivers:", indent=indent, level=self.stream.title)
            for receiver in self.app.receivers:
                self._print_component(self.stream, self.app, receiver, indent=indent + 4, identity="Receiver")
            self.stream.print_line()

    def print_providers(self, indent: int = 4):
        if not utils.is_empty(self.app.providers):
            self.stream.print("Providers:", indent=indent, level=self.stream.title)
            for provider in self.app.providers:
                self._print_component(self.stream, self.app, provider, indent=indent + 4, identity="Provider")
            self.stream.print_line()

    @classmethod
    def _print_permission(cls, stream: PrintStreamWrapper, permission: Permission, indent: int = 0,
                          identity: str = None):
        if permission.is_defined():
            stream.print("%s [%s] %s" % (identity, permission, permission.protection), indent=indent,
                         level=stream.dangerous if permission.is_dangerous() else stream.normal)

    @classmethod
    def _print_component(cls, stream: PrintStreamWrapper, app: App, component: Component, indent: int = 0, identity: str = None):
        if not component.enabled:
            description = "disabled"
            level = stream.useless
            stream = stream.create(max_level=stream.useless)
        elif component.is_dangerous():
            description = "exported"
            level = stream.dangerous if component.is_dangerous() else stream.normal
            stream = stream.create(min_level=stream.dangerous_normal)
        else:
            description = "exported" if component.exported else ""
            level = stream.normal
            stream = stream.create(max_level=stream.normal)
        stream.print("%s [%s/%s] %s" % (identity, app, component, description), indent=indent, level=level)

        if isinstance(component, Activity) or isinstance(component, Service) or isinstance(component, Receiver):
            cls._print_permission(stream, component.permission, indent=indent + 4, identity="Permission")
        elif isinstance(component, Provider):
            stream.print("Authority [%s]" % component.authority, indent=indent + 4, level=level)
            cls._print_permission(
                stream, component.read_permission, indent=indent + 4,
                identity="ReadPermission")
            cls._print_permission(
                stream, component.write_permission, indent=indent + 4,
                identity="writePermission")
            for pattern in component.uri_permission_patterns:
                stream.print("UriPermissionPattern [%s]" % pattern, indent=indent + 4, level=level)
            for permission in component.path_permissions:
                stream.print(
                    "PathPermission [%s]" % permission, indent=indent + 4,
                    level=stream.dangerous if permission.is_dangerous() else stream.normal)
                cls._print_permission(
                    stream, permission.read_permission, indent=indent + 8,
                    identity="ReadPermission")
                cls._print_permission(
                    stream, permission.write_permission, indent=indent + 8,
                    identity="writePermission")

        if not utils.is_empty(component.intents):
            for intent in component.intents:
                cls._print_intent(stream, intent, indent=indent + 4, level=level)

    @classmethod
    def _print_intent(cls, stream: PrintStreamWrapper, intent: IntentFilter, indent: int = 0,
                      level: int = PrintLevel.normal):
        stream.print("IntentFilter:", indent=indent, level=level)
        for action in intent.actions:
            stream.print("Action [%s]" % action, indent=indent + 4, level=level)
        for category in intent.categories:
            stream.print("Category [%s]" % category, indent=indent + 4, level=level)
        for scheme in intent.data_schemes:
            stream.print("Scheme [%s]" % scheme, indent=indent + 4, level=level)
        for scheme in intent.data_scheme_specific_parts:
            stream.print("Scheme [%s]" % scheme, indent=indent + 4, level=level)
        for authority in intent.data_authorities:
            stream.print("Authority [%s]" % authority, indent=indent + 4, level=level)
        for path in intent.data_paths:
            stream.print("Path [%s]" % path, indent=indent + 4, level=level)
        for type in intent.data_types:
            stream.print("Type [%s]" % type, indent=indent + 4, level=level)


class Command(AndroidCommand):
    """
    Retrieve detailed information about installed applications on Android devices
    """

    def init_arguments(self, parser: CommandParser) -> None:
        group = parser.add_mutually_exclusive_group()
        group.add_argument('-t', '--top', action='store_true', default=False,
                           help='fetch current running app only')
        group.add_argument('-a', '--all', action='store_true', default=False,
                           help='fetch all apps')
        group.add_argument('-p', '--packages', metavar="pkg", action='store', nargs='+', default=None,
                           help='fetch target apps only')
        group.add_argument('-u', '--uids', metavar="uid", action='store', nargs='+', type=int, default=None,
                           help='fetch apps with specified uids only')
        group.add_argument('--system', action=BooleanOptionalAction, default=None,
                           help='fetch system/non-system apps only')

        parser.add_argument('--detail', action='store_true', default=False,
                            help='show app detail info')
        parser.add_argument('--dangerous', action='store_true', default=False,
                            help='show app dangerous permissions and components only')

    def run(self, args: AndroidNamespace) -> Optional[int]:
        device = args.device_selector.select()

        if not utils.is_empty(args.packages):
            apps = device.get_apps(*args.packages, detail=args.detail)
        elif not utils.is_empty(args.uids):
            apps = device.get_apps_for_uid(*args.uids, detail=args.detail)
        elif args.system is not None:
            apps = device.get_apps(system=args.system, detail=args.detail)
        elif args.all:
            apps = device.get_apps(detail=args.detail)
        else:
            apps = device.get_apps(device.get_current_package(), detail=args.detail)
        apps = sorted(apps, key=lambda o: tuple(getattr(o, k) for k in ['user_id', 'name']))

        min_level = PrintLevel.min
        if args.dangerous:
            min_level = PrintLevel.dangerous_normal
        stream = PrintStream(min_level=min_level)

        for app in apps:
            printer = AppPrinter(stream, app)
            if not args.dangerous:
                printer.print_app()
                printer.print_requested_permissions()
                printer.print_permissions()
                printer.print_activities()
                printer.print_services()
                printer.print_receivers()
                printer.print_providers()
                continue

            if app.is_dangerous():
                printer.print_app()
                if app.has_dangerous_permission():
                    printer.print_permissions()
                if app.has_dangerous_activity():
                    printer.print_activities()
                if app.has_dangerous_service():
                    printer.print_services()
                if app.has_dangerous_receiver():
                    printer.print_receivers()
                if app.has_dangerous_provider():
                    printer.print_providers()

        return


command = Command()
if __name__ == "__main__":
    command.main()
