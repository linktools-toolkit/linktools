#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : struct.py
@time    : 2019/01/11
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


class PatternMatcher:

    def __init__(self, obj: dict):
        self.path = utils.get_item(obj, "path", type=str, default="")
        self.type = utils.get_item(obj, "type", type=str, default="literal")

    def __str__(self):
        return f"path={self.path}, type={self.type}"

    def __repr__(self):
        return f"PatternMatcher<{self.path}>"


class PathPermission(PatternMatcher):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.read_permission = utils.get_item(obj, "readPermission", type=Permission, default=Permission.default())
        self.write_permission = utils.get_item(obj, "writePermission", type=Permission, default=Permission.default())

    def is_dangerous(self):
        return self.read_permission.is_dangerous() or self.write_permission.is_dangerous()

    def __repr__(self):
        return f"PathPermission<{self.path}>"


class AuthorityEntry:

    def __init__(self, obj: dict):
        self.host = utils.get_item(obj, "host", type=str, default="")
        self.port = utils.get_item(obj, "port", type=int, default=0)

    def __str__(self):
        return f"host={self.host}, port={self.port}"

    def __repr__(self):
        return f"AuthorityEntry<{self.host}>"


class IntentFilter:

    def __init__(self, obj: dict):
        self.actions = utils.get_list_item(obj, "actions", type=str, default=[])
        self.categories = utils.get_list_item(obj, "categories", type=str, default=[])
        self.data_schemes = utils.get_list_item(obj, "dataSchemes", type=str, default=[])
        self.data_scheme_specific_parts = utils.get_list_item(obj, "dataSchemeSpecificParts", type=PatternMatcher,
                                                              default=[])
        self.data_authorities = utils.get_list_item(obj, "dataAuthorities", type=AuthorityEntry, default=[])
        self.data_paths = utils.get_list_item(obj, "dataPaths", type=PatternMatcher, default=[])
        self.data_types = utils.get_list_item(obj, "dataTypes", type=str, default=[])

    def __repr__(self):
        return f"IntentFilter<{self.actions}>"


class Permission:

    @staticmethod
    def default() -> "Permission":
        return Permission({"name": "", "protection": "normal"})

    def __init__(self, obj: dict):
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.protection = utils.get_item(obj, "protection", type=str, default="normal")

    def is_defined(self):
        return not utils.is_empty(self.name)

    def is_dangerous(self):
        return self.protection in ["dangerous", "normal"]

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Permission<{self.name}>"


class Component:

    def __init__(self, obj: dict):
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.exported = utils.get_item(obj, "exported", type=bool, default=False)
        self.enabled = utils.get_item(obj, "enabled", type=bool, default=False)
        self.intents = utils.get_list_item(obj, "intents", type=IntentFilter, default=[])

    def is_dangerous(self):
        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Component<{self.name}>"


class Activity(Component):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.permission = utils.get_item(obj, "permission", type=Permission, default=Permission.default())

    def is_dangerous(self):
        return self.enabled and self.exported and self.permission.is_dangerous()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Activity<{self.name}>"


class Service(Component):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.permission = utils.get_item(obj, "permission", type=Permission, default=Permission.default())

    def is_dangerous(self):
        return self.enabled and self.exported and self.permission.is_dangerous()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Service<{self.name}>"


class Receiver(Component):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.permission = utils.get_item(obj, "permission", type=Permission, default=Permission.default())

    def is_dangerous(self):
        return self.enabled and self.exported and self.permission.is_dangerous()

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Receiver<{self.name}>"


class Provider(Component):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.authority = utils.get_item(obj, "authority", type=str, default="")
        self.read_permission = utils.get_item(obj, "readPermission", type=Permission, default=Permission.default())
        self.write_permission = utils.get_item(obj, "writePermission", type=Permission, default=Permission.default())
        self.uri_permission_patterns = utils.get_list_item(obj, "uriPermissionPatterns", type=PatternMatcher,
                                                           default=[])
        self.path_permissions = utils.get_list_item(obj, "pathPermissions", type=PathPermission, default=[])

    def is_dangerous(self):
        if self.enabled and self.exported:
            if self.read_permission.is_dangerous() or self.write_permission.is_dangerous():
                return True
            for path_permission in self.path_permissions:
                if path_permission.is_dangerous():
                    return True
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Provider<{self.name}>"


class App:

    def __init__(self, obj: dict):
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.app_name = utils.get_item(obj, "appName", type=str, default="")
        self.user_id = utils.get_item(obj, "userId", type=int, default=0)
        self.gids = utils.get_list_item(obj, "gids", type=int, default=[])
        self.source_dir = utils.get_item(obj, "sourceDir", type=str, default="")
        self.data_dir = utils.get_item(obj, "dataDir", type=str, default="")
        self.native_library_dir = utils.get_item(obj, "nativeLibraryDir", type=str, default="")
        self.version_code = utils.get_item(obj, "versionCode", type=str, default="")
        self.version_name = utils.get_item(obj, "versionName", type=str, default="")
        self.enabled = utils.get_item(obj, "enabled", type=bool, default=False)
        self.system = utils.get_item(obj, "system", type=bool, default=False)
        self.debuggable = utils.get_item(obj, "debuggable", type=bool, default=False)
        self.allow_backup = utils.get_item(obj, "allowBackup", type=bool, default=False)
        self.target_sdk_version = utils.get_item(obj, "targetSdkVersion", type=int, default=0)

        self.requested_permissions = utils.get_list_item(obj, "requestedPermissions", type=Permission, default=[])
        self.permissions = utils.get_list_item(obj, "permissions", type=Permission, default=[])
        self.activities = utils.get_list_item(obj, "activities", type=Activity, default=[])
        self.services = utils.get_list_item(obj, "services", type=Service, default=[])
        self.receivers = utils.get_list_item(obj, "receivers", type=Receiver, default=[])
        self.providers = utils.get_list_item(obj, "providers", type=Provider, default=[])

    def get_launch_activity(self) -> Optional[Activity]:
        for activity in self.activities:
            for intent in activity.intents:
                if "android.intent.action.MAIN" in intent.actions and \
                        "android.intent.category.INFO" in intent.categories:
                    return activity

        for activity in self.activities:
            for intent in activity.intents:
                if "android.intent.action.MAIN" in intent.actions and \
                        "android.intent.category.LAUNCHER" in intent.categories:
                    return activity

        return None

    def is_dangerous(self):
        return self.enabled and (
                self.debuggable or
                self.allow_backup or
                self.has_dangerous_permission() or
                self.has_dangerous_activity() or
                self.has_dangerous_service() or
                self.has_dangerous_receiver() or
                self.has_dangerous_provider()
        )

    def has_dangerous_permission(self):
        for permission in self.permissions:
            if permission.is_dangerous():
                return True
        return False

    def has_dangerous_activity(self):
        for activity in self.activities:
            if activity.is_dangerous():
                return True
        return False

    def has_dangerous_service(self):
        for service in self.services:
            if service.is_dangerous():
                return True
        return False

    def has_dangerous_receiver(self):
        for receiver in self.receivers:
            if receiver.is_dangerous():
                return True
        return False

    def has_dangerous_provider(self):
        for provider in self.providers:
            if provider.is_dangerous():
                return True
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"App<{self.name}>"


class Socket:

    def __init__(self, obj: dict):
        self.proto = utils.get_item(obj, "proto", type=str, default="")
        self.state = utils.get_item(obj, "state", type=str, default="")
        self.inode = utils.get_item(obj, "inode", type=int, default=0)
        self.listening = utils.get_item(obj, "listening", type=bool, default=False)

    def is_dangerous(self):
        return self.listening

    def __repr__(self):
        return f"Socket<{self.proto}>"


class InetSocket(Socket):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.local_address = utils.get_item(obj, "localAddress", type=str, default="")
        self.local_port = utils.get_item(obj, "localPort", type=int, default=0)
        self.remote_address = utils.get_item(obj, "remoteAddress", type=str, default="")
        self.remote_port = utils.get_item(obj, "remotePort", type=int, default=0)
        self.uid = utils.get_item(obj, "uid", type=int, default=0)
        self.transmit_queue = utils.get_item(obj, "transmitQueue", type=int, default=0)
        self.receive_queue = utils.get_item(obj, "receiveQueue", type=int, default=0)

    def __repr__(self):
        return f"InetSocket<{self.local_address}:{self.local_port}>"


class UnixSocket(Socket):

    def __init__(self, obj: dict):
        super().__init__(obj)
        self.ref_cnt = utils.get_item(obj, "refCnt", type=int, default=0)
        self.flags = utils.get_item(obj, "flags", type=str, default="")
        self.type = utils.get_item(obj, "type", type=str, default=[])
        self.path = utils.get_item(obj, "path", type=str, default="")
        self.readable = utils.get_item(obj, "readable", type=bool, default=False)
        self.writable = utils.get_item(obj, "writable", type=bool, default=False)

    def is_dangerous(self):
        return self.listening and (self.readable or self.writable)

    def __repr__(self):
        return f"UnixSocket<{self.path}>"


class Process:

    def __init__(self, obj: dict):
        self.pid = utils.get_item(obj, "pid", type=int, default=0)
        self.uid = utils.get_item(obj, "uid", type=int, default=0)
        self.gid = utils.get_item(obj, "gid", type=int, default=0)
        self.state = utils.get_item(obj, "state", type=str, default="")
        self.cmd = utils.get_item(obj, "cmd", type=str, default="")
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.ppid = utils.get_item(obj, "ppid", type=int, default=0)
        self.pgid = utils.get_item(obj, "pgid", type=int, default=0)
        self.sid = utils.get_item(obj, "sid", type=int, default=0)
        self.tty = utils.get_item(obj, "tty", type=int, default=0)
        self.utime = utils.get_item(obj, "utime", type=int, default=0)
        self.stime = utils.get_item(obj, "stime", type=int, default=0)
        self.nice = utils.get_item(obj, "nice", type=int, default=0)
        self.start_time = utils.get_item(obj, "startTime", type=int, default=0)
        self.vsz = utils.get_item(obj, "vsz", type=int, default=0)
        self.rss = utils.get_item(obj, "rss", type=int, default=0)

    def __repr__(self):
        return f"Process<{self.name}>"


class File:

    def __init__(self, obj: dict):
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.path = utils.get_item(obj, "path", type=str, default="")
        self.is_directory = utils.get_item(obj, "isDirectory", type=bool, default=False)
        self.readable = utils.get_item(obj, "readable", type=bool, default=False)
        self.writable = utils.get_item(obj, "writable", type=bool, default=False)
        self.executable = utils.get_item(obj, "executable", type=bool, default=False)

    def __repr__(self):
        return f"File<{self.path}>"


class SystemService:

    def __init__(self, obj: dict):
        self.name = utils.get_item(obj, "name", type=str, default="")
        self.desc = utils.get_item(obj, "desc", type=str, default="")
        self.binder = utils.get_item(obj, "binder", type=str, default="")
        self.owner = utils.get_item(obj, "owner", type=Process, default=None)
        self.users = utils.get_list_item(obj, "users", type=Process, default=[])

    def __repr__(self):
        return f"SystemService<{self.name}>"
