#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Author    : HuJi <jihu.hj@alibaba-inc.com>
# Datetime  : 2022/1/13 下午8:53
# User      : huji
# Product   : PyCharm
# Project   : link

import re
import zipfile
from typing import Optional, Dict, Any, List

from ..types import Error

_INFO_PLIST = "Info.plist"


class IPAError(Error):
    pass


class IPA(object):

    def __init__(self, filename: str):
        self.filename = filename
        self.zip = zipfile.ZipFile(self.filename)
        self._plist = {}
        self._analysis()

    def _analysis(self):
        import plistlib
        plist_path = self.find_file(_INFO_PLIST)
        if plist_path is None:
            raise IPAError("Missing Info.plist")
        plist_data = self.zip.read(plist_path)
        self._plist[_INFO_PLIST] = plistlib.loads(plist_data)

    def find_file(self, name) -> Optional[str]:
        name_list = self.zip.namelist()
        pattern = re.compile(rf'Payload/[^/]+\.app/{name}$')
        for path in name_list:
            m = pattern.match(path)
            if m is not None:
                return m.group()
        return None

    def list_files(self) -> List[str]:
        return self.zip.namelist()

    def read_file(self, filename) -> bytes:
        try:
            return self.zip.read(filename)
        except KeyError:
            raise IPAError(f"Not found {filename}")

    def get_info_plist(self) -> Dict[str, Any]:
        return self._plist[_INFO_PLIST]

    def get_launch_storyboard_name(self) -> str:
        return self.get_info_plist().get("UILaunchStoryboardName")

    def get_display_name(self) -> str:
        return self.get_info_plist().get("CFBundleDisplayName")

    def get_bundle_id(self) -> str:
        return self.get_info_plist().get("CFBundleIdentifier")

    def get_version(self) -> str:
        return self.get_info_plist().get("CFBundleVersion")

    def get_version_string(self) -> str:
        return self.get_info_plist().get("CFBundleShortVersionString")

    def get_permissions(self) -> Dict[str, str]:
        return {k: v for k, v in self.get_info_plist().items() if k.endswith("Description")}
