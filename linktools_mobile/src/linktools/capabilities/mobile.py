#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib

from linktools.core import Capability as _Capability


class Capability(_Capability):

    def __init__(self):
        super().__init__()
        self._root_path = pathlib.Path(os.path.dirname(os.path.dirname(__file__)))

    @property
    def name(self) -> str:
        return "linktools-mobile"

    @property
    def version(self) -> str:
        return "0.9.0.post100.dev0"

    @property
    def develop(self) -> bool:
        return False

    @property
    def release(self) -> bool:
        return False

    @property
    def root_path(self) -> pathlib.Path:
        return self._root_path


__capability__ = Capability()