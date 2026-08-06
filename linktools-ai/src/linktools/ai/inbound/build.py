#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Build command input adapter."


class BuildCommand:
    def __init__(self, application: object) -> None:
        self._application = application

    def execute(self, request: object) -> object:
        return self._application.execute(request)
