#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .container import BaseContainer


class EventContext:

    def __init__(self):
        self.commands: "list[str] | None" = None
        self.containers: "list[BaseContainer] | None" = None
        self.target_containers: "list[BaseContainer] | None" = None
        self.is_full_containers: bool = True
