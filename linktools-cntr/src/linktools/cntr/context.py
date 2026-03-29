#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional

from .container import BaseContainer


class EventContext:

    def __init__(self):
        self.commands: Optional[List[str]] = None
        self.containers: Optional[List[BaseContainer]] = None
        self.target_containers: Optional[List[str]] = None
        self.is_full_containers: bool = True
