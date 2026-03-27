#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, Dict, Any, List, Optional, Callable, Iterable

from .container import BaseContainer

class EventContext:

    def __init__(self):
        self.command: Optional[str] = None
        self.containers: List[BaseContainer] = None
        self.target_containers: List[str] = []
        self.is_full_containers = True
