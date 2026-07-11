#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.cli import BaseCommandGroup
from ..status import StatusCommands
from .inspect import ComposeInspectCommands
from .lifecycle import ComposeLifecycleCommands


class ComposeCommand(ComposeLifecycleCommands, ComposeInspectCommands, StatusCommands, BaseCommandGroup):
    """
    manage the Docker Compose project (up/restart/down/config/validate)
    """

    @property
    def name(self) -> str:
        return "compose"
