#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The single ContainerManager instance backing every ct-cntr command, plus
completion helpers shared by the command modules below."""
from linktools.core import environ

from ..manager import ContainerManager

manager = ContainerManager(environ)


def iter_container_names():
    return [container.name for container in manager.containers.values()]


def iter_installed_container_names():
    return [container.name for container in manager.get_installed_containers()]
