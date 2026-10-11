#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Errors shared by orchestration and integration declarations."""

from linktools.errors import Error


class ContainerError(Error):
    pass


class ContainerTemplateError(ContainerError):
    pass


class NoContainerInstalledError(ContainerError):
    """No installed containers exist for this execution."""
    pass
