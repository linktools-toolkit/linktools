#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concrete generation owners for the builtin integration consumers."""
from ._generation.nginx import NginxGeneration
from ._generation.lldap import LldapGeneration
from ._generation.authelia import AutheliaGeneration
from ._generation.flare import FlareGeneration

__all__ = ("NginxGeneration", "LldapGeneration", "AutheliaGeneration", "FlareGeneration")
