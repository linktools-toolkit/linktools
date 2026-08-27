#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapters for external provider, identity, and transport boundaries."""

from ._identity import StaticPrincipalProvider
from ._nats import NatsPublisher
from ._provider import ProviderClient

__all__ = [
    "NatsPublisher",
    "ProviderClient",
    "StaticPrincipalProvider",
]
