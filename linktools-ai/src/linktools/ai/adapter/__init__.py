#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External adapters implementing lower-level ports."""

from .logfire import LogfireSink
from .nats import NatsPublisher
from .principal import StaticPrincipalProvider
from .provider import ProviderClient

__all__ = [
    "LogfireSink", "NatsPublisher", "ProviderClient", "StaticPrincipalProvider",
]
