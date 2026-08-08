#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External adapters implementing lower-level ports."""

from ._logfire import LogfireSink
from ._nats import NatsPublisher
from ._identity import StaticPrincipalProvider
from ._provider import ProviderClient

__all__ = [
    "LogfireSink", "NatsPublisher", "ProviderClient", "StaticPrincipalProvider",
]
