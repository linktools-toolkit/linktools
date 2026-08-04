#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Console rendering + entry points for ``lt ai``.

Each ``console/*.py`` module exposes one function the thin ``commands/ai``
shells call. They talk to the backend only through
:class:`linktools.ai.cli.client.RuntimeClient` and render to stdout/_logger."""
