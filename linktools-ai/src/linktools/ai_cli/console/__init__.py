#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Console (non-TUI) rendering + entry points for ``lt ai``.

Each ``console/*.py`` module exposes one function the thin ``commands/ai``
shells call. They talk to the backend only through
:class:`linktools.ai_cli.client.RuntimeClient` and render to stdout/_logger --
Textual is never imported here."""
