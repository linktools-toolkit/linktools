#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``linktools.ai_cli`` -- the business layer behind the ``lt ai`` CLI/TUI.

Dependency direction::

    linktools.commands.ai  ->  linktools.ai_cli  ->  linktools.ai

The ``commands/ai`` package holds only thin command shells (name, help, arg
declarations, a single call into this package, an exit code). Everything that
actually loads a project, builds a Runtime, streams events, resolves approvals
or renders output lives here. Both the console commands and the (future) Textual
TUI talk to the backend exclusively through :class:`linktools.ai_cli.client.RuntimeClient`.
"""
