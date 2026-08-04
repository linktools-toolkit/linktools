#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``linktools.ai.cli`` -- the business layer behind the ``lt ai`` CLI.

Dependency direction::

    linktools.commands.ai  ->  linktools.ai.cli  ->  linktools.ai

The ``commands/ai`` package holds only thin command shells (name, help, arg
declarations, a single call into this package, an exit code). Everything that
actually loads a project, builds a Runtime, streams events, resolves approvals
or renders output lives here. Console commands talk to the backend exclusively
through :class:`linktools.ai.cli.client.RuntimeClient`.
"""
