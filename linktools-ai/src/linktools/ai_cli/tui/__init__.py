#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Textual TUI for ``lt ai``.

Textual is an *optional* dependency (``linktools-ai[tui]``); these modules import
it lazily so the rest of the package keeps working without it. :func:`run_tui`
is the single entry point the thin ``lt ai tui`` shell calls; it translates a
missing Textual install into an explicit, actionable error rather than a crash.
"""


def run_tui(
    *, project, remote, base_url=None, model=None, api_key=None, client=None
) -> int:
    try:
        from .app import run_tui as _run_tui
    except ImportError as exc:
        if exc.name and (exc.name == "textual" or exc.name.startswith("textual.")):
            from linktools.cli import CommandError

            raise CommandError(
                "the Textual TUI requires the 'tui' extra: "
                "pip install linktools-ai[tui]"
            ) from exc
        raise
    return _run_tui(
        project=project,
        remote=remote,
        base_url=base_url,
        model=model,
        api_key=api_key,
        client=client,
    )
