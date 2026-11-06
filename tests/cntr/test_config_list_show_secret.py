#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit config disclosure prints values without field display masking."""
from linktools.cntr.commands.config import ConfigCommand
import linktools.cntr.commands._shared as cntr_shared


def test_show_secret_prints_configured_value(monkeypatch, fresh_manager, capsys):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = ConfigCommand()

    fresh_manager.config.persist("DOCKER_APP_PATH", "/srv/app")
    config_command.on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out
    # A plain (non-secret) configured value -- confirms real values (not
    # "***") are printed when show_secret is set.
    assert "***" not in out
    assert "DOCKER_APP_PATH=/srv/app" in out


def test_show_secret_reveals_the_actual_password(monkeypatch, fresh_manager, capsys):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = ConfigCommand()

    fresh_manager.config.persist("SOME_PASSWORD", "hunter2")
    config_command.on_command_list(names=[], show_secret=True)

    out = capsys.readouterr().out
    assert "SOME_PASSWORD=hunter2" in out
