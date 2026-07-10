# -*- coding: utf-8 -*-
"""`ct-cntr config list --show-secret` bypasses the logger's auto-redaction.

self.logger.info(...) goes through LoggingManager's global redaction filter,
which masks anything that looks like a secret/password/token by design (never
leak one into a log file/CI output by accident) -- but that also means
`ct-cntr config list` could never show a real secret value at all, even when
the user explicitly wants to see it. --show-secret opts out for this one
command by printing directly instead of going through the logger.
"""
import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared


def test_show_secret_prints_real_value_bypassing_redaction(monkeypatch, fresh_manager, capsys):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = cntr_main.ConfigCommand()

    fresh_manager.env_config.persist("DOCKER_APP_PATH", "/srv/app")
    config_command.on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out
    # A plain (non-secret) configured value -- confirms real values (not
    # "***") are printed when show_secret is set.
    assert "***" not in out
    assert "DOCKER_APP_PATH=/srv/app" in out


def test_default_still_redacts_via_logger(monkeypatch, fresh_manager, caplog):
    import logging
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = cntr_main.ConfigCommand()

    fresh_manager.env_config.persist("SOME_PASSWORD", "hunter2")
    with caplog.at_level(logging.INFO):
        config_command.on_command_list(names=[], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert "hunter2" not in messages
    assert "SOME_PASSWORD=***" in messages  # confirms the line was actually emitted, just masked


def test_show_secret_reveals_the_actual_password(monkeypatch, fresh_manager, capsys):
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = cntr_main.ConfigCommand()

    fresh_manager.env_config.persist("SOME_PASSWORD", "hunter2")
    config_command.on_command_list(names=[], show_secret=True)

    out = capsys.readouterr().out
    assert "SOME_PASSWORD=hunter2" in out
