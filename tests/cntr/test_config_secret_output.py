#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`ct-cntr config set/get/list` must never default-print a
``ConfigField(secret=True)`` value, and must not rely on the field's NAME
happening to match the logger's own key-pattern heuristic to do so.

Regression: `on_command_set`/`on_command_get` printed/logged the resolved
value directly with no field-aware redaction at all -- only the logger's
automatic pattern-based masking (triggered by names containing e.g.
PASSWORD/SECRET/TOKEN) stood between a secret field and the terminal. A
field explicitly marked ``secret=True`` but with an innocuous name (e.g.
CREDENTIAL) was never masked by `set`/`get` at all.
"""
import linktools.cntr.commands._shared as cntr_shared
from linktools.cntr.commands.config import ConfigCommand
from linktools.core import ConfigField

_SECRET_VALUE = "very-sensitive-value-78231"


def _define_secret_field(manager):
    manager.env_config.define(ConfigField(name="CREDENTIAL", secret=True))


def test_set_redacts_secret_in_logger_output(monkeypatch, fresh_manager, caplog):
    import logging
    _define_secret_field(fresh_manager)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_set(configs={"CREDENTIAL": _SECRET_VALUE})

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL: ***" in messages


def test_get_default_redacts_secret(monkeypatch, fresh_manager, caplog):
    import logging
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_get(keys=["CREDENTIAL"], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL=***" in messages


def test_get_show_secret_reveals_value(monkeypatch, fresh_manager, capsys):
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    ConfigCommand().on_command_get(keys=["CREDENTIAL"], show_secret=True)
    out = capsys.readouterr().out
    assert f"CREDENTIAL={_SECRET_VALUE}" in out


def test_list_default_redacts_secret(monkeypatch, fresh_manager, caplog):
    import logging
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_list(names=[], show_secret=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
    assert "CREDENTIAL=***" in messages


def test_list_show_secret_reveals_value(monkeypatch, fresh_manager, capsys):
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    ConfigCommand().on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out
    assert f"CREDENTIAL={_SECRET_VALUE}" in out


def test_explain_json_does_not_leak_secret(monkeypatch, fresh_manager, capsys):
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    ConfigCommand().on_command_explain(key="CREDENTIAL", as_json=True)
    out = capsys.readouterr().out
    assert _SECRET_VALUE not in out


def test_explain_text_does_not_leak_secret(monkeypatch, fresh_manager, caplog):
    import logging
    _define_secret_field(fresh_manager)
    fresh_manager.env_config.persist("CREDENTIAL", _SECRET_VALUE)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)

    with caplog.at_level(logging.INFO):
        ConfigCommand().on_command_explain(key="CREDENTIAL", as_json=False)

    messages = "\n".join(caplog.messages)
    assert _SECRET_VALUE not in messages
