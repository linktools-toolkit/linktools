#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptProvider with choices should route through choose(), not prompt().

A choices list is a fixed menu (e.g. cntr's DOCKER_TYPE: docker/docker-rootless):
presenting it as a numbered pick via choose() is faster and less error-prone than
free-text entry via prompt(), which previously ignored the terminal's ability to
select and just echoed the choices as a hint string.
"""
from linktools.core._config import ConfigField, ConfigResolver, ConfigSchema, PromptProvider


def _resolve(monkeypatch, field, *, choose_return=None, prompt_return=None):
    calls = []

    def fake_choose(prompt, choices, **kwargs):
        calls.append(("choose", prompt, list(choices), kwargs))
        return choose_return if choose_return is not None else choices[0]

    def fake_prompt(prompt, **kwargs):
        calls.append(("prompt", prompt, kwargs))
        return prompt_return

    monkeypatch.setattr("linktools.rich.choose", fake_choose)
    monkeypatch.setattr("linktools.rich.prompt", fake_prompt)

    schema = ConfigSchema()
    schema.define(field)
    value = ConfigResolver(schema, sources=[]).resolve(field.name).value
    return value, calls


def test_choices_route_through_choose(monkeypatch):
    field = ConfigField(name="DOCKER_TYPE",
                        provider=PromptProvider(choices=["docker", "docker-rootless"]))
    value, calls = _resolve(monkeypatch, field, choose_return="docker-rootless")
    assert value == "docker-rootless"
    assert calls == [("choose", "DOCKER_TYPE", ["docker", "docker-rootless"], {"default": field.provider.default})]


def test_no_choices_routes_through_prompt(monkeypatch):
    field = ConfigField(name="HOST", provider=PromptProvider("HOST"))
    value, calls = _resolve(monkeypatch, field, prompt_return="1.2.3.4")
    assert value == "1.2.3.4"
    assert calls[0][0] == "prompt"


def test_choose_receives_default(monkeypatch):
    field = ConfigField(name="DOCKER_TYPE",
                        provider=PromptProvider(default="docker",
                                                choices=["docker", "docker-rootless"]))
    _, calls = _resolve(monkeypatch, field)
    assert calls[0][3]["default"] == "docker"


def test_field_default_is_used_when_prompt_has_no_default(monkeypatch):
    field = ConfigField(name="APP_PATH", default="/srv/app", provider=PromptProvider())
    _, calls = _resolve(monkeypatch, field, prompt_return="/srv/custom")
    assert calls[0][2]["default"] == "/srv/app"


# --------------------------------------------------------------------------- #
# The field's cast must be forwarded as prompt()'s type= hint. Without this,
# an int/bool field always got a bare string prompt: a real user's answer only
# worked by coincidence (str -> int/bool casts cleanly for valid input), and a
# type-aware non-interactive stand-in (e.g. a test double) had no way to
# return a plausible value or correctly detect "no default, cannot proceed"
# for the field it was actually being asked about.
# --------------------------------------------------------------------------- #

def test_int_cast_forwarded_as_prompt_type(monkeypatch):
    field = ConfigField(name="PORT", cast=int, provider=PromptProvider())
    _, calls = _resolve(monkeypatch, field, prompt_return=8080)
    assert calls[0][0] == "prompt"
    assert calls[0][2]["type"] is int


def test_string_literal_cast_not_forwarded_as_prompt_type(monkeypatch):
    # "path" is a ConfigField.cast literal, not something rich.prompt understands
    # as a type (it only knows str/int/float/bool) -- must fall back to str.
    field = ConfigField(name="APP_PATH", cast="path", provider=PromptProvider())
    _, calls = _resolve(monkeypatch, field, prompt_return="/tmp/x")
    assert calls[0][2]["type"] is str


def test_no_cast_defaults_to_str_prompt_type(monkeypatch):
    field = ConfigField(name="HOST", provider=PromptProvider())
    _, calls = _resolve(monkeypatch, field, prompt_return="1.2.3.4")
    assert calls[0][2]["type"] is str


# --------------------------------------------------------------------------- #
# allow_empty must be forwarded to rich.prompt() (dropped during the v2
# rewrite; restored for cntr's per-DNS-API env var prompts).
# --------------------------------------------------------------------------- #

def test_allow_empty_forwarded_to_prompt():
    field = ConfigField(name="TOKEN", provider=PromptProvider(allow_empty=True))
    schema = ConfigSchema()
    schema.define(field)
    import linktools.rich as rich
    captured = {}
    def spy(message, **kwargs):
        captured.update(kwargs)
        return ""
    real_prompt = rich.prompt
    rich.prompt = spy
    try:
        value = ConfigResolver(schema, sources=[]).resolve("TOKEN").value
    finally:
        rich.prompt = real_prompt
    assert captured["allow_empty"] is True
    assert value == ""


def test_allow_empty_defaults_to_false():
    field = ConfigField(name="TOKEN", provider=PromptProvider())
    schema = ConfigSchema()
    schema.define(field)
    import linktools.rich as rich
    captured = {}
    def spy(message, **kwargs):
        captured.update(kwargs)
        return "x"
    real_prompt = rich.prompt
    rich.prompt = spy
    try:
        ConfigResolver(schema, sources=[]).resolve("TOKEN")
    finally:
        rich.prompt = real_prompt
    assert captured["allow_empty"] is False
