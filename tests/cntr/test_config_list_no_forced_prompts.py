# -*- coding: utf-8 -*-
"""`ct-cntr config list` (no container names) must not force-resolve every
schema-declared manager field -- only ones actually configured.

Regression: on_command_list's "add everything else" fallback used to be
`manager.env_config.keys()`, which (per Config.keys()) includes every
schema-declared field name whether or not it has ever been set -- unlike the
pre-refactor equivalent, `manager.env_config.cache.keys()`, which only ever
returned already-persisted keys. So a manager-level field nothing has
configured yet (e.g. DOCKER_DOWNLOAD_PATH -- a cached=True PromptProvider
field almost nothing actually reads) got force-resolved by `config list`
merely because it's *possible* to configure, prompting for it in a real
terminal even though the user never asked to set it.
"""
import linktools.cntr.__main__ as cntr_main
import linktools.cntr.commands._shared as cntr_shared


def test_persisted_keys_excludes_unconfigured_manager_fields(fresh_manager):
    assert "DOCKER_DOWNLOAD_PATH" not in fresh_manager.env_config.persisted_keys()
    # Sanity: it IS a real, resolvable schema field (just not yet configured).
    assert "DOCKER_DOWNLOAD_PATH" in fresh_manager.env_config.keys()


def test_persisted_keys_includes_explicitly_set_fields(fresh_manager):
    fresh_manager.env_config.persist("DOCKER_DOWNLOAD_PATH", "/srv/downloads")
    assert "DOCKER_DOWNLOAD_PATH" in fresh_manager.env_config.persisted_keys()


def test_config_list_does_not_prompt_for_unconfigured_manager_fields(monkeypatch, fresh_manager):
    import linktools.rich as rich
    real_prompt = rich.prompt

    def fail_if_docker_download_path(message, *a, **kw):
        if message == "DOCKER_DOWNLOAD_PATH":
            raise AssertionError("config list must not prompt for an unconfigured, never-set manager field")
        return real_prompt(message, *a, **kw)

    monkeypatch.setattr(rich, "prompt", fail_if_docker_download_path)
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = cntr_main.ConfigCommand()

    # Must not raise.
    config_command.on_command_list(names=[])


def test_config_list_still_shows_persisted_manager_fields(monkeypatch, fresh_manager, capsys):
    fresh_manager.env_config.persist("DOCKER_DOWNLOAD_PATH", "/srv/downloads")
    monkeypatch.setattr(cntr_shared, "manager", fresh_manager)
    config_command = cntr_main.ConfigCommand()

    config_command.on_command_list(names=[], show_secret=True)
    out = capsys.readouterr().out
    assert "DOCKER_DOWNLOAD_PATH=/srv/downloads" in out
