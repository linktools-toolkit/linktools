# -*- coding: utf-8 -*-
"""An already-persisted value must outrank a field's provider, even one
without cached=True.

Regression: HOST=ConfigField.chain(PromptProvider(), LazyProvider(get_lan_ip))
has no cached=True -- so with only the Environment/RuntimeOverride override
check (an earlier, narrower fix), resolving HOST never even looked at
"container.HOST" in settings.json once it had been migrated/set there;
PromptProvider's own (non-cached) resolution always ran and always prompted.

The old Config implementation's `_map` was `ChainMap(env_vars,
persistent_cache, field_descriptors, global_config)`: the persisted cache was
checked, for EVERY field, before its Prompt/Lazy/Alias descriptor ever ran --
regardless of that descriptor's own `cached=` flag (which only controlled
whether a freshly-computed answer got saved back, not whether an existing one
was read first). ConfigResolver._first_present_before_provider now includes
PersistentSource for exactly this reason.
"""
from linktools.core import ConfigField, LazyProvider, PromptProvider, ChainProvider


def test_host_does_not_reprompt_once_persisted(monkeypatch, fresh_manager):
    import linktools.rich as rich

    def fail(*_a, **_k):
        raise AssertionError("HOST must not be re-prompted once already persisted")

    monkeypatch.setattr(rich, "prompt", fail)

    fresh_manager.env_config.persist("HOST", "203.0.113.5")
    assert fresh_manager.env_config.get("HOST") == "203.0.113.5"


def test_host_still_resolves_via_provider_when_unset(fresh_manager):
    # Sanity: without a persisted value, HOST still falls through to its
    # provider chain (PromptProvider -> LazyProvider(get_lan_ip) in the
    # deterministic test harness) rather than raising.
    value = fresh_manager.env_config.get("HOST")
    assert value  # some non-empty IP-like string from the fallback chain


def test_cached_provider_still_reuses_after_first_compute(tmp_path):
    from linktools.core._config import (
        Config, ConfigSchema, EnvironmentSource, RuntimeOverrideSource,
        PersistentSource, DefaultSource, ConfigStore,
    )
    from linktools.core._locks import LockManager

    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema(allow_unknown=True)
    config = Config(None, schema, sources=[
        EnvironmentSource(""), RuntimeOverrideSource(),
        PersistentSource(store, "test"), DefaultSource(schema),
    ])
    calls = []
    config.update_defaults(
        SECRET=ConfigField(provider=LazyProvider(lambda r: calls.append(1) or "generated", cached=True)),
    )
    assert config.get("SECRET") == "generated"
    config._resolver.clear_memo()
    # Second resolution (fresh resolver state) must reuse the persisted value
    # via the new override check -- never re-invoking the LazyProvider.
    assert config.get("SECRET") == "generated"
    assert calls == [1]
