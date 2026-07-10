# -*- coding: utf-8 -*-
"""Environment/RuntimeOverride must outrank a field's provider.

Regression: ConfigResolver._resolve_inner tried a field's provider before ever
consulting _first_present (env/runtime-override/persistent/default) -- only
falling back to sources if every ChainProvider sub-provider raised. Since
nearly every meaningfully-configured field has a provider, this meant an
env var of the field's own name was silently ignored the moment the provider
could produce a value, and permanently ignored for any cached=True provider
once it persisted an answer (e.g. `NGINX_ROOT_DOMAIN=x ct-cntr up` would be a
silent no-op after the field had been resolved/cached once). This contradicts
the documented Environment > RuntimeOverride > Persistent > Default precedence
(see Environ._create_config / wrap_config's source order).
"""
from linktools.core import (
    AliasProvider, Config, ConfigField, ConfigSchema, ChainProvider,
    DefaultSource, EnvironmentSource, LazyProvider, PersistentSource,
    PromptProvider, RuntimeOverrideSource,
)
from linktools.core._config import ConfigStore
from linktools.core._locks import LockManager


def _make_config(tmp_path, env_prefix=""):
    store = ConfigStore(tmp_path / "settings.json", lock_manager=LockManager(tmp_path / "locks"))
    schema = ConfigSchema(allow_unknown=True)
    return Config(None, schema, sources=[
        EnvironmentSource(env_prefix),
        RuntimeOverrideSource(),
        PersistentSource(store, "test"),
        DefaultSource(schema),
    ])


def test_env_overrides_cached_prompt_provider_already_persisted(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.update_defaults(
        NGINX_ROOT_DOMAIN=ConfigField(provider=PromptProvider(cached=True), default="_"),
    )
    # Simulate the field having already been prompted/cached in a PRIOR run
    # (this Config/resolver's own in-process memo is untouched -- a real
    # `NGINX_ROOT_DOMAIN=x ct-cntr up` sets the env var before the process,
    # and thus before this resolver, even exists).
    config.persist("NGINX_ROOT_DOMAIN", "test.local")

    monkeypatch.setenv("NGINX_ROOT_DOMAIN", "prod.example.com")
    assert config.get("NGINX_ROOT_DOMAIN") == "prod.example.com"


def test_env_overrides_noncached_lazy_provider(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    calls = []
    config.update_defaults(
        DOCKER_UID=ConfigField(provider=LazyProvider(lambda r: calls.append(1) or 1000)),
    )
    monkeypatch.setenv("DOCKER_UID", "2000")
    assert config.get("DOCKER_UID", type=int) == 2000
    assert calls == []  # provider must not even run once env wins


def test_env_overrides_chain_provider_via_fields_own_name(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    config.update_defaults(
        NGINX_ROOT_DOMAIN=ConfigField.chain(
            AliasProvider("ROOT_DOMAIN"), PromptProvider(cached=True), default="_",
        ),
    )
    monkeypatch.setenv("NGINX_ROOT_DOMAIN", "override.example.com")
    assert config.get("NGINX_ROOT_DOMAIN") == "override.example.com"


def test_alias_target_env_var_still_works_as_escape_hatch(tmp_path, monkeypatch):
    # The alias TARGET's own env var (not the field's name) must still resolve
    # through the alias -- this pre-existing behavior must not regress.
    config = _make_config(tmp_path)
    config.update_defaults(
        NGINX_ROOT_DOMAIN=ConfigField.chain(
            AliasProvider("ROOT_DOMAIN"), PromptProvider(cached=True), default="_",
        ),
    )
    monkeypatch.setenv("ROOT_DOMAIN", "alias.example.com")
    assert config.get("NGINX_ROOT_DOMAIN") == "alias.example.com"


def test_runtime_override_outranks_provider(tmp_path):
    config = _make_config(tmp_path)
    config.update_defaults(
        HOST=ConfigField(provider=PromptProvider(cached=True), default="_"),
    )
    config.persist("HOST", "cached-host")
    config.set("HOST", "runtime-host")
    assert config.get("HOST") == "runtime-host"


def test_no_override_still_falls_through_to_provider(tmp_path):
    # Sanity: without any env/runtime override, resolution is unaffected --
    # the provider still runs and its cached answer is still reused.
    config = _make_config(tmp_path)
    calls = []
    config.update_defaults(
        SECRET=ConfigField(provider=LazyProvider(lambda r: calls.append(1) or "generated", cached=True)),
    )
    assert config.get("SECRET") == "generated"
    config._resolver.clear_memo()
    assert config.get("SECRET") == "generated"
    assert calls == [1]  # computed once, reused from PersistentSource on the 2nd call
