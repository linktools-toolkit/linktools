# -*- coding: utf-8 -*-
"""authelia's oidc_clients RedirectURLs must be persistable.

Regression: oidc_clients built client["RedirectURLs"] as a Python `set`
(needed so nginx's write_conf, called by every OIDC-integrated homelab
container, can `.add()` a redirect URI onto it in place) and then persisted
the raw result via `settings.set(...)`. CacheStore's codec is JSON-only, and
sets aren't JSON-serializable, so this raised CacheCodecError the moment
oidc_clients (or _update_files, which re-persists it) ran.
"""


def test_oidc_clients_redirect_urls_is_a_set_in_memory(fresh_manager):
    authelia = fresh_manager.containers["authelia"]
    redirect_urls = authelia.oidc_clients[0]["RedirectURLs"]
    assert isinstance(redirect_urls, set)
    redirect_urls.add("https://example.com/callback")  # must not raise (AttributeError on a list)


def test_oidc_clients_persists_without_crashing(fresh_manager):
    authelia = fresh_manager.containers["authelia"]
    authelia.oidc_clients[0]["RedirectURLs"].add("https://example.com/callback")

    with authelia.settings.transaction() as settings:
        # Must not raise CacheCodecError (RedirectURLs is a set in memory).
        settings.set(f"{authelia._key_prefix}_oidc_clients",
                     authelia._oidc_clients_json_safe(authelia.oidc_clients))

    persisted = authelia.settings.get(f"{authelia._key_prefix}_oidc_clients")
    assert isinstance(persisted[0]["RedirectURLs"], list)
    assert "https://example.com/callback" in persisted[0]["RedirectURLs"]


def test_oidc_clients_reloaded_from_store_is_a_set_again(fresh_manager):
    authelia = fresh_manager.containers["authelia"]
    authelia.oidc_clients[0]["RedirectURLs"].add("https://example.com/callback")
    with authelia.settings.transaction() as settings:
        settings.set(f"{authelia._key_prefix}_oidc_clients",
                     authelia._oidc_clients_json_safe(authelia.oidc_clients))

    # A fresh authelia container instance (simulating the next CLI invocation)
    # must restore RedirectURLs to a set, not leave it as the persisted list.
    from linktools.cntr.registry.loader import ContainerLoader
    from linktools.cntr.repo.context import RepositoryConfigContext
    builtin_context = RepositoryConfigContext(
        root_path=None, file_config=None, url=None, builtin=True,
    )
    fresh_containers = list(ContainerLoader(fresh_manager)._load_one(authelia.root_path, builtin_context))
    reloaded = fresh_containers[0]
    assert isinstance(reloaded.oidc_clients[0]["RedirectURLs"], set)
    assert "https://example.com/callback" in reloaded.oidc_clients[0]["RedirectURLs"]
