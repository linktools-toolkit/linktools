# -*- coding: utf-8 -*-
"""A bare AliasProvider must fall back to the field's own default.

Regression: NGINX_WILDCARD_DOMAIN=ConfigField(default=False,
provider=AliasProvider("WILDCARD_DOMAIN")) crashed `ct-cntr config list` with
ConfigNotFoundError, because a bare (non-chain) AliasProvider has no fallback
when its target is unset -- unlike the pre-refactor legacy API's
`Config.Alias("WILDCARD_DOMAIN") | False`, which fell through to False.
ConfigField.chain(AliasProvider(...), default=...) restores that fallback via
ChainProvider's exception-swallowing (falls through to field.default when
every sub-provider raises).
"""


def test_nginx_wildcard_domain_falls_back_to_default(fresh_manager):
    # WILDCARD_DOMAIN is a free-standing knob (only ever set via env var by an
    # operator who wants every nginx-proxied domain wildcarded); it's not
    # itself a field anywhere, so resolving it here must fall back to False.
    assert fresh_manager.env_config.get("NGINX_WILDCARD_DOMAIN") is False


def test_authelia_ldap_base_dn_uses_alias_target(fresh_manager):
    # The alias target (LLDAP_BASE_DN) is always resolvable via its own
    # LazyProvider, so this doesn't hit the fallback -- confirms the chain
    # wrapper didn't break the normal (non-fallback) path either.
    assert fresh_manager.env_config.get("AUTHELIA_LDAP_BASE_DN") == "dc=_"
