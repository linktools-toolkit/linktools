#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native access rules resolve independently against their producer's sites."""
from types import SimpleNamespace

import pytest
import yaml

from linktools.cntr import ContainerError, ContainerManager, Integration
from linktools.cntr.ext import Authelia, Nginx, endpoint
from linktools.core import Config, ConfigSchema
from linktools.runtime import lazy_load


def fail() -> None:
    raise AssertionError("unrelated lazy declaration value evaluated")


def _consumer(declarations: dict, *, nginx: bool = True, **overrides: object) -> "tuple[object, list[str]]":
    from _harness import builtin_container_type

    reads = []
    values = {"NGINX_AUTH_ENABLE": True, "NGINX_HTTPS_ENABLE": True,
              "AUTHELIA_MIN_AUTH_LEVEL": 2, "NGINX_ROOT_DOMAIN": "example.test"}
    values.update(overrides)

    def config(key: str, **kwargs: object) -> object:
        reads.append(key)
        value = values[key]
        return value() if callable(value) else value

    manager = object.__new__(ContainerManager)
    manager.env_config = Config(None, ConfigSchema(), [])
    manager.env_config.get = config
    containers = {name: SimpleNamespace(name=name, integrations=entries, manager=manager, get_config=config,
                                        env_config=manager.env_config)
                  for name, entries in declarations.items()}
    for name, builtin in (("nginx", "100-nginx"), ("authelia", "102-authelia")):
        container = object.__new__(builtin_container_type(builtin))
        container._name = name
        container.manager = manager
        container.__dict__["integrations"] = ()
        container.get_config = config
        containers[name] = container
    manager.containers = containers
    installed = [container for name, container in containers.items() if nginx or name != "nginx"]
    manager.installed_state = SimpleNamespace(get=lambda resolve=False: installed)
    return containers["authelia"], reads


def test_access_mapping_is_lazy_cached_and_read_only_without_changing_native_fields() -> None:
    calls = []
    native = {"policy": "two_factor", "networks": ["10.0.0.0/8"],
              "subject": ["group:staff"], "resources": [r"^/admin"], "methods": ["GET"],
              "query": [{"key": "mode", "operator": "equal", "value": "admin"}]}
    declaration = Authelia.access(lazy_load(lambda: calls.append("rule") or native))
    assert isinstance(declaration, Integration) and declaration.consumer == "authelia"
    assert declaration.site is None
    assert calls == []
    assert dict(declaration.rule) == native
    assert calls == ["rule"]
    assert declaration.rule is declaration.rule
    assert "domain" not in native
    with pytest.raises(TypeError):
        declaration.rule["policy"] = "bypass"
    assert declaration.redirect_uris == ()


def test_access_keyword_fields_preserve_native_values_and_secondary_site() -> None:
    resources = lazy_load(fail)
    site = Nginx.site(lazy_load(fail))
    declaration = Authelia.access(site=site, policy=None, domain_regex=r"^app\.test$",
                                   resources=resources, methods=["POST"])
    assert declaration.site is site
    assert declaration.rule["policy"] is None
    assert declaration.rule["domain_regex"] == r"^app\.test$"
    assert declaration.rule["resources"] is resources
    assert declaration.rule["methods"] == ["POST"]


def test_access_rule_mapping_and_fields_are_mutually_exclusive_without_resolution() -> None:
    with pytest.raises(TypeError, match="rule mapping or keyword fields"):
        Authelia.access(lazy_load(fail), policy="one_factor")


@pytest.mark.parametrize("reference", ["web", "", 0, [], False, {}])
def test_access_rejects_non_declaration_references_only_when_consumed(reference: object) -> None:
    declaration = Authelia.access(lazy_load(fail), site=reference)
    authelia, reads = _consumer({"app": [Nginx.site(lazy_load(fail)), declaration]})
    assert reads == []
    with pytest.raises(ContainerError, match="Expected an original nginx site or endpoint declaration"):
        authelia.acl_rules


@pytest.mark.parametrize("rule", [None, "policy", ["one_factor"], 1])
def test_access_rejects_non_mapping_rules_only_when_consumed(rule: object) -> None:
    declaration = Authelia.access(rule)
    with pytest.raises(ContainerError, match="rule must be a mapping"):
        declaration.rule


@pytest.mark.parametrize("level,policy", [(1, "one_factor"), (2, "two_factor"), (3, "two_factor")])
def test_acl_consumer_fills_missing_domain_and_policy_without_mutating_input(level: int, policy: str) -> None:
    native = {"subject": ["group:staff"], "networks": ["10.0.0.0/8"]}
    access = Authelia.access(native)
    authelia, reads = _consumer({"app": [Nginx.site("app.test"), access]},
                                AUTHELIA_MIN_AUTH_LEVEL=level)
    assert reads == []
    assert authelia.acl_rules == [dict(native, domain="app.test", policy=policy)]
    assert dict(access.rule) == native
    assert "domain" not in native and "policy" not in native


@pytest.mark.parametrize("native", [
    {"domain": "explicit.test", "policy": "bypass"},
    {"domain": ["first.test", "second.test"], "policy": "deny"},
    {"domain_regex": r"^app\.test$", "policy": "one_factor"},
    {"domain": None, "policy": None},
    {"domain": "", "policy": ""},
])
def test_acl_preserves_explicit_native_domain_and_policy_without_default_reads(native: dict) -> None:
    authelia, reads = _consumer({"app": [Nginx.site("~^app"), Authelia.access(native)]},
                                AUTHELIA_MIN_AUTH_LEVEL=fail)
    assert authelia.acl_rules == [native]
    assert "AUTHELIA_MIN_AUTH_LEVEL" not in reads


def test_acl_default_policy_keeps_domain_regex_instead_of_adding_domain() -> None:
    native = {"domain_regex": r"^app\.test$"}
    authelia, _ = _consumer({"app": [Nginx.site("~^app"), Authelia.access(native)]})
    assert authelia.acl_rules == [{"domain_regex": r"^app\.test$", "policy": "two_factor"}]
    assert "domain" not in native


def test_acl_references_are_producer_local_and_follow_declaration_order() -> None:
    first_site, admin_site, second_site = [Nginx.site(domain) for domain in
                                           ("first.test", "admin.test", "second.test")]
    authelia, _ = _consumer({
        "first": [first_site, admin_site,
                  (Authelia.access(site=admin_site, subject=["group:admins"]),
                   [Authelia.access(site=first_site, policy="one_factor"),
                    Authelia.access(site=first_site, resources=[r"^/api/"])])],
        "second": [second_site, Authelia.access(policy="deny")],
    })
    assert authelia.acl_rules == [
        {"subject": ["group:admins"], "domain": "admin.test", "policy": "two_factor"},
        {"domain": "first.test", "policy": "one_factor"},
        {"resources": [r"^/api/"], "domain": "first.test", "policy": "two_factor"},
        {"domain": "second.test", "policy": "deny"},
    ]


@pytest.mark.parametrize("site,nginx,config", [
    (Nginx.site("", auth=lazy_load(fail)), True, {}),
    (Nginx.site(lazy_load(fail), auth=lazy_load(fail)), False, {}),
    (Nginx.site("app.test", auth=False), True, {"NGINX_AUTH_ENABLE": fail}),
    (Nginx.site("app.test"), True, {"NGINX_AUTH_ENABLE": False}),
])
def test_disabled_sites_or_auth_skip_lazy_access_rules(site: Nginx, nginx: bool, config: dict) -> None:
    authelia, reads = _consumer({"app": [site, Authelia.access(lazy_load(fail))]},
                                nginx=nginx, AUTHELIA_MIN_AUTH_LEVEL=fail, **config)
    assert authelia.acl_rules == []
    assert "AUTHELIA_MIN_AUTH_LEVEL" not in reads


def test_acl_ignores_oidc_callbacks_and_sites_without_explicit_access() -> None:
    authelia, reads = _consumer({"app": [
        Nginx.site(lazy_load(fail)), Authelia.oidc(lazy_load(fail)),
    ]})
    assert authelia.acl_rules == []
    assert reads == []


@pytest.mark.parametrize("server_name", ["_", "*.test", "~^app\\.test$", "a.test b.test"])
def test_nonliteral_site_requires_explicit_native_domain_or_regex(server_name: str) -> None:
    authelia, _ = _consumer({"app": [Nginx.site(server_name), Authelia.access()]})
    with pytest.raises(ContainerError, match="app requires domain or domain_regex"):
        authelia.acl_rules


def test_unknown_site_is_not_resolved_from_another_producer() -> None:
    authelia, _ = _consumer({"first": [Authelia.access(lazy_load(fail))],
                             "second": [Nginx.site("second.test")]})
    with pytest.raises(ContainerError, match="Expected one declared nginx site in first"):
        authelia.acl_rules


def test_acl_consumer_rejects_unknown_authelia_declaration_types() -> None:
    declaration = Integration()
    declaration.consumer = "authelia"
    authelia, _ = _consumer({"app": [declaration]})
    with pytest.raises(ContainerError, match="Invalid Authelia declaration in app"):
        authelia.acl_rules


def test_acl_configuration_keeps_default_deny_and_admin_fallback_last() -> None:
    authelia, _ = _consumer({"app": [Nginx.site("app.test"),
                                    Authelia.access(policy="bypass", resources=[r"^/public"]),
                                    Authelia.access(policy="one_factor")]})
    config = yaml.safe_load(authelia.acl_config)["access_control"]
    assert config["default_policy"] == "deny"
    assert config["rules"] == [
        {"domain": "app.test", "policy": "bypass", "resources": [r"^/public"]},
        {"domain": "app.test", "policy": "one_factor"},
        {"domain": ["example.test", "*.example.test"],
         "subject": ["group:admin", "group:admins", "group:super-admin"], "policy": "two_factor"},
    ]
    assert len(authelia.acl_rules) == 2


def test_access_endpoint_reference_selects_its_site_and_rejects_foreign_producers() -> None:
    recipe = endpoint(object(), domain="app.test")
    access = Authelia.access(site=recipe, policy="one_factor")
    authelia, _ = _consumer({"app": [Nginx.site("other.test"), recipe, access]})
    assert authelia.acl_rules == [{"domain": "app.test", "policy": "one_factor"}]
    authelia, _ = _consumer({"app": [recipe], "other": [access]})
    with pytest.raises(ContainerError, match="Expected one declared nginx site in other"):
        authelia.acl_rules


def test_endpoint_binds_reused_acl_to_actual_site_without_mutating_rule() -> None:
    access = Authelia.access(policy="one_factor")
    first = endpoint(object(), domain="first.test", extensions=[access])
    second = endpoint(object(), domain="second.test", extensions=[access])
    authelia, _ = _consumer({"app": [first, second]})
    assert authelia.acl_rules == [
        {"domain": "first.test", "policy": "one_factor"},
        {"domain": "second.test", "policy": "one_factor"},
    ]
    assert access.site is None and dict(access.rule) == {"policy": "one_factor"}


def test_unbound_access_requires_one_declared_site_even_if_others_are_disabled() -> None:
    authelia, reads = _consumer({"app": [Nginx.site("app.test"), Nginx.site(""),
                                         Authelia.access(lazy_load(fail))]})
    assert reads == []
    with pytest.raises(ContainerError, match="Expected one declared nginx site in app"):
        authelia.acl_rules
    assert reads == []


def test_equivalent_access_site_is_not_a_reference_to_the_original() -> None:
    authelia, _ = _consumer({"app": [Nginx.site("app.test"),
                                     Authelia.access(lazy_load(fail), site=Nginx.site("app.test"))]})
    with pytest.raises(ContainerError, match="Expected one declared nginx site in app"):
        authelia.acl_rules
