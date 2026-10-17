#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical mapping fields do not reorder authorization sequences."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import yaml

from _harness import configure_builtin_inputs
from linktools.cntr.ext import Authelia


def test_acl_mapping_order_is_stable_but_rule_order_is_preserved(fresh_manager, monkeypatch):
    manager = fresh_manager
    configure_builtin_inputs(manager)
    container = manager.containers["authelia"]
    producer = manager.containers["portainer"]
    fields = [("subject", ["group:z", "group:a"]), ("policy", "one_factor"),
              ("resources", ["^/z", "^/a"])]
    entries = []
    monkeypatch.setattr(manager, "iter_integrations", lambda consumer: iter(entries))
    # Resolve the normal nginx snapshot before substituting consumer inputs.
    manager.containers["nginx"].sites

    def render(reverse_keys=False, reverse_rules=False):
        first = Authelia.access(dict(reversed(fields) if reverse_keys else fields))
        second = Authelia.access(policy="deny", domain="other.example.test")
        entries[:] = [(producer, value) for value in ([second, first] if reverse_rules else [first, second])]
        for name in ("acl_rules", "acl_config", "config_path"):
            container.__dict__.pop(name, None)
        return container.acl_config, container.config_path

    initial, path = render()
    assert render(reverse_keys=True) == (initial, path)
    parsed = yaml.safe_load(initial)["access_control"]["rules"]
    assert parsed[0]["subject"] == ["group:z", "group:a"]
    assert parsed[0]["resources"] == ["^/z", "^/a"]
    changed, changed_path = render(reverse_rules=True)
    assert yaml.safe_load(changed)["access_control"]["rules"][:2] == parsed[:2][::-1]
    assert changed_path != path


def test_generated_config_is_stable_across_process_hash_seeds(tmp_path):
    source = textwrap.dedent('''
        import json
        from pathlib import Path
        import sys
        from _harness import make_manager, configure_builtin_inputs
        from linktools.cntr.ext import Authelia
        root = Path(sys.argv[1])
        manager = make_manager(root / "data", root / "temp")
        configure_builtin_inputs(manager)
        manager.config.set("DOCKER_APP_PATH", "/test/apps")
        manager.config.set("LLDAP_ADMIN_PASSWORD", "test-only")
        producer = manager.containers["portainer"]
        fields = {"subject": ["group:z", "group:a"], "policy": "one_factor"}
        rule = Authelia.access({key: fields[key] for key in set(fields)})
        callback = Authelia.oidc(("https://z.test/callback", "https://a.test/callback"))
        original = manager.iter_integrations
        manager.iter_integrations = lambda name: iter(((producer, rule), (producer, callback))) if name == "authelia" else original(name)
        result = {name: manager.containers[name].config_path.name for name in ("authelia", "nginx", "flare")}
        print(json.dumps(result, sort_keys=True))
    ''')
    root = Path(__file__).resolve().parents[2]
    paths = [str(root / "linktools/src"), str(root / "linktools-cntr/src"), str(root / "tests/cntr")]
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), PYTHONDONTWRITEBYTECODE="1")
    results = []
    for seed in ("0", "1", "42"):
        environment["PYTHONHASHSEED"] = seed
        completed = subprocess.run([sys.executable, "-c", source, str(tmp_path / seed)],
                                   env=environment, capture_output=True, text=True, check=True)
        results.append(json.loads(completed.stdout))
    assert results[1:] == results[:1] * 2


def test_resolved_oidc_allowlist_order_is_stable(fresh_manager, monkeypatch):
    from linktools.core import ConfigRef

    manager = fresh_manager
    configure_builtin_inputs(manager)
    container = manager.containers["authelia"]
    producers = [manager.containers["portainer"], manager.containers["nginx"]]
    urls = ("https://z.test/callback", "https://A.test/callback", "https://a.test/callback",
            "https://a.test/callback/", "https://a.test/callback?q=1", "https://a.test/Callback")
    entries = []
    manager.containers["nginx"].sites
    monkeypatch.setattr(manager, "iter_integrations", lambda consumer: iter(entries))

    def render(reverse):
        declarations = []
        for producer, values in zip(producers, (urls[:3], urls[3:])):
            resolved = tuple(reversed(values)) if reverse else values
            key = "TEST_OIDC_CALLBACKS_" + producer.name.upper()
            producer.config.set(key, resolved + ("/incomplete", resolved[0]))
            declaration = Authelia.oidc(ConfigRef(key))
            assert declaration.resolve_redirect_uris(producer.config) == resolved
            declarations.append((producer, declaration))
        entries[:] = reversed(declarations) if reverse else declarations
        for name in ("oidc_redirects", "oidc_client", "config_path"):
            container.__dict__.pop(name, None)
        text = container.render_template(container.get_source_path("templates", "configuration.oidc.yml"))
        return container.oidc_redirects, text, container.config_path

    first = render(False)
    assert render(True) == first
    assert first[0] == tuple(sorted((container.public_url,) + urls))
    # The native secret expression is evaluated later by Authelia.
    rendered = "\n".join(line for line in first[1].splitlines() if "- key: {{ secret" not in line)
    assert yaml.safe_load(rendered)["identity_providers"]["oidc"]["clients"][0]["redirect_uris"] == list(first[0])
