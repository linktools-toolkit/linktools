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
