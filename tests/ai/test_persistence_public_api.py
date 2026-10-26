#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common persistence configuration stays separate from advanced contracts."""

import os
from pathlib import Path
import subprocess
import sys

from linktools.ai import persistence
from linktools.ai.persistence import PersistenceDomain


def test_persistence_root_contains_only_common_configuration() -> None:
    assert set(persistence.__all__) == {
        'Persistence', 'PersistenceStorage', 'PersistencePlan',
        'PersistenceStatus', 'PersistenceDomain',
    }
    assert not hasattr(persistence, 'RuntimeDomain')
    assert not hasattr(persistence, 'RuntimeStateCommands')
    assert not hasattr(persistence, 'RuntimeObjectKeyFactory')


def test_persistence_domain_wire_values_do_not_depend_on_python_names() -> None:
    assert {domain.value for domain in PersistenceDomain} == {
        'conversation', 'execution', 'memory', 'artifact', 'task',
        'evaluation', 'recovery', 'trace', 'checkpoint',
    }


def test_advanced_persistence_imports_do_not_load_runtime() -> None:
    source = Path(__file__).resolve().parents[2] / 'linktools-ai' / 'src'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(source), env.get('PYTHONPATH', '')))
    subprocess.run(
        [sys.executable, '-c', '''
import importlib
import sys
for name in (
    'contracts', 'state', 'runs', 'objects', 'commands', 'capabilities',
    'snapshots', 'codec', 'captures', 'repositories', 'durability',
):
    module = importlib.import_module('linktools.ai.persistence.' + name)
    for symbol in module.__all__:
        assert hasattr(module, symbol), (name, symbol)
assert 'linktools.ai.runtime' not in sys.modules
from linktools.ai.persistence.commands import StateCommands
from linktools.ai.persistence.objects import ObjectKeyFactory
'''],
        env=env, check=True, capture_output=True, text=True,
    )
