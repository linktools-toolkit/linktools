#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ContainerLoader must instantiate the class a container.py actually
defines, not a concrete BaseContainer subclass it merely imports.

Regression: _load_one() scanned module.__dict__.values() and instantiated
the first concrete (non-abstract) BaseContainer subclass found. dict
iteration order follows insertion order, which follows source order --
so `from repo.common import CommonContainer` (a concrete shared base some
container.py imports and subclasses) would be found before the subclass
this file actually defines, and the wrong class would be loaded.
"""
import os
import textwrap

from linktools.cntr.registry.loader import ContainerLoader
from linktools.cntr.repo.context import RepositoryConfigContext


def _write_container_importing_a_shared_concrete_base(tmp_path):
    (tmp_path / "common.py").write_text(textwrap.dedent("""\
        from linktools.cntr.container import BaseContainer


        class CommonContainer(BaseContainer):
            pass
    """))

    (tmp_path / "container.py").write_text(textwrap.dedent("""\
        import os
        from linktools.runtime import import_module_file

        _common = import_module_file(
            "test_loader_class_priority_shared_common",
            os.path.join(os.path.dirname(__file__), "common.py"),
        )
        CommonContainer = _common.CommonContainer


        class AppContainer(CommonContainer):
            pass
    """))


def test_loader_picks_the_locally_defined_subclass_not_the_imported_base(fresh_manager, tmp_path):
    _write_container_importing_a_shared_concrete_base(tmp_path)

    builtin_context = RepositoryConfigContext(
        root_path=None, file_config=None, config=fresh_manager.env_config, url=None, builtin=True,
    )
    loaded = list(ContainerLoader(fresh_manager)._load_one(str(tmp_path), builtin_context))

    assert len(loaded) == 1
    assert loaded[0].__class__.__name__ == "AppContainer"
