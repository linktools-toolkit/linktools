#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contract: the core scenario cleanly without the optional
SQLAlchemy/aiosqlite dependencies, and accessing SqlAlchemyStorage yields a
clear install hint instead of a raw ModuleNotFoundError.

Run in a subprocess with a meta-path finder that blocks the optional packages,
so the real (installed) environment is not disturbed."""

import os
import sys
import textwrap

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_SUBPROCESS = textwrap.dedent(
    """
    import importlib.abc, sys
    _BLOCK = {"sqlalchemy", "aiosqlite", "asyncpg", "asyncmy"}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCK:
                raise ModuleNotFoundError(f"No module named '{name}'", name=name)
            return None

    sys.meta_path.insert(0, _Blocker())
    for m in list(sys.modules):
        if m.split(".")[0] in _BLOCK:
            del sys.modules[m]

    # 1-3: core imports must succeed without the optional deps.
    import linktools.ai              # noqa
    import linktools.ai.storage      # noqa
    from linktools.ai.storage import Storage, FilesystemStorage  # noqa

    # 4: SqlAlchemyStorage access must raise ImportError carrying an install hint.
    try:
        linktools.ai.storage.SqlAlchemyStorage
        raise SystemExit("FAIL: no ImportError when accessing SqlAlchemyStorage")
    except ImportError as exc:
        msg = str(exc)
        if "linktools-ai[" not in msg:
            raise SystemExit(f"FAIL: missing install hint in: {msg!r}")

    # 5: root must not re-export SqlAlchemyStorage (optional dependency).
    assert not hasattr(linktools.ai, "SqlAlchemyStorage"), "root exported SA storage"

    print("OK")
    """
)


def test_core_imports_without_sqlalchemy():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.join(_ROOT, "linktools", "src"),
            os.path.join(_ROOT, "linktools-ai", "src"),
        ]
    )
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK"), result.stdout
