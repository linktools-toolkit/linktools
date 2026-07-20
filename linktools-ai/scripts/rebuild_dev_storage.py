#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-click dev-storage rebuild CLI (plan Phase 9 op 11).

A developer-facing tool (NOT part of the installable core) that wipes +
reconstructs the Filesystem data dir and the SQLite dev DB from scratch,
constructing the SQLAlchemy engine itself (the core rebuild helpers in
``linktools.ai.storage.rebuild`` never construct engines -- §6.4 adapter
boundary). Run::

    python linktools-ai/scripts/rebuild_dev_storage.py \\
        --data-root ./data --db-path ./dev.db

Omit ``--db-path`` to rebuild Filesystem only."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=Path("./data"))
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite dev DB path; omit to rebuild Filesystem only.",
    )
    args = parser.parse_args()

    # The core rebuild helpers do not construct engines; this dev tool does.
    from linktools.ai.storage.rebuild import rebuild_dev_storage

    sqlite_engine = None
    if args.db_path is not None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine
        except ImportError as exc:
            raise SystemExit(
                "rebuilding the SQLite dev DB needs the optional extra; "
                "install with: pip install 'linktools-ai[sqlite]'"
            ) from exc
        # Wipe the old DB file, then construct a fresh engine on it.
        if args.db_path.exists():
            args.db_path.unlink()
        args.db_path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_engine = create_async_engine(
            f"sqlite+aiosqlite:///{args.db_path}"
        )

    try:
        summary = rebuild_dev_storage(
            data_root=args.data_root, sqlite_engine=sqlite_engine
        )
    finally:
        if sqlite_engine is not None:
            asyncio.run(sqlite_engine.dispose())

    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
