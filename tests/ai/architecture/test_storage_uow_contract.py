#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RF-09 architecture gate: the StorageUnitOfWork Protocol shape is frozen by
contract (plan §4.5) and Storage.transaction() returns the PUBLIC UoW type.

This test exists because an earlier round silently weakened the contract:
artifact_records was relaxed to ``| None`` (the spec marks it REQUIRED), a
store field was faked with a self-committing backend, and transaction() kept
returning the private ``_UnitOfWork``. Those slipped past the implementer's own
tests because the tests checked what was built, not what §4.5 demanded. The
checks below make the frozen shape a merge gate."""

import ast
from pathlib import Path

from linktools.ai.storage.protocols import StorageUnitOfWork

_SRC = Path(__file__).resolve().parents[3] / "linktools-ai" / "src" / "linktools" / "ai"


def _uow_annotations() -> "dict[str, str]":
    # Quoted annotations may be stored either as the evaluated value (the bare
    # string ``AssetStore``) or as the source text including quote chars
    # (``'AssetStore'``); strip surrounding quotes so the test is robust to
    # whichever form the runtime chose.
    raw = dict(StorageUnitOfWork.__annotations__)
    out: "dict[str, str]" = {}
    for k, v in raw.items():
        s = v if isinstance(v, str) else getattr(v, "__name__", str(v))
        out[k] = s.strip().strip("'\"")
    return out


def _transaction_return(module_path: Path, class_name: str) -> str:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == class_name:
            for item in cls.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "transaction"
                ):
                    return ast.unparse(item.returns)
            raise AssertionError(
                f"{class_name} has no transaction() method in {module_path}"
            )
    raise AssertionError(f"class {class_name} not found in {module_path}")


def test_uow_field_set_matches_the_frozen_contract():
    # §4.5 fixed interface: exactly these nine stores, no more, no fewer.
    ann = _uow_annotations()
    assert set(ann) == {
        "assets",
        "artifact_records",
        "sessions",
        "runs",
        "events",
        "checkpoints",
        "approvals",
        "idempotency",
        "jobs",
    }, f"StorageUnitOfWork field set drifted: {set(ann)}"


def test_uow_required_stores_are_not_optional():
    # §4.5: artifact_records is REQUIRED (no ``| None``); assets + jobs are
    # optional. assets is Optional because a session-bound asset backend is not
    # yet wired (the SQLAlchemy asset backend takes a session_factory, not a
    # per-call AsyncSession, so it cannot join the UoW's single session without
    # self-committing and breaking atomicity) -- None is the honest value, not a
    # fake. jobs is Optional for backward compatibility. Relaxing
    # artifact_records to optional would be a regression.
    ann = _uow_annotations()
    assert ann["assets"] == "AssetStore | None", ann["assets"]
    assert ann["artifact_records"] == "ArtifactRecordStore", ann["artifact_records"]
    assert ann["jobs"] == "JobStore | None", ann["jobs"]


def test_uow_store_fields_carry_no_any():
    # §4.5 / RF-09: no store field is typed ``Any``.
    ann = _uow_annotations()
    offenders = [name for name, value in ann.items() if "Any" in value]
    assert not offenders, f"UoW store fields use Any: {offenders}"


def test_storage_transaction_returns_public_uow():
    # §4.5: Storage.transaction() returns the PUBLIC StorageUnitOfWork, never the
    # private _UnitOfWork concrete class.
    ret = _transaction_return(_SRC / "storage" / "facade.py", "Storage")
    assert "StorageUnitOfWork" in ret, (
        f"Storage.transaction() returns {ret!r}; must be the public StorageUnitOfWork"
    )
    assert "_UnitOfWork" not in ret, (
        f"Storage.transaction() leaks the private _UnitOfWork: {ret!r}"
    )


def test_sqlalchemy_transaction_manager_returns_public_uow():
    ret = _transaction_return(
        _SRC / "storage" / "sqlalchemy" / "facade.py",
        "_SqlAlchemyTransactionManager",
    )
    assert "StorageUnitOfWork" in ret, ret
    assert "_UnitOfWork" not in ret, ret


def test_filesystem_transaction_manager_returns_public_uow():
    ret = _transaction_return(
        _SRC / "storage" / "transaction.py",
        "NoCrossStoreTransactions",
    )
    assert "StorageUnitOfWork" in ret, ret
    assert "_UnitOfWork" not in ret, ret
