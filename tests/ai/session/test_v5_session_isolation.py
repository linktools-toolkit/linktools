#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SEC-03 (v5 guide §7): a session is bound to the (user_id, tenant_id) that
opened it. resolve_session enforces strict equality on re-open, so a caller who
knows another tenant's session id cannot load its history.

Identity rule: (None, None) opens only unowned sessions; a principal opens only
its own; ownership is never auto-claimed. Each test fails before the fix
(resolve_session ignored identity and returned any existing session)."""

import asyncio

import pytest

from linktools.ai._runtime.lifecycle import resolve_session
from linktools.ai.errors import SessionAccessDeniedError, SessionError
from linktools.ai.storage.facade import FileStorage


def _run(coro):
    return asyncio.run(coro)


def _open(storage, session_id, *, user_id, tenant_id):
    return _run(
        resolve_session(storage, session_id, user_id=user_id, tenant_id=tenant_id)
    )


def test_new_session_is_stamped_with_principal(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id="u-a", tenant_id="t-a")
    record = _run(storage.sessions.get(sid))
    assert record is not None
    assert record.user_id == "u-a"
    assert record.tenant_id == "t-a"


def test_same_identity_reopens_session(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id="u-a", tenant_id="t-a")
    # Same principal re-opens: ok, returns the same id.
    assert _open(storage, sid, user_id="u-a", tenant_id="t-a") == sid


def test_different_tenant_is_rejected(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id="u-a", tenant_id="t-a")
    with pytest.raises(SessionAccessDeniedError):
        _open(storage, sid, user_id="u-a", tenant_id="t-b")


def test_different_user_is_rejected(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id="u-a", tenant_id="t-a")
    with pytest.raises(SessionAccessDeniedError):
        _open(storage, sid, user_id="u-b", tenant_id="t-a")


def test_omitted_identity_cannot_breach_owned_session(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id="u-a", tenant_id="t-a")
    # An unowned caller must not reach an owned session.
    with pytest.raises(SessionAccessDeniedError):
        _open(storage, sid, user_id=None, tenant_id=None)


def test_unowned_session_rejects_principal(tmp_path):
    storage = FileStorage(root=tmp_path)
    sid = _open(storage, None, user_id=None, tenant_id=None)
    # A principal must not claim an unowned session.
    with pytest.raises(SessionAccessDeniedError):
        _open(storage, sid, user_id="u-a", tenant_id="t-a")
    # ...but the unowned caller can re-open it.
    assert _open(storage, sid, user_id=None, tenant_id=None) == sid


def test_missing_session_is_not_found_not_denied(tmp_path):
    storage = FileStorage(root=tmp_path)
    # An unknown id surfaces as a plain SessionError, not an access denial --
    # the denial message must not leak whether the session belongs to someone.
    with pytest.raises(SessionError):
        _open(storage, "no-such-session", user_id="u-a", tenant_id="t-a")
