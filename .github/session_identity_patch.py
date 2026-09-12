#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path


path = Path("linktools-ai/src/linktools/ai/runtime/state/_repositories.py")
text = path.read_text(encoding="utf-8")
old = '''def _projected_record(
    repository: _RepositoryBase,
    current: StoredRecord,
    value: object,
) -> StoredRecord:
    _require_tenant(value, repository._tenant_id)
    identity = _canonical_record_identity(current.kind, value)
    projected = repository._stored(
        current.kind, identity, value, state=_record_state(value)
    )
    return replace(projected, storage_version=current.storage_version + 1)


def _require_explicit_session_agent_id(value: SessionRecord) -> None:
'''
new = '''def _projected_record(
    repository: _RepositoryBase,
    current: StoredRecord,
    value: object,
) -> StoredRecord:
    _require_tenant(value, repository._tenant_id)
    if current.kind == "session" and isinstance(value, SessionRecord):
        _require_session_identity(
            _decode_enveloped_domain(current.data, SessionRecord),
            value,
        )
    identity = _canonical_record_identity(current.kind, value)
    projected = repository._stored(
        current.kind, identity, value, state=_record_state(value)
    )
    return replace(projected, storage_version=current.storage_version + 1)


def _require_session_identity(
    current: SessionRecord,
    candidate: SessionRecord,
) -> None:
    if (
        candidate.session_id,
        candidate.tenant_id,
        candidate.owner_principal_id,
        candidate.agent_id,
        candidate.history_id,
    ) != (
        current.session_id,
        current.tenant_id,
        current.owner_principal_id,
        current.agent_id,
        current.history_id,
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _require_explicit_session_agent_id(value: SessionRecord) -> None:
'''
if text.count(old) != 1:
    raise RuntimeError("projected record block did not match exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")

path = Path("tests/ai/test_session_admission.py")
text = path.read_text(encoding="utf-8")
anchor = '''        updated = await state.conversation.sessions.compare_and_swap(
'''
if text.count(anchor) != 1:
    raise RuntimeError("session admission compare_and_swap anchor did not match")
insert = '''        with pytest.raises(AIError) as identity_error:
            await state.conversation.sessions.compare_and_swap(
                admitted.session_id,
                tenant_id=admitted.tenant_id,
                expected_revision=admitted.revision,
                next_record=replace(
                    admitted,
                    revision=admitted.revision + 1,
                    history_id="different-history",
                    updated_at=datetime.now(timezone.utc),
                ),
            )
        assert identity_error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

'''
path.write_text(text.replace(anchor, insert + anchor, 1), encoding="utf-8")
