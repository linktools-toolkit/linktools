#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Architecture locks for the v5 bug/security closure (guide ).

One representative check per fixed item (2 security + 1 isolation + 5
correctness), so a future change -- or a deleted per-area test file -- cannot
silently re-introduce the gap each fix closed. Mixes behavioral locks with
source/structure inspection (the same style test_final_closure_invariants uses).
"""

import dataclasses
import inspect

import pytest

from linktools.ai.runtime.assembly.lifecycle import resolve_session
from linktools.ai.errors import SessionAccessDeniedError
from linktools.ai.sandbox.local import _run_file_tool_sync
from linktools.ai.runtime import Runtime
from linktools.ai.governance.security.pipeline import validate_model_decision
from linktools.ai.governance.security.secured_model import SecuredModel
from linktools.ai.session.models import SessionRecord
from linktools.ai.storage.facade import FilesystemStorage


# --- symlink reads confined to resolved roots -----------------------


def test_v5_read_path_resolves_and_rejects_symlink_escape(tmp_path):
    runtime = tmp_path / "runtime"
    outside = tmp_path / "outside"
    runtime.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP-SECRET")
    (runtime / "link.txt").symlink_to(outside / "secret.txt")

    result = _run_file_tool_sync("read_file", {"path": "runtime/link.txt"}, runtime, [])

    assert "error" in result
    assert "TOP-SECRET" not in str(result)


# --- every model pipeline decision enforced -------------------------


def test_v5_secured_model_honors_modify_and_validates():
    src = inspect.getsource(SecuredModel)
    # MODIFY payload applied whenever non-None (composite pipelines return
    # ALLOW with a payload), and a model decision validator exists.
    assert "validate_model_decision" in src
    assert "_replace_last_user_text" in src
    assert "_replace_model_response_output" in src
    # validate_model_decision rejects REQUIRE_APPROVAL at before_model.
    from linktools.ai.governance.security.pipeline import PipelineAction, PipelineDecision
    from linktools.ai.errors import PipelineExecutionError

    with pytest.raises(PipelineExecutionError):
        validate_model_decision(
            PipelineDecision(action=PipelineAction.REQUIRE_APPROVAL), stage="before"
        )


# --- sessions bound to principal/tenant -----------------------------


def test_v5_session_record_carries_owner_and_resolve_enforces(tmp_path):
    fields = {f.name for f in dataclasses.fields(SessionRecord)}
    assert {"user_id", "tenant_id"} <= fields

    import asyncio

    storage = FilesystemStorage(root=tmp_path)
    sid = asyncio.run(resolve_session(storage, None, user_id="u-a", tenant_id="t-a"))
    with pytest.raises(SessionAccessDeniedError):
        asyncio.run(resolve_session(storage, sid, user_id="u-a", tenant_id="t-b"))


# --- recovery serialized + retryable --------------------------------


def test_v5_runtime_recovery_is_serialized():
    # The crash-recovery guard lives on RunCoordinator (Runtime delegates all
    # run-lifecycle methods there); the lock is created per-instance.
    from linktools.ai.run.coordinator import RunCoordinator
    from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

    assert hasattr(RunCoordinator, "_ensure_recovered")
    import tempfile

    storage = FilesystemStorage(root=tempfile.mkdtemp())
    rt = Runtime.build(
        storage=storage,
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )
    assert hasattr(rt._coordinator, "_recovery_lock")
    assert hasattr(rt._coordinator, "_recovery_done")


# --- file session messages published only after commit -------------


def test_v5_file_complete_publishes_messages_after_run_transition():
    from linktools.ai.storage.filesystem import commit as commit_mod

    src = inspect.getsource(commit_mod.FilesystemRunCommitCoordinator.complete)
    # Session messages are appended AFTER the run transition (commit point).
    pos_transition = src.index("RUN_TRANSITIONED")
    pos_messages = src.index("_append_messages_once")
    assert pos_transition < pos_messages, (
        "session messages must be published after the SUCCEEDED commit point"
    )


# --- critical-event failures are not swallowed ----------------------


def test_v5_file_critical_event_helper_does_not_swallow():
    from linktools.ai.storage.filesystem import commit as commit_mod

    src = inspect.getsource(
        commit_mod.FilesystemRunCommitCoordinator._append_critical_event_once
    )
    # The append must NOT be wrapped in a broad except (it must propagate so the
    # journal is retained).
    assert "except Exception" not in src, (
        "_append_critical_event_once must not swallow the append failure"
    )
    assert "metadata={" in src and "commit_id" in src


# --- post-commit hooks isolated from terminal state -----------------


def test_v5_runner_runs_after_run_before_complete():
    from linktools.ai.agent.engine import AgentEngine

    src = inspect.getsource(AgentEngine.execute)
    pos_after_run = src.index("run_after_run")
    pos_complete = src.index("_commit_coordinator.complete")
    assert pos_after_run < pos_complete, (
        "after_run must run before the commit so its failure does not corrupt a "
        "committed run"
    )


# --- only the original user prompt is persisted ---------------------


def test_v5_runner_persists_original_prompt_not_model_prompt():
    from linktools.ai.agent import engine as runner_mod

    src = inspect.getsource(runner_mod.AgentEngine.execute)
    assert "user_content = prompt" not in src, (
        "the USER session message must be request.prompt, never the concatenated "
        "model prompt"
    )
    # Session history is folded into the MODEL prompt with explicit role
    # prefixes (so an assistant turn is never disguised as user content). The
    # formatter moved to the prompt domain's PromptBuilder; the
    # runner delegates composition to it. Lock the behavior directly rather
    # than the old "function name in runner source" structural proxy.
    from datetime import datetime, timezone

    from linktools.ai.prompt.builder import PromptBuilder
    from linktools.ai.session.models import MessageRole, SessionMessage

    now = datetime.now(timezone.utc)

    def _msg(role, content):
        return SessionMessage(
            id=f"{role.value}-{content}",
            session_id="s",
            sequence=0,
            role=role,
            content=content,
            run_id=None,
            created_at=now,
        )

    folded = PromptBuilder.format_session_history(
        [_msg(MessageRole.ASSISTANT, "hi"), _msg(MessageRole.USER, "ok")]
    )
    assert "ASSISTANT: hi" in folded and "USER: ok" in folded, (
        "session history must be folded with role prefixes so an assistant turn "
        "is not disguised as user content"
    )
    # And the runner must still drive composition through PromptBuilder (not
    # re-inline it): both call sites are present in execute()'s source.
    assert "PromptBuilder.build_base_prompt" in src and "PromptBuilder.combine" in src
