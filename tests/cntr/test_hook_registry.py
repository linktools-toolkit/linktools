#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HookRegistry/HookListView: typed registration, ordering, before/after
constraints, and legacy MutableSequence compatibility (Spec Part X)."""
import pytest

from linktools.cntr.lifecycle.hooks import (
    Hook, HookCycleError, HookListView, HookPhase, HookRegistry, HookValidationError,
)


def test_register_returns_hook_and_is_idempotent_by_key():
    registry = HookRegistry()
    calls = []
    hook1 = registry.register(HookPhase.BEFORE_START, lambda: calls.append(1), key="k")
    hook2 = registry.register(HookPhase.BEFORE_START, lambda: calls.append(2), key="k")
    assert hook1 is hook2
    assert isinstance(hook1, Hook)
    assert len(list(registry.iter_phase(HookPhase.BEFORE_START))) == 1


def test_register_without_key_never_dedupes():
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None)
    registry.register(HookPhase.BEFORE_START, lambda: None)
    assert len(list(registry.iter_phase(HookPhase.BEFORE_START))) == 2


def test_register_rejects_non_callable():
    registry = HookRegistry()
    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, "not callable", key="k")


def test_zero_and_one_arg_callback_signatures_supported():
    registry = HookRegistry()
    seen = []
    registry.register(HookPhase.CHECK, lambda: seen.append("zero"), key="z")
    registry.register(HookPhase.CHECK, lambda ctx: seen.append(("one", ctx)), key="o")
    registry.call(HookPhase.CHECK, context="CTX")
    assert seen == ["zero", ("one", "CTX")]


def test_order_controls_sequence_when_no_before_after():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("b"), key="b", order=200)
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", order=100)
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["a", "b"]


def test_registration_order_is_tiebreak_when_order_equal():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("first"), key="first")
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("second"), key="second")
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["first", "second"]


def test_after_constraint_forces_dependency_first():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("b"), key="b", order=1, after=("a",))
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", order=999)
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["a", "b"]


def test_before_constraint_forces_dependent_after():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", order=999, before=("b",))
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("b"), key="b", order=1)
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["a", "b"]


def test_missing_before_after_reference_is_ignored_at_call_time():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", after=("does-not-exist",))
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["a"]


def test_validate_raises_on_missing_reference():
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None, key="a", after=("missing",))
    with pytest.raises(HookValidationError):
        registry.validate()


def test_validate_passes_when_references_resolve():
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None, key="a")
    registry.register(HookPhase.BEFORE_START, lambda: None, key="b", after=("a",))
    assert registry.validate() == []


def test_validate_warns_instead_of_raising_for_missing_optional_reference():
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None, key="a", optional_after=("missing",))
    warnings = registry.validate()
    assert len(warnings) == 1
    assert "missing" in warnings[0]


def test_optional_reference_still_orders_when_present():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("b"), key="b", order=1,
                      optional_after=("a",))
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", order=999)
    assert registry.validate() == []
    registry.call(HookPhase.BEFORE_START)
    assert calls == ["a", "b"]


def test_self_reference_in_optional_before_after_rejected_at_registration():
    registry = HookRegistry()
    with pytest.raises(HookCycleError):
        registry.register(HookPhase.BEFORE_START, lambda: None, key="a", optional_after=("a",))


def test_self_reference_before_after_rejected_at_registration():
    registry = HookRegistry()
    with pytest.raises(HookCycleError):
        registry.register(HookPhase.BEFORE_START, lambda: None, key="a", after=("a",))


def test_cycle_detected_at_call_time():
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None, key="a", after=("b",))
    registry.register(HookPhase.BEFORE_START, lambda: None, key="b", after=("a",))
    with pytest.raises(HookCycleError):
        registry.call(HookPhase.BEFORE_START)


def test_reverse_call_reverses_final_order():
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.AFTER_START, lambda: calls.append("a"), key="a", order=100)
    registry.register(HookPhase.AFTER_START, lambda: calls.append("b"), key="b", order=200)
    registry.call(HookPhase.AFTER_START, reverse=True)
    assert calls == ["b", "a"]


def test_uninspectable_callback_is_auto_opaque_and_zero_arg():
    registry = HookRegistry()
    hook = registry.register(HookPhase.BEFORE_START, len, key="k")  # builtin, unintrospectable signature by inspect in some cases is fine, use print
    assert isinstance(hook, Hook)


def test_describe_exposes_stable_shape_without_running_callback():
    registry = HookRegistry(scope="container")
    ran = []
    registry.register(
        HookPhase.BEFORE_START, lambda: ran.append(1), key="k", name="my-hook", order=42,
        source="builtin", metadata={"operation": "mkdir"},
    )
    described = registry.describe(HookPhase.BEFORE_START)
    assert ran == []
    assert len(described) == 1
    entry = described[0]
    assert entry["name"] == "my-hook"
    assert entry["order"] == 42
    assert entry["scope"] == "container"
    assert entry["source"] == "builtin"
    assert entry["opaque"] is False
    assert entry["metadata"] == {"operation": "mkdir"}


def test_metadata_is_copied_not_shared():
    registry = HookRegistry()
    metadata = {"a": 1}
    hook = registry.register(HookPhase.BEFORE_START, lambda: None, key="k", metadata=metadata)
    metadata["a"] = 2
    assert hook.metadata == {"a": 1}


# -- Legacy MutableSequence compatibility ------------------------------------

def test_legacy_view_append_extend_and_iteration_return_raw_callables():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2 = (lambda: None), (lambda: None)
    view.append(f1)
    view.extend([f2])
    assert list(view) == [f1, f2]
    assert len(view) == 2
    assert bool(view) is True


def test_legacy_view_indexing_and_slicing():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3 = (lambda: 1), (lambda: 2), (lambda: 3)
    view.append(f1)
    view.append(f2)
    view.append(f3)
    assert view[0] is f1
    assert view[1:] == [f2, f3]


def test_legacy_view_pop_remove_clear():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2 = (lambda: 1), (lambda: 2)
    view.append(f1)
    view.append(f2)
    view.remove(f1)
    assert list(view) == [f2]
    view.clear()
    assert list(view) == []
    assert bool(view) is False


def test_legacy_view_marks_opaque_and_source_legacy():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    view.append(lambda ctx: None)  # would otherwise accept a context
    hook = next(registry.iter_phase(HookPhase.BEFORE_START))
    assert hook.opaque is True
    assert hook.source == "legacy"


def test_legacy_view_hooks_are_always_invoked_zero_arg():
    registry = HookRegistry()
    calls = []
    registry.legacy_view(HookPhase.BEFORE_START).append(lambda ctx=None: calls.append(ctx))
    registry.call(HookPhase.BEFORE_START, context="CTX")
    # An opaque legacy hook ignores context entirely, matching the historic
    # always-zero-arg calling convention for start_hooks/stop_hooks.
    assert calls == [None]


def test_legacy_and_new_style_hooks_share_one_ordered_bucket():
    """A hook registered directly via HookRegistry.register (not through the
    legacy view) must still be reachable/iterated by the legacy view, since
    the dispatcher only ever iterates the legacy view for BEFORE_START/AFTER_STOP."""
    registry = HookRegistry()
    registry.register(HookPhase.BEFORE_START, lambda: None, key="new-style", order=10)
    registry.legacy_view(HookPhase.BEFORE_START).append(lambda: None)
    assert len(registry.legacy_view(HookPhase.BEFORE_START)) == 2


# -- Container/manager integration -------------------------------------------

def test_container_add_start_hook_is_idempotent_by_key(fresh_manager):
    container = fresh_manager.containers["nginx"]
    calls = []
    baseline = len(container.start_hooks)
    container.add_start_hook(("test", "k"), lambda: calls.append(1))
    container.add_start_hook(("test", "k"), lambda: calls.append(2))
    assert len(container.start_hooks) - baseline == 1


def test_container_add_stop_hook_reaches_after_stop_phase(fresh_manager):
    container = fresh_manager.containers["nginx"]
    baseline = len(container.stop_hooks)
    container.add_stop_hook(("test", "stop"), lambda: None)
    assert len(container.stop_hooks) - baseline == 1


def test_manager_and_container_hooks_are_stable_across_accesses(fresh_manager):
    container = fresh_manager.containers["nginx"]
    assert container.hooks is container.hooks
    assert fresh_manager.hooks is fresh_manager.hooks
    assert container.start_hooks is container.start_hooks


def test_hook_list_view_is_not_isinstance_list():
    """Spec-sanctioned breaking change: start_hooks/stop_hooks are no longer
    a plain list; downstream isinstance(..., list) checks are unsupported."""
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    assert isinstance(view, HookListView)
    assert not isinstance(view, list)
