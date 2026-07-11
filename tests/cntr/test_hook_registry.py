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


def test_missing_required_before_after_reference_blocks_call():
    """call() must auto-validate before running anything -- a missing
    *required* before/after reference is never silently skipped."""
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("a"), key="a", after=("does-not-exist",))
    with pytest.raises(HookValidationError):
        registry.call(HookPhase.BEFORE_START)
    assert calls == []


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


# -- Callback invocation-mode validation (Spec Part VI section 38) -----------

def test_two_required_positional_params_rejected_at_registration():
    registry = HookRegistry()

    def two_required(a, b):
        pass

    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, two_required, key="bad")


def test_keyword_only_required_param_rejected_at_registration():
    registry = HookRegistry()

    def kwonly_required(*, x):
        pass

    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, kwonly_required, key="bad")


def test_star_args_callback_prefers_context_invocation():
    registry = HookRegistry()
    seen = []
    registry.register(HookPhase.BEFORE_START, lambda *args: seen.append(args), key="k")
    registry.call(HookPhase.BEFORE_START, context="CTX")
    assert seen == [("CTX",)]


def test_optional_context_param_receives_context_when_given():
    registry = HookRegistry()
    seen = []
    registry.register(HookPhase.BEFORE_START, lambda ctx=None: seen.append(ctx), key="k")
    registry.call(HookPhase.BEFORE_START, context="CTX")
    assert seen == ["CTX"]


def test_optional_context_param_called_zero_arg_when_context_is_none():
    registry = HookRegistry()
    seen = []
    registry.register(HookPhase.BEFORE_START, lambda ctx=None: seen.append(ctx), key="k")
    registry.call(HookPhase.BEFORE_START, context=None)
    assert seen == [None]


# -- key/metadata validation (Spec section 39) --------------------------------

def test_hook_key_must_be_hashable():
    registry = HookRegistry()
    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, lambda: None, key=["not", "hashable"])


def test_hook_metadata_must_be_json_compatible():
    registry = HookRegistry()
    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, lambda: None, key="k",
                          metadata={"callback": lambda: None})


def test_hook_metadata_rejects_arbitrary_object():
    class _Thing:
        pass

    registry = HookRegistry()
    with pytest.raises(HookValidationError):
        registry.register(HookPhase.BEFORE_START, lambda: None, key="k",
                          metadata={"thing": _Thing()})


def test_hook_metadata_accepts_nested_json_compatible_values():
    registry = HookRegistry()
    hook = registry.register(HookPhase.BEFORE_START, lambda: None, key="k",
                             metadata={"a": [1, "two", {"b": None, "c": True}]})
    assert hook.metadata == {"a": [1, "two", {"b": None, "c": True}]}


# -- HookListView.insert() real positional semantics (Spec section 41) -------

def test_legacy_insert_at_start():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2 = (lambda: 1), (lambda: 2)
    view.append(f1)
    view.insert(0, f2)
    assert list(view) == [f2, f1]


def test_legacy_insert_in_middle():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3 = (lambda: 1), (lambda: 2), (lambda: 3)
    view.append(f1)
    view.append(f2)
    view.insert(1, f3)
    assert list(view) == [f1, f3, f2]


def test_legacy_insert_at_end():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3 = (lambda: 1), (lambda: 2), (lambda: 3)
    view.append(f1)
    view.append(f2)
    view.insert(len(view), f3)
    assert list(view) == [f1, f2, f3]


def test_legacy_insert_index_beyond_end_clamps_to_append():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2 = (lambda: 1), (lambda: 2)
    view.append(f1)
    view.insert(999, f2)
    assert list(view) == [f1, f2]


def test_legacy_insert_negative_index_clamps_to_start():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2 = (lambda: 1), (lambda: 2)
    view.append(f1)
    view.insert(-5, f2)
    assert list(view) == [f2, f1]


def test_legacy_insert_does_not_disturb_formal_hook_position():
    """A formal hook's own order value fixes its position; a legacy insert
    can only reposition other legacy hooks around it."""
    registry = HookRegistry()
    calls = []
    registry.register(HookPhase.BEFORE_START, lambda: calls.append("formal"), key="formal", order=250)
    view = registry.legacy_view(HookPhase.BEFORE_START)
    view.append(lambda: calls.append("legacy-1"))
    view.insert(0, lambda: calls.append("legacy-0"))
    registry.call(HookPhase.BEFORE_START)
    # order=250 sorts before the legacy segment's order=500, regardless of
    # where legacy-0/legacy-1 land relative to each other.
    assert calls == ["formal", "legacy-0", "legacy-1"]


# -- Slice assignment/deletion (Spec section 42) ------------------------------

def test_slice_assignment_contiguous_replace():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3, f4 = (lambda: 1), (lambda: 2), (lambda: 3), (lambda: 4)
    view.append(f1)
    view.append(f2)
    view.append(f3)
    view[1:2] = [f4]
    assert list(view) == [f1, f4, f3]


def test_slice_assignment_contiguous_grow():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3 = (lambda: 1), (lambda: 2), (lambda: 3)
    view.append(f1)
    view.append(f2)
    view[1:1] = [f3]
    assert list(view) == [f1, f3, f2]


def test_slice_assignment_extended_step_requires_matching_length():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3, f4 = (lambda: 1), (lambda: 2), (lambda: 3), (lambda: 4)
    view.append(f1)
    view.append(f2)
    view.append(f3)
    view.append(f4)
    view[::2] = [lambda: 10, lambda: 30]
    assert len(view) == 4
    with pytest.raises(ValueError):
        view[::2] = [lambda: 1]


def test_slice_deletion():
    registry = HookRegistry()
    view = registry.legacy_view(HookPhase.BEFORE_START)
    f1, f2, f3 = (lambda: 1), (lambda: 2), (lambda: 3)
    view.append(f1)
    view.append(f2)
    view.append(f3)
    del view[1:2]
    assert list(view) == [f1, f3]


# -- Plan auto-validates hook ordering ----------------------------------------

def test_plan_fails_when_hook_ordering_is_invalid(fresh_manager):
    from linktools.cntr.container import ContainerError
    nginx = fresh_manager.containers["nginx"]
    nginx.hooks.register(HookPhase.BEFORE_START, lambda: None, key="broken", after=("missing-dep",))
    with pytest.raises(ContainerError):
        fresh_manager.planner.plan("up", names=["nginx"])
