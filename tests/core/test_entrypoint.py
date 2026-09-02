#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def test_select_entry_points_prefers_select_over_mapping_interface(monkeypatch):
    from linktools.core import _entrypoint

    class SelectableGroups(dict):
        def __init__(self):
            super().__init__({"linktools.capability": ("legacy",)})
            self.select_calls = []

        def select(self, **kwargs):
            self.select_calls.append(kwargs)
            return ("selected",)

        def get(self, *args, **kwargs):
            raise AssertionError("deprecated mapping interface must not be used")

    entries = SelectableGroups()
    _entrypoint.get_entry_points.cache_clear()
    try:
        monkeypatch.setattr(_entrypoint.metadata, "entry_points", lambda: entries)
        assert _entrypoint.select_entry_points("linktools.capability") == ("selected",)
        assert entries.select_calls == [{"group": "linktools.capability"}]
    finally:
        _entrypoint.get_entry_points.cache_clear()
