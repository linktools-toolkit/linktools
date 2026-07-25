"""Process-local entry-point metadata snapshot."""

from functools import lru_cache

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


@lru_cache(maxsize=1)
def get_entry_points():
    return metadata.entry_points()


def select_entry_points(group):
    entries = get_entry_points()
    if isinstance(entries, dict):
        return tuple(entries.get(group, ()))
    select = getattr(entries, "select", None)
    if select is not None:
        return tuple(select(group=group))
    return tuple(entry for entry in entries if getattr(entry, "group", None) == group)
