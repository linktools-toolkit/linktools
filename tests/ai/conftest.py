#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
from collections.abc import Generator
from types import FrameType

import pytest

_TIMEOUT_SECONDS = 90
_MAX_TIMEOUT_SECONDS = 600
_MARKER_NAME = "diagnostic_timeout"


def _raise_timeout(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise KeyboardInterrupt("diagnostic per-test timeout")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "diagnostic_timeout(seconds): set the POSIX diagnostic test timeout",
    )


def pytest_report_header(config: pytest.Config) -> str | None:
    del config
    if not _supports_alarm():
        return "diagnostic per-test SIGALRM timeout: unavailable on this platform"
    return None


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    del config
    marker = pytest.mark.diagnostic_timeout(_MAX_TIMEOUT_SECONDS)
    for item in items:
        if item.originalname == "test_complete_query_io":
            item.add_marker(marker)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item,
    nextitem: pytest.Item | None,
) -> Generator[None, None, None]:
    del nextitem
    timeout = _timeout_for(item)
    if not _supports_alarm():
        yield
        return
    timed_out = False
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def timeout_handler(signum: int, frame: FrameType | None) -> None:
        nonlocal timed_out
        timed_out = True
        _raise_timeout(signum, frame)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(
                signal.ITIMER_REAL,
                previous_timer[0],
                previous_timer[1],
            )
        if timed_out:
            item.add_report_section(
                "call",
                "diagnostic-timeout",
                f"nodeid={item.nodeid}\nbudget_seconds={timeout}",
            )


def _supports_alarm() -> bool:
    return all(
        hasattr(signal, name)
        for name in ("SIGALRM", "ITIMER_REAL", "setitimer", "getitimer")
    )


def _timeout_for(item: pytest.Item) -> int:
    marker = item.get_closest_marker(_MARKER_NAME)
    if marker is None:
        return _TIMEOUT_SECONDS
    if len(marker.args) != 1 or marker.kwargs:
        raise pytest.UsageError(
            f"{_MARKER_NAME} requires one positional integer argument: "
            f"{item.nodeid}"
        )
    value = marker.args[0]
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        or value > _MAX_TIMEOUT_SECONDS
    ):
        raise pytest.UsageError(
            f"{_MARKER_NAME} must be a positive integer <= {_MAX_TIMEOUT_SECONDS}: "
            f"{item.nodeid}"
        )
    return value
