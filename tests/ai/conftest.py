#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import signal
from collections.abc import Generator
from types import FrameType

import pytest

_TIMEOUT_SECONDS = 90


def _raise_timeout(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise KeyboardInterrupt("diagnostic per-test timeout")


@pytest.fixture(autouse=True)
def _local_execution_backend_optional_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    monkeypatch.setattr(LocalExecutionBackend, "_metric_recorder", None, raising=False)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(
    item: pytest.Item,
    nextitem: pytest.Item | None,
) -> Generator[None, None, None]:
    del item, nextitem
    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
