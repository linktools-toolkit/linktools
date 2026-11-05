#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logging preserves messages and attaches scoped thread-local context."""

import logging
import threading

import pytest

from linktools.core._logging import LoggingManager


@pytest.fixture
def manager():
    value = LoggingManager()
    try:
        yield value
    finally:
        value.close()


def _capture(manager, name, msg, *args):
    logger = manager.get_logger(name)
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        logger.info(msg, *args)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    return records


@pytest.mark.parametrize("message,args,expected", [
    ("password=%s status=401", ("test-value",), "password=test-value status=401"),
    ("clone %s", ("https://example:dummy@example.test/repo",),
     "clone https://example:dummy@example.test/repo"),
    ("Authorization: Bearer example-token", (), "Authorization: Bearer example-token"),
    ("payload %s", ({"api_key": "fixture-value"},), "payload {'api_key': 'fixture-value'}"),
])
def test_log_records_preserve_message_and_arguments(manager, message, args, expected):
    records = _capture(manager, "lt.test.messages", message, *args)
    assert len(records) == 1
    assert records[0].msg == message
    assert records[0].args == (args[0] if len(args) == 1 and isinstance(args[0], dict) else args)
    assert records[0].getMessage() == expected


def test_context_attaches_fields_and_pops_on_exit(manager):
    assert manager.current_context() == {}
    with manager.context(task_id="outer"):
        with manager.context(task_id="inner", device="phone"):
            assert manager.current_context() == {"task_id": "inner", "device": "phone"}
        assert manager.current_context() == {"task_id": "outer"}
    assert manager.current_context() == {}


def test_context_is_thread_local(manager):
    barrier = threading.Event()
    seen = {}

    def worker():
        with manager.context(device="emulator-5554"):
            barrier.wait()
            seen["worker"] = manager.current_context()

    thread = threading.Thread(target=worker)
    thread.start()
    with manager.context(device="main-thread-device"):
        barrier.set()
        thread.join()
        assert manager.current_context() == {"device": "main-thread-device"}
    assert seen["worker"] == {"device": "emulator-5554"}


def test_context_fields_reach_records_without_rewriting_messages(manager):
    with manager.context(task_id="abc", password="fixture-value"):
        records = _capture(manager, "lt.test.context", "token=%s", "fixture-token")
    assert records[0].task_id == "abc"
    assert records[0].password == "fixture-value"
    assert records[0].getMessage() == "token=fixture-token"
    assert not hasattr(_capture(manager, "lt.test.context", "next")[0], "task_id")


def test_logger_access_and_close_preserve_factory_lifecycle(manager):
    factory_before = logging.getLogRecordFactory()
    logger = manager.get_logger("lt.test.lifecycle")
    first = logging.getLogRecordFactory()
    assert manager.get_logger("lt.test.lifecycle") is logger
    assert logging.getLogRecordFactory() is first
    manager.close()
    assert logging.getLogRecordFactory() is factory_before


def test_closing_one_manager_retains_other_context(manager):
    other = LoggingManager()
    try:
        manager.get_logger("lt.test.multiple")
        other.get_logger("lt.test.multiple")
        with manager.context(first="one"), other.context(second="two"):
            before = _capture(other, "lt.test.multiple", "before")[0]
            assert (before.first, before.second) == ("one", "two")
            manager.close()
            after = _capture(other, "lt.test.multiple", "after")[0]
            assert not hasattr(after, "first")
            assert after.second == "two"
    finally:
        other.close()
