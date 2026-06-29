#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading as _threading
import typing as _t


_logger = None
_event_handler_lock = _threading.RLock()
_event_handler_name = "__EventHandlerMixin_event_handler"


def _get_logger():
    global _logger
    if _logger is None:
        from ..core import environ
        _logger = environ.get_logger("events")
    return _logger


class _EventHandler(dict):

    def __init__(self):
        super().__init__()
        self._lock = _threading.RLock()

    @property
    def lock(self) -> "_threading.RLock":
        return self._lock


class EventHandlerMixin(object):
    """Dispatch named events to registered handlers."""

    @property
    def _event_handler(self) -> "_EventHandler":
        value = getattr(self, _event_handler_name, None)
        if value is None:
            with _event_handler_lock:
                value = getattr(self, _event_handler_name, None)
                if value is None:
                    value = _EventHandler()
                    setattr(self, _event_handler_name, value)
        return value

    def on(self, event: str, callback: "_t.Callable[..., _t.Any]", times: int = None):
        logger = _get_logger()
        handler = self._event_handler
        with handler.lock:
            logger.debug(f"Register event `{event}` handler `{callback}`")
            callbacks = handler.get(event, None)
            if callbacks is None:
                callbacks = handler[event] = dict()
            callbacks[callback] = {
                "time": 0,
                "max_times": times,
            }

    def off(self, event: str, callback: "_t.Callable[..., _t.Any]"):
        logger = _get_logger()
        handler = self._event_handler
        with handler.lock:
            logger.debug(f"Unregister event `{event}` handler `{callback}`")
            if event in handler:
                callbacks = handler.get(event)
                try:
                    callbacks.pop(callback)
                except KeyError:
                    pass

    def once(self, event: str, callback: "_t.Callable[..., _t.Any]"):
        self.on(event, callback, 1)

    def trigger(self, event: str, *args: "_t.Any", **kwargs: "_t.Any"):
        logger = _get_logger()
        handler = self._event_handler
        invoke_list, remove_list = [], []
        with handler.lock:
            if event in handler:
                callbacks = handler.get(event)
                for callback, info in callbacks.items():
                    invoke_list.append(callback)
                    info["time"] += 1
                    if info["max_times"] is not None and info["time"] >= info["max_times"]:
                        remove_list.append(callback)
            for callback in remove_list:
                callbacks.pop(callback)
        logger.debug(f"Event `{event}` invoke {len(invoke_list)} callbacks")
        for callback in invoke_list:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Event `{event}` handler `{callback}` error", exc_info=e)
