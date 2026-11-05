#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measurement wrappers that leave persistence decisions unchanged."""
import functools
import hashlib
import inspect
import json
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.engine import Connection, ExecutionContext

_PATH = ContextVar('official_runtime_trace_path', default=())
_FACT = ContextVar('official_runtime_trace_fact', default=None)


class Trace:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.turn = -1
        self.rows = []
        self.transactions = []
        self.current = {}
        self.active = False
        self.counter = 0
        self.facts = []
        self.calls = []
        self.modules = {}
        seen = set()
        for name, module in tuple(sys.modules.items()):
            if not name.startswith(('linktools.ai.', 'pydantic_ai_harness.step_persistence')):
                continue
            source = getattr(module, '__file__', None)
            if source and Path(source).is_file():
                self.modules[name] = {'path': source, 'sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()}
            for attr, cls in tuple(vars(module).items()):
                if inspect.isfunction(cls) and inspect.iscoroutinefunction(cls) and cls.__module__ == name:
                    monkeypatch.setattr(module, attr, self.wrap(cls, name + ':' + attr))
                if not inspect.isclass(cls) or cls.__module__ != name or id(cls) in seen:
                    continue
                seen.add(id(cls))
                for method, fn in tuple(vars(cls).items()):
                    if inspect.iscoroutinefunction(fn):
                        monkeypatch.setattr(cls, method, self.wrap(fn, name + ':' + cls.__name__ + '.' + method))
                    elif inspect.isasyncgenfunction(fn):
                        monkeypatch.setattr(cls, method, self.wrap_gen(fn, name + ':' + cls.__name__ + '.' + method))

    def wrap(self, fn: Callable[..., Awaitable[Any]], label: str) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(fn)
        async def call(*args: Any, **kwargs: Any) -> Any:
            token = _PATH.set((*_PATH.get(), label))
            fact_token = None
            if self.active:
                self.calls.append({'turn': self.turn, 'label': label, 'path': _PATH.get()})
            if label.endswith(('SqlStepStore.append_event', 'SqlStepStore.save_snapshot', 'SqlStepStore.register_run')):
                obj = args[1] if len(args) > 1 else next(iter(kwargs.values()))
                fact = {'label': label, 'turn': self.turn}
                for field in ('kind', 'node', 'status', 'step_index', 'sequence', 'idempotency_key', 'run_id', 'reason', 'event_type', 'snapshot_id'):
                    value = getattr(obj, field, None)
                    if value is not None:
                        fact[field] = str(value)
                if hasattr(obj, 'model_dump'):
                    fact['keys'] = list(obj.model_dump())
                    fact['data'] = obj.model_dump(mode='json', exclude={'messages'})
                messages = getattr(obj, 'messages', None)
                if messages is not None:
                    from pydantic_ai.messages import ModelMessagesTypeAdapter
                    fact['message_count'] = len(messages)
                    fact['message_bytes'] = len(ModelMessagesTypeAdapter.dump_json(messages))
                    fact['messages_sha256'] = hashlib.sha256(ModelMessagesTypeAdapter.dump_json(messages)).hexdigest()
                self.facts.append(fact)
                fact_token = _FACT.set(len(self.facts) - 1)
            try:
                return await fn(*args, **kwargs)
            finally:
                if fact_token is not None:
                    _FACT.reset(fact_token)
                _PATH.reset(token)
        return call

    def wrap_gen(self, fn: Callable[..., AsyncIterator[Any]], label: str) -> Callable[..., AsyncIterator[Any]]:
        @functools.wraps(fn)
        async def call(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            generator = fn(*args, **kwargs)
            while True:
                token = _PATH.set((*_PATH.get(), label))
                try:
                    value = await anext(generator)
                except StopAsyncIteration:
                    return
                finally:
                    _PATH.reset(token)
                yield value
        return call

    def begin(self, connection: Connection) -> None:
        if not self.active:
            return
        self.counter += 1
        item = {'id': self.counter, 'turn': self.turn, 'begin_path': _PATH.get(), 'rows': []}
        self.current[id(connection)] = item
        self.transactions.append(item)

    def sql(self, connection: Connection, statement: str, context: ExecutionContext) -> None:
        if not self.active:
            return
        item = self.current.get(id(connection))
        tables = re.findall(r'(?:FROM|INTO|UPDATE|JOIN)\s+(ai_[a-z_]+)', statement, flags=re.I)
        row = {'turn': self.turn, 'transaction': None if item is None else item['id'],
               'word': statement.lstrip().split()[0].upper(), 'tables': list(dict.fromkeys(tables)),
               'statement': statement, 'path': _PATH.get(), 'fact': _FACT.get()}
        params = getattr(context, 'compiled_parameters', None)
        if params:
            row['kinds'] = sorted({str(v) for p in params for k,v in p.items() if re.match(r'^(kind|state)(_|$)', k)})
            row['identity'] = {k: str(v) for p in params for k,v in p.items() if k in ('scope','key','kind','record_key')}
        self.rows.append(row)
        if item is not None:
            item['rows'].append(len(self.rows)-1)

    def commit(self, connection: Connection) -> None:
        item = self.current.pop(id(connection), None)
        if item is not None:
            item['commit_path'] = _PATH.get()
            item['writes'] = sum(self.rows[i]['word'] in {'INSERT','UPDATE','DELETE'} for i in item['rows'])

    def write(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({'rows': self.rows, 'transactions': self.transactions, 'facts': self.facts, 'calls': self.calls, 'modules_at_install': self.modules}, indent=2, default=str))
