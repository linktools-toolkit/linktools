#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched SQLite persistence micro-workload. No model calls, tests or CI.

Run once per source checkout in separate processes with the same interpreter.
SQL column_value_bytes means SQLite length(CAST(value AS BLOB)); it includes all
columns (including generated columns), is NOT SQLite physical storage size, and
uses decimal/text representations for numeric/date values. dbstat is reported
separately for physical B-tree pages, payload and unused bytes. No VACUUM or
manual checkpoint is performed. SQL phase counters exclude measurement queries,
which use a separate stdlib sqlite3 read-only connection.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from sqlalchemy.engine import Connection
    from linktools.ai.execution import ExecutionRecord

_Result = TypeVar("_Result")


def args_and_source() -> tuple[argparse.Namespace, dict[str, str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--copies', type=int, default=20)
    args = parser.parse_args()
    args.source_root = args.source_root.resolve()
    if args.copies < 1:
        parser.error('--copies must be positive')
    for path in (args.database, Path(str(args.database) + '-wal'), Path(str(args.database) + '-shm'), args.output):
        if path.exists():
            parser.error(f'refusing to overwrite existing artifact: {path}')
    def git(*items: str) -> str:
        return subprocess.check_output(['git', '-C', str(args.source_root), *items], text=True).strip()
    sha = git('rev-parse', 'HEAD')
    if sha != git('rev-parse', args.expected_sha):
        parser.error(f'checkout HEAD {sha} does not match --expected-sha')
    if git('status', '--porcelain', '--untracked-files=no'):
        parser.error('source checkout has tracked modifications')
    # Do not use installed linktools packages or repository test imports.
    sources = [args.source_root / name / 'src' for name in ('linktools', 'linktools-common', 'linktools-ai')]
    sys.path[:0] = [str(path) for path in sources if path.is_dir()]
    return args, {'sha': sha, 'tree': git('rev-parse', 'HEAD^{tree}'), 'path': str(args.source_root)}


def quoted(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def storage_snapshot(path: Path) -> dict[str, object]:
    result = {'files': {suffix or 'main': Path(str(path) + suffix).stat().st_size
                       if Path(str(path) + suffix).exists() else 0 for suffix in ('', '-wal', '-shm')}}
    if not path.exists():
        return result
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        result['sqlite_version'] = sqlite3.sqlite_version
        result['page_size'] = connection.execute('PRAGMA page_size').fetchone()[0]
        result['page_count'] = connection.execute('PRAGMA page_count').fetchone()[0]
        result['freelist_count'] = connection.execute('PRAGMA freelist_count').fetchone()[0]
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ai_%' ORDER BY name")]
        summaries = {}
        for table in tables:
            info = list(connection.execute('PRAGMA table_xinfo(' + quoted(table) + ')'))
            columns = [row[1] for row in info if row[6] != 1]
            expressions = ['COALESCE(SUM(COALESCE(length(CAST(' + quoted(name) + ' AS BLOB)),0)),0)' for name in columns]
            values = connection.execute('SELECT count(*),' + ','.join(expressions) + ' FROM ' + quoted(table)).fetchone()
            summary = {'rows': values[0], 'all_column_value_bytes': sum(values[1:]),
                       'column_value_bytes': dict(zip(columns, values[1:])),
                       'generated_columns': [row[1] for row in info if row[6] in (2, 3)]}
            if table == 'ai_state_records':
                rows = connection.execute('SELECT kind,count(*),' + ','.join(expressions)
                                          + ' FROM ' + quoted(table) + ' GROUP BY kind ORDER BY kind').fetchall()
                summary['by_kind'] = {row[0]: {'rows': row[1], 'all_column_value_bytes': sum(row[2:]),
                    'column_value_bytes': dict(zip(columns, row[2:]))} for row in rows}
            summaries[table] = summary
        result['tables'] = summaries
        result['all_business_table_column_value_bytes'] = sum(value['all_column_value_bytes'] for value in summaries.values())
        try:
            rows = connection.execute('SELECT name,count(*),sum(pgsize),sum(payload),sum(unused),max(mx_payload) '
                                      'FROM dbstat GROUP BY name ORDER BY name').fetchall()
            result['dbstat'] = {row[0]: dict(zip(('pages', 'page_bytes', 'payload_bytes', 'unused_bytes', 'max_cell_payload'), row[1:])) for row in rows}
            indexes = [row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='ai_state_records'")]
            names = ['ai_state_records', *indexes]
            result['records_btree_including_indexes'] = {key: sum(result['dbstat'].get(name, {}).get(key, 0) or 0 for name in names)
                                                       for key in ('pages', 'page_bytes', 'payload_bytes', 'unused_bytes')}
        except sqlite3.DatabaseError as error:
            result['dbstat_unavailable'] = str(error)
    return result


async def workload(args: argparse.Namespace, source: dict[str, str]) -> dict[str, object]:
    from sqlalchemy import event
    from sqlalchemy.engine import Engine
    from linktools.ai.agent import AgentBindingContract
    from linktools.ai.core import ExecutionLineageKind, ExecutionStatus, canonical_json_bytes, service_principal
    from linktools.ai.execution import ExecutionRecord
    from linktools.ai.inputs import TaskGraphBindingCaptureStore
    from linktools.ai.model import StoredUserInput
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.persistence.codec import encode_domain
    from linktools.ai.spec import AgentSpec
    from linktools.ai.storage import StoredPayload
    from linktools.ai.task import (TaskBindingContract, Task, TaskGraph, TaskNode,
                                  TaskGraphAdmission, TaskGraphRequest, TaskGraphLimits)

    # Report import origins to prove the requested checkout, not an installed SDK.
    import linktools.ai.persistence as persistence_module
    if not Path(persistence_module.__file__).resolve().is_relative_to(args.source_root):
        raise RuntimeError('linktools.ai was not imported from the requested source checkout')
    counters = defaultdict(Counter)
    phase = 'imports_and_setup'
    durations = {}
    snapshots = {}

    def statement(_connection: Connection, _cursor: object, sql: str, _parameters: object,
                  _context: object, executemany: bool) -> None:
        operation = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else 'EMPTY'
        counter = counters[phase]
        counter['statement_calls'] += 1
        counter[operation] += 1
        if operation in {'INSERT', 'UPDATE', 'DELETE', 'REPLACE'}:
            counter['DML'] += 1
        if executemany:
            counter['executemany_calls'] += 1

    def begin(_connection: Connection) -> None:
        counters[phase]['transaction_begin_events'] += 1
    def commit(_connection: Connection) -> None:
        counters[phase]['commit_events'] += 1
    def rollback(_connection: Connection) -> None:
        counters[phase]['rollback_events'] += 1
    listeners = [('before_cursor_execute', statement), ('begin', begin), ('commit', commit), ('rollback', rollback)]
    for name, callback in listeners:
        event.listen(Engine, name, callback)

    async def run_phase(name: str, operation: Callable[[], Awaitable[_Result]], *,
                        sample: bool = False) -> _Result:
        nonlocal phase
        phase = name
        start = perf_counter()
        try:
            return await operation()
        finally:
            durations[name] = perf_counter() - start
            if sample:
                snapshots[name] = storage_snapshot(args.database)
            phase = 'between_phases'

    namespace, tenant = 'records-measure', 'tenant'
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    description = 'output contract ' * 64
    schema = {'type': 'string', 'description': description}
    agent_binding = AgentBindingContract(
        agent_spec=AgentSpec('agent', model='default', tool_retries=3, output_retries=3),
        model_contract={'provider': 'test', 'model': 'fixture'}, selected=(), subagents=(),
        output_mode='text', output_schema=schema)
    task_binding = TaskBindingContract('task', 1, 'none', schema, None, 3, 0)
    inputs = {'copies_per_kind': args.copies, 'schema': schema, 'prompt': 'prompt',
              'now': now.isoformat(), 'hot_cas_per_execution': 2,
              'views_per_execution': 2, 'cold_full_reads_per_execution': 1,
              'capture_declaration': {'instructions': description}}
    def execution(kind: str, index: int) -> ExecutionRecord:
        identity = f'{kind}-{index:03d}'
        return ExecutionRecord(execution_id=identity, session_id=None, parent_execution_id=None,
            root_execution_id=identity, previous_execution_id=None, fork_base_execution_id=None,
            lineage_kind=ExecutionLineageKind.RUN, status=ExecutionStatus.STARTED,
            revision=1, event_seq=0, agent_run_seq=1 if kind == 'agent' else 0,
            error_code=None, safe_error_details={}, created_at=now, updated_at=now,
            mode='run' if kind == 'agent' else None, planning=False if kind == 'agent' else None,
            thinking=False if kind == 'agent' else None,
            binding=agent_binding if kind == 'agent' else task_binding,
            principal_id='principal', principal_kind='service',
            stored_user_input=StoredUserInput('text', StoredPayload.inline_text('prompt')))
    values = {kind: [execution(kind, index) for index in range(args.copies)] for kind in ('agent', 'task')}
    class Runner:
        pass
    task = Task.from_runner('capture.custom', Runner(), contract={
        'version': 1, 'type': 'custom', 'effect_policy': 'none', 'output_contract': {'kind': 'json'},
        'reconcile': False, 'config': {'instructions': description}})
    principal = service_principal(tenant, 'capture-owner')
    graphs = [TaskGraph(f'graph-{index:03d}', (TaskNode('task', task=task),)) for index in range(args.copies)]
    admissions = [TaskGraphAdmission.from_request(TaskGraphRequest(graph, principal, graph.graph_id, TaskGraphLimits())) for graph in graphs]
    inputs['binding_wire_sha256'] = {kind: hashlib.sha256(canonical_json_bytes(encode_domain(binding))).hexdigest()
                                   for kind, binding in (('agent', agent_binding), ('task', task_binding))}
    inputs['binding_wire_bytes'] = {kind: len(canonical_json_bytes(encode_domain(binding)))
                                  for kind, binding in (('agent', agent_binding), ('task', task_binding))}
    inputs['admission_digests'] = [admission.initial_request_digest for admission in admissions]
    args.database.parent.mkdir(parents=True, exist_ok=True)
    configuration = PersistenceStorage.sqlite(args.database)
    state = Persistence(configuration)
    try:
        await run_phase('initialize', lambda: state.initialize(namespace=namespace, tenant_id=tenant), sample=True)
        repository = state.execution.executions
        async def create(kind: str) -> None:
            for value in values[kind]:
                assert await repository.create_execution(value) == value
        for kind in ('agent', 'task'):
            await run_phase(kind + '_create', lambda kind=kind: create(kind), sample=True)
        captures = TaskGraphBindingCaptureStore(namespace, state.task.tasks.state_store)
        async def capture() -> None:
            for admission, graph in zip(admissions, graphs, strict=True):
                result = await captures.capture(admission, graph, tasks=(task,))
                assert result.tasks[(task.id, task.revision)]['config']['instructions'] == description
        await run_phase('capture_create', capture, sample=True)
        async def cas(kind: str) -> None:
            for index, value in enumerate(values[kind]):
                for _ in range(2):
                    changed = replace(value, revision=value.revision + 1, event_seq=value.event_seq + 1)
                    assert await repository.compare_and_swap(value.execution_id, tenant_id=tenant,
                        expected_revision=value.revision, next_record=changed) == changed
                    value = changed
                values[kind][index] = value
        async def views(kind: str) -> None:
            for value in values[kind]:
                for _ in range(2):
                    pair = await repository.get_view_fresh(value.execution_id, tenant_id=tenant)
                    assert pair[1].binding_kind == kind and pair[1].event_seq == value.event_seq
        for kind in ('agent', 'task'):
            await run_phase(kind + '_hot_cas', lambda kind=kind: cas(kind), sample=True)
            await run_phase(kind + '_warm_view', lambda kind=kind: views(kind))
        await run_phase('close_warm', state.close, sample=True)
        state = Persistence(PersistenceStorage.sqlite(args.database))
        await run_phase('reopen', lambda: state.initialize(namespace=namespace, tenant_id=tenant))
        repository = state.execution.executions
        # Views run before full reads, so they cannot benefit from body hydration.
        for kind in ('agent', 'task'):
            await run_phase(kind + '_cold_view', lambda kind=kind: views(kind))
        async def full_reads(kind: str) -> None:
            for value in values[kind]:
                assert await repository.get_fresh(value.execution_id, tenant_id=tenant) == value
        for kind in ('agent', 'task'):
            await run_phase(kind + '_cold_full_read', lambda kind=kind: full_reads(kind))
        captures = TaskGraphBindingCaptureStore(namespace, state.task.tasks.state_store)
        async def capture_reads() -> None:
            for admission in admissions:
                result = await captures.load(admission)
                assert result.tasks[(task.id, task.revision)]['config']['instructions'] == description
        await run_phase('capture_cold_read', capture_reads)
        await run_phase('close_cold', state.close, sample=True)
    finally:
        phase = 'cleanup'
        await state.close()
        for name, callback in listeners:
            event.remove(Engine, name, callback)
    return {
        'format': 'pr308-records-measure-v1', 'label': args.label, 'source': source,
        'module_files': {'linktools.ai.persistence': str(Path(persistence_module.__file__).resolve())},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': sys.version, 'platform': platform.platform(),
        'dependencies': dict(sorted((dist.metadata['Name'], dist.version) for dist in importlib.metadata.distributions()
                                    if dist.metadata.get('Name'))),
        'input': inputs, 'input_sha256': hashlib.sha256(canonical_json_bytes(inputs)).hexdigest(),
        'sql_by_phase': {name: dict(counter) for name, counter in counters.items()},
        'seconds_by_phase': durations, 'storage_by_phase': snapshots,
        'notes': [
            'Fixed synthetic persistence workload: 20 Agent + 20 Task executions and 20 Task graph binding captures by default; no models/tools execute.',
            'Initialization and reopen are separate. SQL counters are SQLAlchemy events; BEGIN statements and begin events are different counters.',
            'Column-value byte sums cover every SQL column including generated values. Numeric values use SQLite CAST-to-BLOB text lengths, not physical cell encoding.',
            'dbstat reports actual SQLite table/index B-tree pages. File sizes include WAL/SHM separately; no VACUUM or manual checkpoint.',
            'No elapsed-time claim from a single run; no network/disk-IO or non-SQLite conclusion.',
            'Fresh-read means new repository after reopening, not clearing the OS filesystem cache; content reads may warm during the phase.',
        ],
    }


if __name__ == '__main__':
    args, source = args_and_source()
    result = asyncio.run(workload(args, source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    final = result['storage_by_phase']['close_cold']
    print(json.dumps({'label': args.label, 'source': source, 'input_sha256': result['input_sha256'],
                      'output': str(args.output), 'records': final['tables']['ai_state_records'],
                      'files': final['files']}, ensure_ascii=False, indent=2))
