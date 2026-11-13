#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated SQLite effect-ledger ordering-index comparison."""

import hashlib
import json
from sqlalchemy import create_engine, insert, select
from linktools.ai.persistence._schema import build_runtime_sql_metadata
from linktools.ai.persistence._plan import PersistenceDomain

def main() -> None:
    m = build_runtime_sql_metadata(frozenset({PersistenceDomain.CHECKPOINT}))
    e = create_engine('sqlite://')
    m.create_all(e)
    t = m.tables['ai_step_tool_operations']
    rows = []
    for tenant in range(4):
        for run in range(128):
            scope = f'{tenant}/run/{run}'.encode()
            parent = f'{tenant}/execution/{run // 4}'.encode()
            for call in range(128):
                key = f'{tenant}/{run:04}/{call:04}'.encode()
                rows.append(dict(store_key=str(tenant).encode(), key=key, scope_key=scope, subject_id=str(call).encode(), parent_key=parent, sort_key=key, state='completed', storage_revision=1, lease_owner=None, lease_fence=1, lease_expires_at=None, payload_json='{}', key_locator=hashlib.sha256(key).hexdigest(), scope_locator=hashlib.sha256(scope).hexdigest(), parent_locator=hashlib.sha256(parent).hexdigest(), sort_prefix=key[:128]))
    with e.begin() as conn:
        conn.execute(insert(t), rows)
    c = e.raw_connection()
    cur = c.cursor()
    for columns in ('store_key_sort_prefix', 'scope_locator_sort_prefix'):
        names = 'store_key,sort_prefix' if columns.startswith('store') else 'scope_locator,sort_prefix'
        cur.execute('CREATE INDEX ix_ai_step_tool_operations_' + columns + ' ON ai_step_tool_operations (' + names + ')')
    cur.execute('ANALYZE')
    scope = b'0/run/64'
    parent = b'0/execution/16'
    queries = {'run_list': ('SELECT * FROM ai_step_tool_operations WHERE store_key=? AND scope_locator=? AND scope_key=? ORDER BY sort_prefix,sort_key,key', (b'0', hashlib.sha256(scope).hexdigest(), scope)), 'execution_list': ('SELECT * FROM ai_step_tool_operations WHERE store_key=? AND parent_locator=? AND parent_key=? ORDER BY sort_prefix,sort_key,key', (b'0', hashlib.sha256(parent).hexdigest(), parent)), 'exact_call': ('SELECT * FROM ai_step_tool_operations WHERE store_key=? AND scope_locator=? AND scope_key=? AND subject_id=?', (b'0', hashlib.sha256(scope).hexdigest(), scope, b'64')), 'partition_page128': ('SELECT * FROM ai_step_tool_operations WHERE store_key=? ORDER BY sort_prefix,sort_key,key LIMIT 128', (b'0',))}
    for phase in ('before', 'after'):
        out = {}
        for name, (sql, args) in queries.items():
            count = [0]

            def tick() -> int:
                count[0] += 1
                return 0
            c.set_progress_handler(tick, 100)
            n = len(cur.execute(sql, args).fetchall())
            c.set_progress_handler(None, 0)
            out[name] = {'rows': n, 'vm_steps_approx': count[0] * 100, 'plan': [x[3] for x in cur.execute('EXPLAIN QUERY PLAN ' + sql, args)]}
        print(phase, json.dumps(out))
        if phase == 'before':
            for name in ('ix_ai_step_tool_operations_store_key_sort_prefix', 'ix_ai_step_tool_operations_scope_locator_sort_prefix'):
                cur.execute('DROP INDEX ' + name)
            cur.execute('ANALYZE')
if __name__ == '__main__':
    main()
