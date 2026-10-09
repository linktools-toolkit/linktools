#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic Web client stream and projection invariants, without a model."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_web_client_preserves_identity_and_reads_fragmented_streams() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the optional Web client tests")
    source = Path(__file__).parents[2] / "linktools-ai/src/linktools/ai/web/assets/console.js"
    script = r'''
import {readFileSync} from 'node:fs';
import assert from 'node:assert/strict';
const source=readFileSync(process.argv[1], 'utf8');
const {modelKey,historyKey,upsertModel,mergePage,parseSSE,readSSE,metricValue}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const items=new Map();
const base={execution_id:'root',agent_run_seq:1,model_request_seq:1};
upsertModel(items,{...base,status:'SUCCEEDED',usage:{input_tokens:2}});
upsertModel(items,{...base,status:'RUNNING',usage:null});
upsertModel(items,{...base,execution_id:'child',status:'RUNNING'});
upsertModel(items,{...base,agent_run_seq:2,status:'RUNNING'});
assert.equal(items.size,3);
assert.equal(items.get(modelKey(base)).usage.input_tokens,2);
const original=[{id:'a',value:1},{id:'b',value:2}];
assert.deepEqual(mergePage(original,[{id:'a',value:3},{id:'c',value:4}],x=>x.id),[{id:'a',value:3},{id:'b',value:2},{id:'c',value:4}]);
assert.notEqual(historyKey({...base,message_seq:1,part_index:0,item_kind:'tool_call'}),historyKey({...base,execution_id:'child',message_seq:1,part_index:0,item_kind:'tool_call'}));
assert.equal(parseSSE(': heartbeat'),null);
const input='id: cursor\r\ndata: {"text":"你好🌱"}\r\n\r\nevent: snapshot\r\ndata: {"status":"SUCCEEDED"}\r\n\r\n';
const bytes=new TextEncoder().encode(input), observed=[];
const stream=new ReadableStream({start(controller){for(const byte of bytes)controller.enqueue(new Uint8Array([byte]));controller.close();}});
await readSSE(new Response(stream),item=>observed.push(item),new AbortController().signal);
assert.equal(observed.length,2);
assert.equal(observed[0].data.text,'你好🌱');
assert.equal(observed[0].id,'cursor');
assert.equal(observed[1].event,'snapshot');
assert.equal(metricValue({metric:'linktools.execution.failure_ratio',unit:'ratio'},{value:0.125}),'12.5%');
assert.equal(metricValue({metric:'duration',unit:'ns'},{value:null}),'—');
'''
    result = subprocess.run([node, "--input-type=module", "-e", script, str(source)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_web_dom_interactions_ignore_stale_work_and_preserve_mutation_identity() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the optional Web client tests")
    root = Path(__file__).parents[2]
    script = Path(__file__).with_name("web_console_dom.mjs")
    assets = root / "linktools-ai/src/linktools/ai/web/assets"
    result = subprocess.run([node, str(script), str(assets)], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
