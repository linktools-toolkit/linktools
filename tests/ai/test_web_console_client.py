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
const {modelKey,historyKey,eventKey,upsertModel,mergePage,parseSSE,readSSE,metricValue,duration,modelLabel,usageLabel,promptLayers}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
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
for(const event_type of ['MODEL_REQUEST_STARTED','MODEL_REQUEST_FINISHED','TOOL_CALL_STARTED','TOOL_CALL_FINISHED','ASSISTANT_PART_COMPLETED']){
  const payload={agent_run_seq:1,model_request_seq:2,call_id:'tool:1',message_seq:3,part_index:0};
  const live={execution_id:'root',event:{event_type,durable_seq:null,payload}};
  const replay={...live,event:{...live.event,durable_seq:9}};
  assert.equal(eventKey(live),eventKey(replay));
  assert.notEqual(eventKey(live),eventKey({...live,execution_id:'child'}));
  assert.notEqual(eventKey(live),eventKey({...live,event:{...live.event,payload:{...payload,agent_run_seq:2}}}));
  const distinct={...payload,model_request_seq:3,call_id:'tool:2',part_index:1};
  assert.notEqual(eventKey(live),eventKey({...live,event:{...live.event,payload:distinct}}));
}
assert.notEqual(eventKey({execution_id:'root',event:{event_type:'EXECUTION_RESUMED',durable_seq:1}}),eventKey({execution_id:'root',event:{event_type:'EXECUTION_RESUMED',durable_seq:2}}));
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
assert.equal(duration(999),'999 ns');
assert.equal(duration(1000),'1.000 us');
assert.equal(duration(2000000),'2.000 ms');
assert.equal(metricValue({metric:'duration',unit:'ns'},{value:1000}),'1.000 us');
assert.equal(modelLabel({model_name:'specific',route_id:'route',name:'name',model:'model'}),'specific');
assert.equal(modelLabel({route_id:'route'}),'route');
assert.equal(modelLabel({name:'recorded-model'}),'recorded-model');
assert.equal(usageLabel({input_tokens:10,output_tokens:4,cache_read_tokens:3,cache_write_tokens:2}),'10 in / 4 out · 3 cache read / 2 cache write');
const layers=Object.fromEntries(promptLayers({
 instructions:['duplicate instruction mirror'],
 messages:[{parts:[{part_kind:'system-prompt',content:'real system prompt'},{part_kind:'user-prompt',content:['hello',{media_type:'image/png',size:2048,digest:'a'.repeat(64)}]}]}],
 parameters:{instruction_parts:[{content:'fixed workspace guidance',name:'workspace',dynamic:false},{content:'repository overlay',name:'repository',dynamic:true}],function_tools:[{name:'read_file'}],native_tools:[],revealed_tool_names:['read_file'],deferred_capability_ids:[],output_mode:'text',allow_text_output:true,allow_image_output:false}
}));
assert.equal(Object.keys(layers).length,7);
assert.equal(layers['System Prompt'],'1 part(s) · ~18 chars');
assert.match(layers['Fixed Instructions (F0/F1)'],/sources: workspace/);
assert.match(layers['Dynamic Instructions (O)'],/sources: repository/);
assert.equal(layers['Conversation Context'],'1 message(s) · 1 part(s) · user-prompt=1');
assert.equal(layers['Input Attachments'],'1 attachment · image/png');
assert.equal(layers['Tool Contract'],'1 function · 0 native · 1 revealed · 0 deferred capabilities');
assert.equal(layers['Model Output Contract'],'mode=text · text output=yes · image output=no');
assert.doesNotMatch(JSON.stringify(layers),/duplicate instruction mirror|real system prompt|fixed workspace guidance|repository overlay/);
assert.equal(Object.fromEntries(promptLayers({messages:[{parts:[{kind:'system-prompt',content:'你好🌱'}]}]}))['System Prompt'],'1 part(s) · ~3 chars');

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
    for mode in ("writable", "readonly", "workspace", "races"):
        result = subprocess.run([node, str(script), str(assets), mode], capture_output=True, text=True, timeout=20, check=False)
        assert result.returncode == 0, result.stderr


def test_web_messages_render_safe_markdown_and_stored_input() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the optional Web client tests")
    assets = Path(__file__).parents[2] / "linktools-ai/src/linktools/ai/web/assets"
    script = Path(__file__).with_name("web_console_messages.mjs")
    result = subprocess.run([node, str(script), str(assets)], capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
