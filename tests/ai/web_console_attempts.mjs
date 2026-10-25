import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(process.argv[2],'utf8');
const {updateStreamAttempts,streamAttemptIdentity,eventKey}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const requests=new Map();
const event=(event_type,stream_attempt_seq=1,extra={},execution_id='root')=>({execution_id,event:{event_type,payload:{agent_run_seq:1,model_request_seq:1,stream_attempt_seq,...extra}}});
const send=(...args)=>updateStreamAttempts(requests,event(...args));
const request=(run=1,seq=1,execution='root')=>requests.get(`${execution}:${run}:${seq}`);
const current=()=>request().attempts.get(request().current);

send('ASSISTANT_TEXT_DELTA',1,{text:'partial'});
assert.equal(current().status,'RUNNING');assert.equal(current().text,'partial');
send('MODEL_ATTEMPT_STARTED');
send('MODEL_ATTEMPT_STARTED');
assert.equal(current().text,'partial','replayed boundary cannot erase text');
send('ASSISTANT_THINKING_DELTA',1,{text:'thought'});
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:2,text:'other request'});
send('ASSISTANT_TEXT_DELTA',1,{agent_run_seq:2,text:'later run'});
send('ASSISTANT_TEXT_DELTA',1,{text:'other agent'},'child');
send('MODEL_ATTEMPT_FAILED',1,{status:'FAILED',retry_scheduled:true});
assert.equal(current().status,'FAILED');assert.equal(current().text,'partial');
send('MODEL_ATTEMPT_STARTED',2,{replaces_stream_attempt_seq:1});
assert.equal(current().text,'');assert.equal(current().thinking,'');
assert.equal(request(1,2).attempts.get(1).text,'other request');
assert.equal(request(2).attempts.get(1).text,'later run');
assert.equal(request(1,1,'child').attempts.get(1).text,'other agent');
send('ASSISTANT_TEXT_DELTA',2,{text:'second'});
send('MODEL_ATTEMPT_STARTED',1);
send('ASSISTANT_TEXT_DELTA',1,{text:'stale'});
send('MODEL_ATTEMPT_FAILED',1,{status:'FAILED'});
send('MODEL_REQUEST_FINISHED',1,{status:'SUCCEEDED'});
assert.equal(current().seq,2);assert.equal(current().text,'second');assert.equal(current().status,'RUNNING');
assert.equal(request().attempts.get(1).text,'partial');assert.equal(request().attempts.get(1).status,'FAILED');
send('MODEL_ATTEMPT_STARTED',3,{replaces_stream_attempt_seq:2});
assert.equal(request().attempts.get(2).status,'FAILED','replacement itself retires old active attempt');
send('ASSISTANT_TEXT_DELTA',3,{text:'winner'});
send('MODEL_REQUEST_FINISHED',3,{status:'SUCCEEDED',message_seq:8});
send('MODEL_ATTEMPT_FAILED',3,{status:'FAILED'});
send('MODEL_ATTEMPT_STARTED',3);
send('ASSISTANT_TEXT_DELTA',3,{text:'late'});
assert.equal(current().status,'SUCCEEDED');assert.equal(current().text,'winner');

send('MODEL_ATTEMPT_STARTED',1,{model_request_seq:3});
send('MODEL_ATTEMPT_FAILED',1,{model_request_seq:3,status:'CANCELLED',retry_scheduled:false});
send('MODEL_ATTEMPT_STARTED',1,{model_request_seq:3});
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:3,text:'after cancel'});
assert.equal(request(1,3).attempts.get(1).status,'CANCELLED');assert.equal(request(1,3).attempts.get(1).text,'');
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:4,text:'truncated'});
updateStreamAttempts(requests,{execution_id:'root',event:{event_type:'STREAM_RESET',payload:{}}});
assert.equal(request(1,4).attempts.get(1).text,'');assert.equal(request(1,4).current,1);
assert.equal(request().attempts.get(1).text,'partial','cursor gaps do not erase failed attempt diagnostics');
assert.equal(request(1,1,'child').attempts.get(1).text,'other agent');
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:4,text:'retained tail'});
assert.equal(request(1,4).attempts.get(1).truncated,true);
updateStreamAttempts(requests,{execution_id:'root',event:{event_type:'EXECUTION_FAILED'}});
assert.equal(request(1,4).attempts.get(1).status,'FAILED');assert.equal(current().status,'SUCCEEDED');
assert.equal(request(1,1,'child').attempts.get(1).status,'RUNNING');

for(const payload of [{text:'legacy'},{text:'partial identity',stream_attempt_seq:2},{agent_run_seq:1,model_request_seq:1,stream_attempt_seq:0}]){
  const item={execution_id:'root',event:{event_type:'ASSISTANT_TEXT_DELTA',payload}};
  assert.equal(streamAttemptIdentity(item),null);assert.equal(updateStreamAttempts(requests,item),false);
}
for(const kind of ['MODEL_ATTEMPT_STARTED','MODEL_ATTEMPT_FAILED','ASSISTANT_PART_COMPLETED']) {
  const first=event(kind,1,{part_index:0}),second=event(kind,2,{part_index:0});
  assert.notEqual(eventKey(first),eventKey(second));
  assert.equal(eventKey(first),eventKey({...first,event:{...first.event,durable_seq:9}}));
  assert.notEqual(eventKey(first),eventKey(event(kind,1,{part_index:0,model_request_seq:2})));
}
console.log('Stream attempts isolate requests, retain failed partials and ignore stale boundaries');
