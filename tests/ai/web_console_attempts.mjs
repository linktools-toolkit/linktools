import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(process.argv[2],'utf8');
const {updateStreamAttempts,streamAttemptIdentity,eventKey,streamResponsePreview,streamAttemptText}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
if(process.argv[3]) {
  const requests=new Map(),emptyStart=process.argv[5]==='empty-start';
  for(const item of JSON.parse(readFileSync(process.argv[3],'utf8'))) {
    updateStreamAttempts(requests,item);
    if(emptyStart && item.event.payload.part_start===true && item.event.payload.text==='')
      assert.equal(streamResponsePreview([...requests.values()][0]).text,'','empty starts are consumed before later text');
  }
  assert.equal(requests.size,1);
  const request=[...requests.values()][0];
  assert.deepEqual([...request.attempts.values()].map(attempt=>attempt.status),emptyStart?['SUCCEEDED']:['FAILED','SUCCEEDED','SUCCEEDED']);
  assert.equal(request.status,'SUCCEEDED');
  assert.equal(streamResponsePreview(request).text,process.argv[4]);
  if(!emptyStart)assert.equal(streamAttemptText(request.attempts.get(1)),'discarded');
  assert.equal(streamResponsePreview(request).text.includes('discarded'),false);
  console.log('Real Runtime watch events preserve SDK continuation and retry projection');process.exit(0);
}
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
send('MODEL_ATTEMPT_FINISHED',1,{status:'FAILED',retry_scheduled:true});
assert.equal(current().status,'FAILED');assert.equal(current().text,'partial');
send('MODEL_ATTEMPT_STARTED',2,{replaces_stream_attempt_seq:1});
assert.equal(current().text,'');assert.equal(current().thinking,'');
assert.equal(request(1,2).attempts.get(1).text,'other request');
assert.equal(request(2).attempts.get(1).text,'later run');
assert.equal(request(1,1,'child').attempts.get(1).text,'other agent');
send('ASSISTANT_TEXT_DELTA',2,{text:'second'});
send('MODEL_ATTEMPT_STARTED',1);
send('ASSISTANT_TEXT_DELTA',1,{text:'stale'});
send('MODEL_ATTEMPT_FINISHED',1,{status:'FAILED'});
send('MODEL_REQUEST_FINISHED',1,{status:'SUCCEEDED'});
assert.equal(current().seq,2);assert.equal(current().text,'second');assert.equal(current().status,'RUNNING');
assert.equal(request().attempts.get(1).text,'partial');assert.equal(request().attempts.get(1).status,'FAILED');
send('MODEL_ATTEMPT_FINISHED',2,{status:'FAILED',retry_scheduled:true});
send('MODEL_ATTEMPT_STARTED',3,{replaces_stream_attempt_seq:2});
assert.equal(request().attempts.get(2).status,'FAILED');
send('ASSISTANT_TEXT_DELTA',3,{text:'winner'});
send('MODEL_ATTEMPT_FINISHED',3,{status:'SUCCEEDED'});
send('MODEL_ATTEMPT_STARTED',4,{replaces_stream_attempt_seq:null});
assert.equal(request().attempts.get(3).status,'SUCCEEDED');assert.equal(request().attempts.get(3).text,'winner');
send('ASSISTANT_TEXT_DELTA',4,{text:' continued'});
send('MODEL_ATTEMPT_FINISHED',4,{status:'SUCCEEDED'});
assert.equal(request().finished,false,'transport success is not logical request completion');
send('MODEL_REQUEST_FINISHED',4,{status:'SUCCEEDED',message_seq:8});
send('MODEL_ATTEMPT_FINISHED',4,{status:'FAILED'});
send('MODEL_ATTEMPT_STARTED',4);
send('ASSISTANT_TEXT_DELTA',4,{text:'late'});
assert.equal(current().status,'SUCCEEDED');assert.equal(current().text,' continued');
assert.equal(request().attempts.get(3).text,'winner');assert.equal(request().status,'SUCCEEDED');

send('MODEL_ATTEMPT_STARTED',1,{model_request_seq:3});
send('MODEL_ATTEMPT_FINISHED',1,{model_request_seq:3,status:'CANCELLED',retry_scheduled:false});
send('MODEL_ATTEMPT_STARTED',1,{model_request_seq:3});
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:3,text:'after cancel'});
assert.equal(request(1,3).attempts.get(1).status,'CANCELLED');assert.equal(request(1,3).attempts.get(1).text,'');
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:4,text:'truncated'});
updateStreamAttempts(requests,{execution_id:'root',event:{event_type:'STREAM_RESET',payload:{}}});
assert.equal(request(1,4).attempts.get(1).text,'');assert.equal(request(1,4).current,1);
assert.equal(request().attempts.get(1).text,'partial','cursor gaps do not erase failed attempt diagnostics');
assert.equal(streamResponsePreview(request()).text,'winner continued','a completed logical response is not cleared by another request gap');
assert.equal(request(1,1,'child').attempts.get(1).text,'other agent');
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:4,text:'retained tail'});
assert.equal(request(1,4).attempts.get(1).truncated,true);
send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:5,text:'completed segment'});
send('MODEL_ATTEMPT_FINISHED',1,{model_request_seq:5,status:'SUCCEEDED'});
updateStreamAttempts(requests,{execution_id:'root',event:{event_type:'EXECUTION_FAILED'}});
assert.equal(request(1,4).attempts.get(1).status,'FAILED');assert.equal(current().status,'SUCCEEDED');
assert.equal(request(1,5).attempts.get(1).status,'SUCCEEDED');assert.equal(request(1,5).status,'FAILED');
assert.equal(request(1,1,'child').attempts.get(1).status,'RUNNING');

const part=(attempt,index,text,start=true)=>send('ASSISTANT_TEXT_DELTA',attempt,{model_request_seq:6,part_index:index,part_start:start,text});
const preview=()=>streamResponsePreview(request(1,6)).text;
part(1,0,'old prefix ');part(1,1,'keep other part');
send('MODEL_ATTEMPT_FINISHED',1,{model_request_seq:6,status:'SUCCEEDED'});
send('MODEL_ATTEMPT_STARTED',2,{model_request_seq:6,replaces_stream_attempt_seq:null});
part(2,0,'new prefix ');part(2,2,' extra');
assert.equal(preview(),'new prefix keep other part extra','overlapping parts do not replace the whole earlier attempt');
assert.equal(request(1,6).attempts.get(1).status,'SUCCEEDED');
part(2,0,'continued ',false);assert.equal(preview(),'new prefix continued keep other part extra');
send('MODEL_ATTEMPT_FINISHED',2,{model_request_seq:6,status:'FAILED',retry_scheduled:true});
assert.equal(preview(),'old prefix keep other part');
assert.equal(streamAttemptText(request(1,6).attempts.get(2)),'new prefix continued  extra');
send('MODEL_ATTEMPT_STARTED',3,{model_request_seq:6,replaces_stream_attempt_seq:2});
part(3,0,'final prefix ');assert.equal(preview(),'final prefix keep other part');
part(3,0,'');assert.equal(preview(),'keep other part','an empty start clears precisely that part');
send('MODEL_ATTEMPT_FINISHED',3,{model_request_seq:6,status:'SUCCEEDED'});
send('MODEL_REQUEST_FINISHED',3,{model_request_seq:6,status:'SUCCEEDED'});

send('ASSISTANT_TEXT_DELTA',1,{model_request_seq:7,part_index:0,part_start:true,text:'old snapshot'});
send('MODEL_ATTEMPT_FINISHED',1,{model_request_seq:7,status:'SUCCEEDED'});
send('MODEL_ATTEMPT_STARTED',2,{model_request_seq:7,replaces_stream_attempt_seq:null});
updateStreamAttempts(requests,{execution_id:'root',event:{event_type:'STREAM_RESET',payload:{}}});
send('ASSISTANT_TEXT_DELTA',2,{model_request_seq:7,part_index:0,part_start:false,text:'tail after missing start'});
assert.equal(streamResponsePreview(request(1,7)).text,'tail after missing start','an unseen part reset cannot splice a retained successful snapshot to a disconnected tail');
assert.equal(streamResponsePreview(request(1,7)).truncated,true);
assert.equal(request(1,7).attempts.get(1).status,'SUCCEEDED','a gap is not an attempt failure');
assert.equal(preview(),'keep other part','completed requests remain intact');
assert.equal(request(1,1,'child').attempts.get(1).text,'other agent');
send('MODEL_ATTEMPT_FINISHED',2,{model_request_seq:7,status:'FAILED'});
assert.equal(streamResponsePreview(request(1,7)).text,'','failed tails do not restore stale successful snapshot content');
assert.equal(streamAttemptText(request(1,7).attempts.get(2)),'tail after missing start');

for(const payload of [{text:'legacy'},{text:'partial identity',stream_attempt_seq:2},{agent_run_seq:1,model_request_seq:1,stream_attempt_seq:0}]){
  const item={execution_id:'root',event:{event_type:'ASSISTANT_TEXT_DELTA',payload}};
  assert.equal(streamAttemptIdentity(item),null);assert.equal(updateStreamAttempts(requests,item),false);
}
for(const kind of ['MODEL_ATTEMPT_STARTED','MODEL_ATTEMPT_FINISHED','ASSISTANT_PART_COMPLETED']) {
  const first=event(kind,1,{part_index:0}),second=event(kind,2,{part_index:0});
  assert.notEqual(eventKey(first),eventKey(second));
  assert.equal(eventKey(first),eventKey({...first,event:{...first.event,durable_seq:9}}));
  assert.notEqual(eventKey(first),eventKey(event(kind,1,{part_index:0,model_request_seq:2})));
}
console.log('Stream attempts isolate requests, retain failed partials and ignore stale boundaries');
