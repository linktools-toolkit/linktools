import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(process.argv[2],'utf8');
const start=source.indexOf('async function refreshNativeRequests(');
const end=source.indexOf('\nasync function loadDetail(',start);
const modelKey=item=>`${item.execution_id}:${item.agent_run_seq}:${item.model_request_seq}`;
const item=(id,sequence)=>({execution_id:id,agent_run_seq:1,model_request_seq:sequence});
async function exercise(items,behavior=()=>{}){
  const calls=[], inspector={hidden:false};
  const view={id:'a',items:[],pending:new Map(items.map(value=>[modelKey(value),{item:value,error:null}])),loading:false};
  const state={nativeTimeline:view,generation:1,selectedExecution:'a',view:'executions',tab:'models'};
  const context={URLSearchParams,Map,Error,state,$:()=>inspector,modelKey,enc:encodeURIComponent,
    renderNativeViews(){},scheduleNativeRefresh(){},acceptLiveResponse(){},
    mergeNativeItems:(old,rows)=>[...old,...rows],
    api:async path=>{calls.push(path);await behavior({path,calls,state,inspector,view});
      const url=new URL(path,'http://localhost');
      return {items:url.searchParams.getAll('identity').map(value=>item(decodeURIComponent(url.pathname.split('/')[3]),Number(value.split(':')[1])))};
    }};
  vm.createContext(context);vm.runInContext(source.slice(start,end),context);
  await context.refreshNativeRequests(view,1);
  return {calls,view};
}
let result=await exercise([item('a',1),item('a',2),item('b',1)]);
assert.equal(result.calls.length,2);assert.equal(result.view.pending.size,0);
assert.ok(result.calls.every(path=>path.includes('/models-exact?')));
assert.equal(new URL(result.calls[0],'http://localhost').searchParams.getAll('identity').length,2);
result=await exercise(Array.from({length:201},(_,index)=>item('a',index+1)));
assert.equal(result.calls.length,2);assert.equal(result.view.pending.size,0);
result=await exercise([item('a',1),item('a',2)],({path})=>{
  if(new URL(path,'http://localhost').searchParams.getAll('identity').includes('1:2'))throw new Error('missing');
});
assert.equal(result.calls.length,3);assert.equal(result.view.pending.size,1);
assert.equal(result.view.items[0].model_request_seq,1);
result=await exercise([item('a',1),item('a',2)],({inspector})=>{inspector.hidden=true;throw new Error('closed');});
assert.equal(result.calls.length,1);assert.equal(result.view.pending.size,2);
result=await exercise([item('a',1),item('a',2)],({state})=>{state.generation++;});
assert.equal(result.calls.length,1);assert.equal(result.view.items.length,0);
console.log('Exact batch UI: 3 individual requests -> 2 execution batches; chunking, isolation, dismissal and stale response passed');
