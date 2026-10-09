// Lightweight DOM interaction contract tests. This is not a browser/layout test.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const root=process.argv[2], html=readFileSync(resolve(root,'index.html'),'utf8');
class Element {
  constructor(tag='div') {
    this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.hidden=false;this.disabled=false;
    this.value='';this.checked=false;this.open=false;this.className='';this.scrollTop=0;this.clientHeight=300;this._text='';
    this.classList={toggle:(name,on)=>{const names=new Set(this.className.split(' ').filter(Boolean));if(on??!names.has(name))names.add(name);else names.delete(name);this.className=[...names].join(' ');},remove:name=>this.classList.toggle(name,false),add:name=>this.classList.toggle(name,true)};
  }
  get id(){return this._id;}
  set id(value){this._id=value;nodes.set(value,this);}
  get textContent(){return this._text+this.children.map(child=>child.textContent || '').join('');}
  set textContent(value){this._text=String(value);this.children=[];}
  get scrollHeight(){return this.children.length*80;}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this._text='';this.children=[...children];}
  focus(){}
  showModal(){this.open=true;}
  close(){this.open=false;}
  click(){if(!this.disabled)return this.onclick?.({preventDefault(){},target:this});}
  requestSubmit(){return this.onsubmit?.({preventDefault(){},submitter:new Element('button')});}
}
const nodes=new Map(),all=[];
for(const match of html.matchAll(/<([a-z][a-z0-9-]*)\b([^>]*)>/gi)){
  const node=new Element(match[1]);
  for(const attribute of match[2].matchAll(/([a-z][a-z0-9-]*)(?:="([^"]*)")?/gi)){
    const [,name,value='']=attribute;
    if(name==='id'){node.id=value;nodes.set(value,node);}
    else if(name.startsWith('data-'))node.dataset[name.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())]=value;
    else if(name==='value')node.value=value;
    else if(name==='class')node.className=value;
    else if(name==='hidden')node.hidden=true;
  }
  all.push(node);
}
const handlers=new Map();
globalThis.document={getElementById:id=>nodes.get(id),createElement:tag=>new Element(tag),querySelectorAll:selector=>all.filter(node=>Object.hasOwn(node.dataset,selector.slice(6,-1)))};
globalThis.window={addEventListener:(name,handler)=>{if(!handlers.has(name))handlers.set(name,[]);handlers.get(name).push(handler);}};
let hash='';globalThis.location={get hash(){return hash;},set hash(value){hash=value;queueMicrotask(()=>handlers.get('hashchange')?.forEach(handler=>handler()));}};
let counter=0;Object.defineProperty(globalThis,"crypto",{value:{randomUUID:()=>`request-${++counter}`},configurable:true});globalThis.confirm=()=>true;globalThis.prompt=()=> 'Retry prompt';
const calls=[],delays=new Map(),timelineOverrides=new Map(),streamBlocks=new Map(),detailResponses=new Map(),unavailableTimelines=new Set();
const info=(id,session_id=id[0])=>({execution_id:id,agent_id:'default',session_id,status:'SUCCEEDED',binding_kind:'agent',lineage_kind:'ROOT',created_at:'2026-01-01T00:00:00Z',started_at:'2026-01-01T00:00:00Z',terminal_at:'2026-01-01T00:00:01Z'});
const sessions=new Map(['a','b'].map(id=>[id,{session_id:id,agent_id:'default',status:'OPEN',revision:0,history_quality:'complete',metadata:{title:id==='a'?'Alpha':'Beta'}}]));
const executions=new Map([['a-run',info('a-run')],['b-run',info('b-run')]]);
const timeline=id=>timelineOverrides.get(id) || ({items:[{execution_id:`${id}-run`,status:'SUCCEEDED',created_at:'2026-01-01T00:00:00Z',user_input:`${id} question`,conversation_committed:true,items:[{item_kind:'assistant',content:`${id} answer`}]}],next_cursor:null});
function response(value,status=200){return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});}
function deferred(path){let release;const promise=new Promise(resolve=>{release=resolve;});delays.set(path,promise);return value=>{delays.delete(path);release(response(value));};}
let forkAttempts=0;
globalThis.fetch=async(path,options={})=>{
  const url=new URL(path,'http://127.0.0.1:8765'),key=url.pathname==='/api/session'?'/api/sessions/'+url.searchParams.get('session_id'):url.pathname.startsWith('/api/session/')?'/api/sessions/'+url.searchParams.get('session_id')+url.pathname.slice('/api/session'.length):url.pathname,method=options.method || 'GET',body=options.body?JSON.parse(options.body):null;
  calls.push({key,method,body,query:Object.fromEntries(url.searchParams)});
  if(delays.has(method+' '+key))return delays.get(method+' '+key);
  if(key==='/api/config')return response({asset_root:'/workspace/.linktools',read_only:false,memory_scope:'default',capabilities:[{kind:'agent',id:'default',revision:1}],metric_names:[]});
  if(key==='/api/sessions'&&method==='GET')return response({items:[...sessions.values()],next_cursor:null});
  if(key==='/api/executions')return response({items:[...executions.values()],next_cursor:null,recent_scan:url.searchParams.get('recent')==='true'});
  if(detailResponses.has(key))return response(detailResponses.get(key));
  if(key==='/api/metrics')return response({items:[]});
  if(url.pathname==='/api/session'){const id=url.searchParams.get('session_id');if(url.searchParams.get('include_timeline')==='false')return response({session:sessions.get(id),timeline:null});if(unavailableTimelines.has(id))return response({code:'SESSION_HISTORY_UNAVAILABLE'},503);return response({session:sessions.get(id),timeline:timeline(id)});}
  const execution=key.match(/^\/api\/executions\/([^/]+)$/);
  if(execution)return response(executions.get(execution[1]) || info(execution[1],null));
  if(key.endsWith('/events') && streamBlocks.has(key.split('/')[3]))return new Response(new ReadableStream({start(controller){streamBlocks.get(key.split('/')[3]).controller=controller;},cancel(){}}));
  if(key.endsWith('/events'))return new Response('event: snapshot\ndata: '+JSON.stringify(executions.get(key.split('/')[3]))+'\n\n');
  if(key.endsWith('/history'))return response({items:[{execution_id:key.split('/')[3],message_seq:1,item_kind:'assistant',content:'History marker'}],next_cursor:null});
  if(key.endsWith('/models')||key.endsWith('/trace')||key.endsWith('/transcript'))return response({items:[],next_cursor:null});
  if(key.endsWith('/fork')&&key.startsWith('/api/sessions/')){
    forkAttempts++;
    if(forkAttempts===1)return response({code:'STORAGE_COMMIT_UNKNOWN'},503);
    sessions.set(body.new_session_id,{...sessions.get('a'),session_id:body.new_session_id});
    return response({session_id:body.new_session_id});
  }
  throw new Error(`Unexpected request ${method} ${key}`);
};
const core=readFileSync(resolve(root,'console.js'),'utf8');
const coreURL='data:text/javascript;base64,'+Buffer.from(core).toString('base64');
const source=readFileSync(resolve(root,'app.js'),'utf8').replace("'./console.js'",JSON.stringify(coreURL));
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const tick=()=>new Promise(resolve=>setTimeout(resolve,10));
const settle=async()=>{for(let i=0;i<5;i++)await tick();};
const node=id=>nodes.get(id),tab=name=>all.find(element=>element.dataset.tab===name),view=name=>all.find(element=>element.dataset.view===name);
await settle();
assert.equal(node('list').children.length,2);
location.hash='#session=a';await settle();
assert.equal(node('conversation-title').textContent,'Alpha');
assert.match(node('conversation').textContent,/a answer/);

// A session response cannot overwrite a newer navigation.
const releaseA=deferred('GET /api/sessions/a');
location.hash='#session=a';await tick();location.hash='#session=b';await settle();
releaseA({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.equal(node('conversation-title').textContent,'Beta');

// Changing inspector tabs invalidates the previous request.
const releaseModels=deferred('GET /api/executions/b-run/models');
tab('models').click();await tick();tab('history').click();await settle();
releaseModels({items:[{model_request_seq:999,status:'SUCCEEDED',model:{},execution_id:'b-run'}],next_cursor:null});await settle();
assert.match(node('inspector-content').textContent,/History marker/);
assert.doesNotMatch(node('inspector-content').textContent,/999/);

// A pending execution action cannot navigate away from a newer selection.
location.hash='#session=a';await settle();
const releaseRetry=deferred('POST /api/executions/a-run/retry');
node('retry').click();await tick();location.hash='#session=b';await settle();
releaseRetry({execution_id:'a-retried'});await settle();
assert.equal(location.hash,'#session=b');
assert.equal(node('conversation-title').textContent,'Beta');

// While metadata for a newly selected execution is loading, old controls are disabled.
const releaseExecution=deferred('GET /api/executions/a-run');
location.hash='#session=a';await settle();
assert.equal(node('retry').disabled,true);
releaseExecution(info('a-run'));await settle();assert.equal(node('retry').disabled,false);

// An uncertain fork preserves both destination identity and idempotency key.
await node('fork-session').click();await settle();
await node('fork-session').click();await settle();
const forks=calls.filter(call=>call.key==='/api/sessions/a/fork');
assert.equal(forks.length,2);
assert.equal(forks[0].body.new_session_id,forks[1].body.new_session_id);
assert.equal(forks[0].body.request_id,forks[1].body.request_id);

// Dismissing an in-flight creation cannot close or navigate a newer dialog.
const releaseCreate=deferred('POST /api/sessions');
node('new-session').click();node('new-title').value='<img src=x onerror=alert(1)>';
node('new-agent').value='default';node('new-form').requestSubmit();await tick();
const oldSession=node('new-dialog').dataset.sessionId;
node('new-dialog').close();node('new-session').click();const newerSession=node('new-dialog').dataset.sessionId;
const priorHash=location.hash;releaseCreate({session_id:oldSession});await settle();
assert.equal(node('new-dialog').open,true);assert.equal(node('new-dialog').dataset.sessionId,newerSession);assert.equal(location.hash,priorHash);
node('new-dialog').close();

// Repeated submit during one in-flight request invokes start only once.
location.hash='#session=a';await settle();
const releaseSend=deferred('POST /api/sessions/a/messages');node('prompt').value='Repeated message';
node('composer').requestSubmit();node('composer').requestSubmit();await tick();
assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').length,1);
node('prompt').value='Next draft';releaseSend({execution_id:'a-run'});await settle();assert.equal(node('prompt').value,'Next draft');

// Refreshing an old turn must not clear a new live turn selected in the same session.
executions.set('a-new',{...info('a-new'),status:'RUNNING'});
const older=timeline('a').items[0];
timelineOverrides.set('a',{items:[older,{...older,execution_id:'a-new',user_input:'new question',conversation_committed:false,items:[]}],next_cursor:null});
location.hash='#session=a&execution=a-run';await settle();
const releaseRefresh=deferred('GET /api/sessions/a');node('refresh').click();await tick();
streamBlocks.set('a-new',{});
const newTurn=node('conversation').children.find(child=>child.textContent.includes('new question'));
await newTurn.children[0].children.find(child=>child.tagName==='BUTTON').click();await settle();
const stream=streamBlocks.get('a-new');
stream.controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:'a-new',agent_id:'default',depth:0,event:{event_type:'ASSISTANT_TEXT_DELTA',payload:{text:'New execution text'}}},cursor:null})+'\n\n'));
await settle();assert.match(node('live').textContent,/New execution text/);
releaseRefresh({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.match(node('live').textContent,/New execution text/);
stream.controller.close();streamBlocks.delete('a-new');timelineOverrides.delete('a');
executions.set('a-new',{...info('a-new'),status:'SUCCEEDED'});
await settle();

// A late session read cannot re-open the conversation pane over Metrics.
const releaseAgain=deferred('GET /api/sessions/a');location.hash='#session=a';await tick();view('metrics').click();await settle();
releaseAgain({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.equal(node('metrics-view').hidden,false);assert.equal(node('conversation-view').hidden,true);
assert.equal(node('metric-name').tagName,'INPUT');
executions.set('orphan',{...info('orphan',null),status:'RECOVERY_REQUIRED'});location.hash='#execution=orphan';await settle();
assert.equal(node('composer').hidden,true);assert.equal(node('stop-run').hidden,false);assert.ok(node('live'));

// Exact-ID lookup preserves opaque sessions and distinguishes standalone executions.
const opaque=' ../team/conversation?notes#你好 ';
sessions.set(opaque,{...sessions.get('a'),session_id:opaque,metadata:{title:'Opaque conversation'}});
timelineOverrides.set(opaque,{items:[],next_cursor:null});
node('open-kind').value='session';node('open-id').value=opaque;node('open-form').requestSubmit();await settle();
assert.equal(new URLSearchParams(location.hash.slice(1)).get('session'),opaque);
assert.equal(node('conversation-title').textContent,'Opaque conversation');
node('open-kind').value='execution';node('open-id').value='b-run';node('open-form').requestSubmit();await settle();
assert.equal(location.hash,'#execution=b-run');assert.equal(node('composer').hidden,true);

// Former history/trace summaries remain human-readable without an eager trace fetch.
const beforeTrace=calls.filter(call=>call.key.endsWith('/trace')).length;
detailResponses.set('/api/executions/b-run/models',{items:[{execution_id:'b-run',agent_run_seq:1,depth:0,model_request_seq:1,purpose:'agent',status:'SUCCEEDED',model:{model_name:'Recorded model'},duration_ns:1000,usage:{input_tokens:10,output_tokens:4,cache_read_tokens:3,cache_write_tokens:2},request:{messages:[{parts:[{part_kind:'system-prompt',content:'real system prompt'}]}],parameters:{instruction_parts:[{name:'workspace',content:'fixed',dynamic:false}],output_mode:'text'}},content_included:true}],next_cursor:null});
tab('models').click();await settle();
assert.match(node('inspector-content').textContent,/Recorded model · 1.000 us/);
assert.match(node('inspector-content').textContent,/3 cache read \/ 2 cache write/);
assert.match(node('inspector-content').textContent,/Prompt architecture/);
assert.match(node('inspector-content').textContent,/System Prompt1 part\(s\) · ~18 chars/);
assert.match(node('inspector-content').textContent,/Fixed Instructions \(F0\/F1\)/);
assert.equal(calls.filter(call=>call.key.endsWith('/trace')).length,beforeTrace);
detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'b-run',step_event_seq:1,payload:{kind:'MODEL_RESPONSE',status:'SUCCEEDED',scope:'root',agent_run_seq:1,step_index:0,model_request_seq:1,purpose:'agent',duration_ns:2000000,token_usage:{input_tokens:10,output_tokens:4}}}],next_cursor:null});
tab('trace').click();await settle();
assert.match(node('inspector-content').textContent,/request #1/);
assert.match(node('inspector-content').textContent,/agent · 2.000 ms · 10 in \/ 4 out/);

// Exact newest ordering is an explicit scan action, never a refresh side effect.
view('executions').click();await settle();
const scans=()=>calls.filter(call=>call.query.recent==='true').length;
assert.equal(scans(),0);
node('list-action').value='recent';node('list-action').onchange();
assert.equal(node('filter-session').disabled,true);
node('filter-form').requestSubmit();node('filter-form').requestSubmit();await settle();assert.equal(scans(),1);
assert.match(node('list-scope').textContent,/scanned visible execution metadata/);
node('refresh').click();await settle();assert.equal(scans(),1);
assert.equal(node('list-action').value,'paged');assert.equal(node('filter-session').disabled,false);
node('filter-session').value=opaque;node('filter-form').requestSubmit();await settle();
assert.equal(calls.filter(call=>call.key==='/api/executions').at(-1).query.session_id,opaque);
node('settings').click();assert.match(node('settings-content').textContent,/Asset root\/workspace\/\.linktools/);node('settings-dialog').close();

// Unavailable history cannot hide readable metadata or make refresh show stale values.
unavailableTimelines.add('a');sessions.set('a',{...sessions.get('a'),revision:3,cwd:'old/path',history_quality:'partial',active_execution_id:null});
node('open-kind').value='session';node('open-id').value='a';node('open-form').requestSubmit();await settle();
assert.equal(node('conversation-title').textContent,'Alpha');
assert.match(node('session-details').textContent,/"revision": 3/);
assert.match(node('conversation').textContent,/SESSION_HISTORY_UNAVAILABLE/);
assert.doesNotMatch(node('conversation').textContent,/Ready when you are/);
assert.ok(calls.some(call=>call.key==='/api/sessions/a' && call.query.include_timeline==='false'));
sessions.set('a',{...sessions.get('a'),revision:4,cwd:'new/path',active_execution_id:'other-run'});
node('refresh').click();await settle();
assert.match(node('session-details').textContent,/"revision": 4/);
assert.match(node('session-details').textContent,/new\/path/);
assert.match(node('session-details').textContent,/other-run/);
assert.doesNotMatch(node('session-details').textContent,/old\/path/);

// Refresh reconciles controls even when an empty session has no execution.
sessions.set(opaque,{...sessions.get(opaque),status:'OPEN',active_execution_id:null});
node('open-kind').value='session';node('open-id').value=opaque;node('open-form').requestSubmit();await settle();
assert.equal(node('send').disabled,false);
sessions.set(opaque,{...sessions.get(opaque),status:'CLOSED'});
node('refresh').click();await settle();
assert.match(node('conversation-meta').textContent,/CLOSED/);
assert.equal(node('send').disabled,true);assert.equal(node('prompt').disabled,true);
console.log('DOM contracts passed: stale navigation/detail/action, disabled stale controls, uncertain fork, interrupted dialog, repeated submit, metrics navigation');
