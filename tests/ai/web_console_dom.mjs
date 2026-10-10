// Lightweight DOM interaction contract tests. This is not a browser/layout test.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const readOnly=process.argv[3]==='readonly';
const root=process.argv[2], html=readFileSync(resolve(root,'index.html'),'utf8');
class Element {
  constructor(tag='div') {
    this.tagName=tag.toUpperCase();this.children=[];this.dataset={};this.hidden=false;this.disabled=false;
    this.isConnected=false;this.value='';this.checked=false;this.open=false;this.className='';this.scrollTop=0;this.clientHeight=300;this._text='';
    this.classList={toggle:(name,on)=>{const names=new Set(this.className.split(' ').filter(Boolean));if(on??!names.has(name))names.add(name);else names.delete(name);this.className=[...names].join(' ');},remove:name=>this.classList.toggle(name,false),add:name=>this.classList.toggle(name,true)};
  }
  get id(){return this._id;}
  set id(value){this._id=value;nodes.set(value,this);}
  get textContent(){return this._text+this.children.map(child=>child.textContent || '').join('');}
  set textContent(value){this.replaceChildren();this._text=String(value);}
  get scrollHeight(){return this.children.length*80;}
  connect(value){this.isConnected=value;if(this.id){if(value)nodes.set(this.id,this);else if(nodes.get(this.id)===this)nodes.delete(this.id);}this.children.forEach(child=>child.connect?.(value));if(!value && document.activeElement===this)document.activeElement=null;}
  append(...children){this.children.push(...children);children.forEach(child=>{child.parentElement=this;child.connect?.(this.isConnected);});}
  replaceChildren(...children){this.children.forEach(child=>child.connect?.(false));this._text='';this.children=[];this.append(...children);}
  focus(){document.activeElement=this;}
  scrollIntoView(){}
  setAttribute(name,value){this.attributes ??= {};this.attributes[name]=String(value);}
  getAttribute(name){return this.attributes?.[name];}
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
    else if(name.startsWith('aria-'))node.setAttribute(name,value);
    else if(name==='class')node.className=value;
    else if(name==='hidden')node.hidden=true;
  }
  node.isConnected=true;all.push(node);
}
const handlers=new Map();
globalThis.document={getElementById:id=>nodes.get(id)?.isConnected?nodes.get(id):null,createElement:tag=>new Element(tag),querySelectorAll:selector=>all.filter(node=>Object.hasOwn(node.dataset,selector.slice(6,-1)))};
globalThis.window={addEventListener:(name,handler)=>{if(!handlers.has(name))handlers.set(name,[]);handlers.get(name).push(handler);}};
let hash='';globalThis.location={get hash(){return hash;},set hash(value){hash=value;queueMicrotask(()=>handlers.get('hashchange')?.forEach(handler=>handler()));}};
let counter=0;Object.defineProperty(globalThis,"crypto",{value:{randomUUID:()=>`request-${++counter}`},configurable:true});globalThis.confirm=()=>true;globalThis.prompt=()=> 'Retry prompt';
const calls=[],delays=new Map(),timelineOverrides=new Map(),streamBlocks=new Map(),detailResponses=new Map(),unavailableTimelines=new Set();
const info=(id,session_id=id[0])=>({execution_id:id,agent_id:'default',session_id,status:'SUCCEEDED',binding_kind:'agent',lineage_kind:'ROOT',created_at:'2026-01-01T00:00:00Z',started_at:'2026-01-01T00:00:00Z',terminal_at:'2026-01-01T00:00:01Z'});
const sessions=new Map(['a','b'].map(id=>[id,{session_id:id,agent_id:'default',status:'OPEN',revision:0,history_quality:'complete',metadata:{title:id==='a'?'Alpha':'Beta'}}]));
const executions=new Map([['a-run',info('a-run')],['b-run',info('b-run')]]);
const timeline=id=>timelineOverrides.get(id) || ({items:[{execution_id:`${id}-run`,status:'SUCCEEDED',created_at:'2026-01-01T00:00:00Z',user_input:`${id} question`,conversation_committed:true,items:[{item_kind:'assistant',content:`${id} answer`}]}],next_cursor:null});
function response(value,status=200){return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});}
function deferred(path){let release,reject;const promise=new Promise((resolve,fail)=>{release=resolve;reject=fail;});delays.set(path,promise);const resolve=(value,status=200)=>{delays.delete(path);release(response(value,status));};resolve.reject=error=>{delays.delete(path);reject(error);};return resolve;}
let forkAttempts=0,cancelAttempts=0,endMode=null;
globalThis.fetch=async(path,options={})=>{
  const url=new URL(path,'http://127.0.0.1:8765'),key=url.pathname==='/api/session'?'/api/sessions/'+url.searchParams.get('session_id'):url.pathname.startsWith('/api/session/')?'/api/sessions/'+url.searchParams.get('session_id')+url.pathname.slice('/api/session'.length):url.pathname,method=options.method || 'GET',body=options.body?JSON.parse(options.body):null;
  calls.push({key,method,body,query:Object.fromEntries(url.searchParams)});
  if(delays.has(method+' '+key))return delays.get(method+' '+key);
  if(key==='/api/config')return response({asset_root:'/workspace/.linktools',read_only:readOnly,memory_scope:'default',capabilities:[{kind:'agent',id:'default',revision:1}],metric_names:[]});
  if(key==='/api/sessions'&&method==='GET')return response({items:[...sessions.values()],next_cursor:null});
  if(key==='/api/executions' && endMode?.listError)throw new Error('list unavailable');
  if(key==='/api/executions')return response({items:[...executions.values()],next_cursor:null,recent_scan:url.searchParams.get('recent')==='true'});
  if(detailResponses.has(key))return response(detailResponses.get(key));
  if(key==='/api/metrics')return response({items:[]});
  if(url.pathname==='/api/session'){const id=url.searchParams.get('session_id');if(url.searchParams.get('include_timeline')==='false')return response({session:sessions.get(id),timeline:null});if(unavailableTimelines.has(id))return response({code:'SESSION_HISTORY_UNAVAILABLE'},503);return response({session:sessions.get(id),timeline:timeline(id)});}
  const execution=key.match(/^\/api\/executions\/([^/]+)$/);
  if(execution)return response(executions.get(execution[1]) || info(execution[1],null));
  if(key.endsWith('/events') && streamBlocks.has(key.split('/')[3]))return new Response(new ReadableStream({start(controller){streamBlocks.get(key.split('/')[3]).controller=controller;},cancel(){}}));
  if(key.endsWith('/events'))return new Response('event: snapshot\ndata: '+JSON.stringify(executions.get(key.split('/')[3]))+'\n\n');
  if(key.endsWith('/history'))return response({items:[{execution_id:key.split('/')[3],message_seq:1,item_kind:'assistant',content:'History marker'}],next_cursor:null});
  if(key.endsWith('/models')||key.endsWith('/trace')||key.endsWith('/transcript')||key.endsWith('/recovery'))return response({items:[],next_cursor:null});
  if(endMode && key==='/api/executions/orphan/cancel'){
    if(endMode.cancelError){const error=endMode.cancelError;endMode.cancelError=null;if(endMode.commitCancel)executions.get('orphan').status='CANCELLING';throw new Error(error);}
    if(endMode.blockCancel)return response({code:'STORAGE_CONFLICT'},409);
    endMode.cancelAccepted=true;
    executions.get('orphan').status=endMode.keepRecovery?'RECOVERY_REQUIRED':endMode.directTerminal?'CANCELLED':'CANCELLING';
    if(endMode.directTerminal)sessions.get('a').active_execution_id=null;
    return response({execution_id:'orphan',cancelled:Boolean(endMode.directTerminal)});
  }
  if(endMode && key==='/api/executions/orphan/recover'){
    assert.ok(executions.get('orphan').status==='CANCELLING' || executions.get('orphan').status==='RECOVERY_REQUIRED' && endMode.cancelAccepted,'must prove durable cancellation before recovery');
    if(endMode.recoverError){const error=endMode.recoverError;endMode.recoverError=null;throw new Error(error);}
    if(endMode.effectError)return response({code:'TOOL_EFFECT_OUTCOME_UNKNOWN',safe_details:{operation_id:'effect'}},409);
    if(!endMode.keepCancelling)executions.get('orphan').status='CANCELLED';
    if(!endMode.keepOwner)sessions.get('a').active_execution_id=endMode.nextOwner || null;
    return response({execution_id:'orphan'},202);
  }
  if(key==='/api/executions/orphan/cancel'){
    cancelAttempts++;
    if(cancelAttempts===1)return response({code:'STORAGE_CONFLICT'},409);
    executions.set('orphan',{...executions.get('orphan'),status:'CANCELLING'});
    return response({execution_id:'orphan',cancelled:false});
  }
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
const node=id=>nodes.get(id),tab=name=>({click(){if(node('inspector').hidden)node('inspector-toggle').click();return all.find(element=>element.dataset.tab===name).click();}}),view=name=>all.find(element=>element.dataset.view===name);
await settle();
assert.equal(node('list').children.length,2);
location.hash='#session=a';await settle();
assert.equal(node('conversation-title').textContent,'Alpha');
assert.match(node('conversation').textContent,/a answer/);

// Progressive disclosure keeps explicit controls, focus, and authoritative ownership.
assert.equal(node('inspector').hidden,true);assert.equal(node('current-status').textContent,'Session available');
assert.equal(node('inspector-toggle').getAttribute('aria-expanded'),'false');
const beforeDetails=calls.length;node('inspector-toggle').click();
assert.equal(node('inspector').hidden,false);assert.equal(document.activeElement,node('inspector-close'));
assert.equal(node('inspector-toggle').getAttribute('aria-expanded'),'true');
node('inspector-close').click();assert.equal(document.activeElement,node('inspector-toggle'));
assert.equal(node('inspector').hidden,true);assert.equal(calls.length,beforeDetails);
node('action-menu').open=true;node('action-menu').onkeydown({key:'Escape'});
assert.equal(node('action-menu').open,false);assert.equal(document.activeElement,node('actions-toggle'));
node('options-toggle').click();assert.equal(node('composer-options').hidden,false);
assert.equal(node('options-toggle').getAttribute('aria-expanded'),'true');node('options-toggle').click();
for(const id of ['planning','thinking','memory','files','rename-session','fork-session','close-session','retry','fork-run','end-stopped','export','detail-more','turns-more','list-more'])assert.ok(node(id),`missing retained control ${id}`);
assert.match(html,/<details id="action-menu"[\s\S]*?id="rename-session"[\s\S]*?id="fork-run"[\s\S]*?<\/details>/);
assert.match(html,/<details id="open-tools"[\s\S]*?id="open-record"[\s\S]*?<\/details>/);
assert.deepEqual(all.filter(item=>item.dataset.tab).map(item=>item.dataset.tab),['overview','history','transcript','models','trace','recovery']);
if(readOnly){
  for(const id of ['send','prompt','planning','thinking','memory','files','new-session','retry','fork-run','end-stopped','rename-session','fork-session','close-session'])assert.equal(node(id).disabled,true,id);
  assert.equal(node('connection').hidden,false);assert.equal(node('connection').textContent,'Read-only');
  node('inspector-toggle').click();assert.equal(node('inspector').hidden,false);assert.equal(node('export').disabled,false);
  tab('history').click();await settle();assert.match(node('inspector-content').textContent,/History marker/);
  tab('recovery').click();await settle();assert.match(node('inspector-content').textContent,/require an execution Runtime/);
  node('open-tools').open=true;node('open-kind').value='execution';node('open-id').value='b-run';node('open-form').requestSubmit();await settle();
  assert.equal(node('inspector').hidden,false);assert.equal(node('open-record').disabled,false);
  assert.equal(calls.filter(call=>call.method==='POST').length,0);
  console.log('Read-only disclosure contracts passed');process.exit(0);
}
sessions.get('a').active_execution_id='active-other';executions.set('active-other',{...info('active-other','a'),status:'STARTED'});
location.hash='#session=a&execution=a-run';await settle();
assert.equal(node('current-status').textContent,'Session occupied');assert.equal(node('show-active').hidden,false);
assert.doesNotMatch(node('current-status').textContent,/SUCCEEDED/);
node('show-active').click();await settle();assert.match(node('current-status').textContent,/Session occupied · STARTED/);
assert.equal(node('inspector').hidden,false);assert.equal(node('cancel').hidden,false);
node('inspector-close').click();assert.equal(node('cancel').hidden,false);
sessions.get('a').active_execution_id=null;location.hash='#session=a';await settle();

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

// A child trace read cannot apply its selectors after the user changes selection.
detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'trace-child',step_event_seq:1,payload:{agent_run_seq:7,model_request_seq:9}}],next_cursor:null});
tab('trace').click();await settle();
const readChild=node('inspector-content').children[0].children.find(child=>child.tagName==='BUTTON' && child.textContent==='Read content');
const releaseTraceChild=deferred('GET /api/executions/trace-child');
readChild.click();await tick();location.hash='#session=a';await settle();tab('models').click();await settle();
const afterTraceNavigation=calls.length;
releaseTraceChild(info('trace-child','b'));await settle();
assert.equal(location.hash,'#session=a');
assert.match(all.find(element=>element.dataset.tab==='models').className,/\bactive\b/);
assert.ok(!calls.slice(afterTraceNavigation).some(call=>call.key==='/api/executions/a-run/history'));
detailResponses.delete('/api/executions/b-run/trace');

for(const destination of ['tab','execution','current']){
  location.hash='#session=b';await settle();
  detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'trace-child',step_event_seq:1,payload:{agent_run_seq:7,model_request_seq:9}}],next_cursor:null});
  tab('trace').click();await settle();
  const read=node('inspector-content').children[0].children.find(child=>child.tagName==='BUTTON' && child.textContent==='Read content');
  const release=deferred('GET /api/executions/trace-child');read.click();await tick();
  if(destination==='tab')tab('models').click();
  if(destination==='execution')location.hash='#session=b&execution=b-run';
  await settle();const beforeReadback=calls.length;release(info('trace-child','b'));await settle();
  const content=calls.slice(beforeReadback).filter(call=>call.key.endsWith('/history'));
  if(destination==='current'){
    assert.equal(content.length,1);assert.equal(content[0].key,'/api/executions/trace-child/history');
    assert.equal(content[0].query.agent_run_seq,'7');assert.equal(content[0].query.model_request_seq,'9');
  }else{
    assert.equal(content.length,0,destination);
    if(destination==='tab')assert.match(all.find(element=>element.dataset.tab==='models').className,/\bactive\b/);
    else assert.match(node('execution-meta').textContent,/b-run/);
  }
  detailResponses.delete('/api/executions/b-run/trace');
}

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
assert.equal(node('inspector').hidden,true,'Send must preserve the conversation-only view');
for(const open of [false,true]){
  if(node('inspector').hidden===open)node('inspector-toggle').click();
  const renamed=deferred('POST /api/sessions/a/update');node('rename-session').click();await tick();renamed({session_id:'a'});await settle();
  assert.equal(node('inspector').hidden,!open,'Rename must preserve the panel choice');
}
node('inspector-close').click();

// Known admission rejection gets a new key only on the next explicit send.
for(const code of ['SESSION_BUSY','SESSION_CONFLICT']) {
  node('prompt').value=`Rejected ${code}`;
  const rejected=deferred('POST /api/sessions/a/messages');
  node('composer').requestSubmit();await tick();
  rejected({code,operation_id:'web',safe_details:{reason:'<script>safe detail</script>'},exception_message:'private diagnostic'},409);await settle();
  const first=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  assert.equal(node('prompt').value,`Rejected ${code}`);
  assert.match(node('notice').textContent,/this message was not started/);
  assert.match(node('notice').textContent,/POST \/api\/session\/messages/);
  assert.match(node('notice').textContent,/<script>safe detail<\/script>/);
  assert.doesNotMatch(node('notice').textContent,/private diagnostic/);
  assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1),first);
  const retried=deferred('POST /api/sessions/a/messages');
  node('composer').requestSubmit();await tick();
  const next=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  assert.notEqual(next.body.request_id,first.body.request_id);
  assert.equal(next.body.prompt,first.body.prompt);
  retried({execution_id:'a-run'});await settle();
}

// Unknown commit and network outcomes preserve the identity of the original send.
for(const failure of ['STORAGE_COMMIT_UNKNOWN','network']) {
  node('prompt').value=`Uncertain ${failure}`;
  const uncertain=deferred('POST /api/sessions/a/messages');
  node('composer').requestSubmit();await tick();
  const first=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  if(failure==='network')uncertain.reject(new TypeError('Network unavailable'));
  else uncertain({code:failure},503);
  await settle();
  const repeated=deferred('POST /api/sessions/a/messages');
  node('composer').requestSubmit();await tick();
  assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1).body.request_id,first.body.request_id);
  repeated({execution_id:'a-run'});await settle();
}

// A rejected send cannot place its error in a newer conversation.
node('prompt').value='Old selection draft';
const oldSend=deferred('POST /api/sessions/a/messages');
node('composer').requestSubmit();await tick();location.hash='#session=b';await settle();
oldSend({code:'SESSION_BUSY'},409);await settle();
assert.equal(node('conversation-title').textContent,'Beta');assert.equal(node('notice').textContent,'');
location.hash='#session=a';await settle();

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
const liveEvent=(event_type,seq,depth=0,payload={})=>stream.controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:depth?'child':'a-new',agent_id:'default',depth,event:{event_type,durable_seq:seq,payload}},cursor:null})+'\n\n'));
liveEvent('TOOL_CALL_STARTED',1);await settle();assert.equal(node('live-activity').open,false);
node('live-activity').open=true;node('live-activity').ontoggle();node('live-activity').children[0].focus();liveEvent('MODEL_REQUEST_STARTED',2);await settle();assert.equal(node('live-activity').open,true);assert.equal(document.activeElement,node('live-activity').children[0]);
liveEvent('EXECUTION_FAILED',3,1);
liveEvent('MODEL_REQUEST_FINISHED',4,0,{agent_run_seq:1,model_request_seq:1,status:'FAILED'});
liveEvent('TOOL_CALL_FINISHED',5,0,{agent_run_seq:1,call_id:'failed-call',tool_name:'query',status:'FAILED',error_code:'TOOL_RETRY_REQUIRED'});
liveEvent('MODEL_REQUEST_FINISHED',6,0,{agent_run_seq:1,model_request_seq:2,status:'CANCELLED'});
liveEvent('TOOL_CALL_FINISHED',7,0,{agent_run_seq:1,call_id:'successful-call',tool_name:'query',status:'SUCCEEDED'});
for(const [index,event] of ['EXECUTION_START_UNKNOWN','EXECUTION_RECOVERY_REQUIRED','CANCEL_REQUESTED','EXECUTION_CANCELLED'].entries())liveEvent(event,8+index);
await settle();
for(const name of ['execution failed','model request finished · FAILED','tool call finished · FAILED · TOOL_RETRY_REQUIRED','model request finished · CANCELLED','execution start unknown','execution recovery required','cancel requested','execution cancelled'])assert.ok(node('live').children.some(child=>child.className.includes('live-event') && child.textContent.includes(name)),name);
assert.ok(node('live-activity').children.some(child=>child.className.includes('live-event') && child.textContent.includes('tool call finished · SUCCEEDED')));
assert.ok(!node('live').children.some(child=>child.className.includes('live-event') && child.textContent.includes('SUCCEEDED')));
const focusedChild=node('live').children.find(child=>child.className.includes('live-event') && child.textContent.includes('execution failed')).children.find(child=>child.tagName==='BUTTON');focusedChild.focus();liveEvent('TOOL_CALL_STARTED',12,0,{agent_run_seq:1,call_id:'next-call'});await settle();assert.equal(document.activeElement.dataset.liveFocus,focusedChild.dataset.liveFocus);
releaseRefresh({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.match(node('live').textContent,/New execution text/);assert.equal(node('live-activity').open,true);
const focusBeforeRefresh=document.activeElement.dataset.liveFocus;
node('refresh').click();await settle();assert.equal(node('live-activity').open,true);
assert.equal(document.activeElement.dataset.liveFocus,focusBeforeRefresh);
stream.controller.close();streamBlocks.delete('a-new');timelineOverrides.delete('a');
executions.set('a-new',{...info('a-new'),status:'SUCCEEDED'});
await settle();

// A late session read cannot re-open the conversation pane over Metrics.
const releaseAgain=deferred('GET /api/sessions/a');location.hash='#session=a';await tick();view('metrics').click();await settle();
releaseAgain({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.equal(node('metrics-view').hidden,false);assert.equal(node('conversation-view').hidden,true);
assert.equal(node('metric-name').tagName,'INPUT');
executions.set('orphan',{...info('orphan',null),status:'RECOVERY_REQUIRED'});location.hash='#execution=orphan';await settle();
assert.equal(node('composer').hidden,true);assert.equal(node('inspector').hidden,false);assert.equal(document.activeElement,node('inspector-close'));node('inspector-close').click();assert.equal(node('stop-run').hidden,false);assert.ok(node('live'));

// Status alone never initiates cleanup; one stopped-executor confirmation gates it.
const recoveries=()=>calls.filter(call=>call.key==='/api/executions/orphan/recover');
const cancellations=()=>calls.filter(call=>call.key==='/api/executions/orphan/cancel');
tab('recovery').click();await settle();
for(const status of ['START_UNKNOWN','WAITING_DEFERRED','WAITING_RETRY','FINALIZING','SUCCEEDED']){
  executions.set('orphan',{...info('orphan',null),status});node('refresh').click();await settle();
  assert.equal(node('resume-run').disabled,true,status);
}
for(const status of ['PENDING_START','STARTED','CANCELLING','RECOVERY_REQUIRED']){
  executions.set('orphan',{...info('orphan',null),status});node('refresh').click();await settle();
  assert.equal(node('end-stopped').disabled,false);
  assert.equal(node('resume-run').disabled,false,status);
  assert.equal(recoveries().length,0);assert.equal(cancellations().length,0);
}
let recoveryConfirmation='';
globalThis.confirm=message=>{recoveryConfirmation=message;return false;};
node('end-stopped').click();await settle();
assert.match(recoveryConfirmation,/previous executor has stopped/);assert.equal(recoveries().length,0);assert.equal(cancellations().length,0);
globalThis.confirm=()=>true;
const resetEnd=async(mode={},status='STARTED')=>{
  endMode=mode;sessions.get('a').active_execution_id='orphan';
  executions.set('orphan',{...info('orphan','a'),status});location.hash='#execution=orphan';await settle();
};
await resetEnd();
const releaseEndCancel=deferred('POST /api/executions/orphan/cancel');
node('end-stopped').click();node('end-stopped').click();await tick();
assert.equal(cancellations().length,1);assert.equal(node('end-stopped').disabled,true);
for(const id of ['stop-run','retry','fork-run'])assert.equal(node(id).disabled,true);
executions.get('orphan').status='CANCELLING';releaseEndCancel({execution_id:'orphan',cancelled:false});await settle();
assert.equal(recoveries().length,1);assert.equal(executions.get('orphan').status,'CANCELLED');
assert.equal(sessions.get('a').active_execution_id,null);assert.match(node('notice').textContent,/ended and session released/);
assert.equal(calls.filter(call=>call.key==='/api/sessions/a' && call.query.include_timeline==='false').length>0,true);

// Unknown cancellation stays on the same key; a committed cancellation skips resend.
for(const commitCancel of [false,true]){
  await resetEnd({cancelError:'network outcome unknown',commitCancel});
  const start=cancellations().length,recoverStart=recoveries().length;
  await node('end-stopped').click();await settle();
  assert.equal(recoveries().length,recoverStart);assert.match(node('notice').textContent,/network outcome unknown/);
  await node('end-stopped').click();await settle();
  const sent=cancellations().slice(start);assert.equal(sent.length,commitCancel?1:2);
  if(!commitCancel)assert.equal(sent[0].body.request_id,sent[1].body.request_id);
  assert.match(node('notice').textContent,/ended and session released/);
}
await resetEnd({keepRecovery:true},'RECOVERY_REQUIRED');const beforeRecoveryRequired=recoveries().length;
assert.equal(node('end-stopped').hidden,false);assert.match(node('end-stopped').textContent,/free session/);
await node('end-stopped').click();await settle();
assert.equal(recoveries().length,beforeRecoveryRequired+1);assert.match(node('notice').textContent,/ended and session released/);

// Interruption after durable cancel is safe: a fresh attempt reads CANCELLING first.
await resetEnd({},'CANCELLING');const beforeResumeCancel=cancellations().length;
await node('end-stopped').click();await settle();assert.equal(cancellations().length,beforeResumeCancel);
assert.match(node('notice').textContent,/ended and session released/);

await resetEnd({recoverError:'recovery reply lost'});const beforeUnknownRecovery=recoveries().length;
await node('end-stopped').click();await settle();assert.equal(recoveries().length,beforeUnknownRecovery+1);
assert.match(node('notice').textContent,/recovery reply lost/);
// If that request completed, the next explicit attempt must not recover again.
executions.get('orphan').status='CANCELLED';sessions.get('a').active_execution_id=null;
node('refresh').click();await settle();assert.equal(node('end-stopped').disabled,false);
await node('end-stopped').click();await settle();assert.equal(recoveries().length,beforeUnknownRecovery+1);
assert.match(node('notice').textContent,/ended and session released/);

// Uncertain recovery keeps its key, while Resume and End use separate operations.
await resetEnd({recoverError:'unknown end response'});
await node('end-stopped').click();await settle();const uncertainEnd=recoveries().at(-1);
await node('end-stopped').click();await settle();
assert.equal(recoveries().at(-1).body.request_id,uncertainEnd.body.request_id);
assert.match(node('notice').textContent,/ended and session released/);

await resetEnd({recoverError:'unknown end response'});tab('recovery').click();await settle();
await node('end-stopped').click();await settle();const endBeforeResume=recoveries().at(-1);
await node('resume-run').click();await settle();
assert.notEqual(recoveries().at(-1).body.request_id,endBeforeResume.body.request_id);

await resetEnd();tab('recovery').click();await settle();
const failedResume=deferred('POST /api/executions/orphan/recover');
node('resume-run').click();await tick();const resumeBeforeEnd=recoveries().at(-1);
failedResume.reject(new Error('unknown resume response'));await settle();
await node('end-stopped').click();await settle();
assert.notEqual(recoveries().at(-1).body.request_id,resumeBeforeEnd.body.request_id);
assert.match(node('notice').textContent,/ended and session released/);

for(const [mode,status,message] of [
  [{blockCancel:true},'RECOVERY_REQUIRED',/STORAGE_CONFLICT/],
  [{keepCancelling:true},'STARTED',/still CANCELLING/],
  [{keepOwner:true},'STARTED',/session release is not confirmed/],
  [{effectError:true},'STARTED',/TOOL_EFFECT_OUTCOME_UNKNOWN/],
]){
  await resetEnd(mode,status);const before=recoveries().length;
  await node('end-stopped').click();await settle();assert.match(node('notice').textContent,message);
  assert.doesNotMatch(node('notice').textContent,/ended and session released/);
  if(mode.blockCancel)assert.equal(recoveries().length,before);
}
await resetEnd({directTerminal:true},'PENDING_START');const beforeDirect=recoveries().length;
await node('end-stopped').click();await settle();assert.equal(recoveries().length,beforeDirect);
assert.match(node('notice').textContent,/ended and session released/);
await resetEnd({nextOwner:'a-new'});await node('end-stopped').click();await settle();
assert.match(node('notice').textContent,/another execution now owns/);

// Failed owner readback remains retryable after a terminal refresh, without controls.
await resetEnd();const failedOwnerRead=deferred('GET /api/sessions/a');
node('end-stopped').click();await tick();failedOwnerRead.reject(new Error('owner read unavailable'));await settle();
assert.match(node('notice').textContent,/owner read unavailable/);
node('refresh').click();await settle();assert.equal(node('end-stopped').disabled,false);
const beforeOwnerRetry=recoveries().length;await node('end-stopped').click();await settle();
assert.equal(recoveries().length,beforeOwnerRetry);assert.match(node('notice').textContent,/ended and session released/);

// Auxiliary rendering failures do not overturn confirmed terminal/owner facts.
await resetEnd();endMode.listError=true;await node('end-stopped').click();await settle();
assert.match(node('notice').textContent,/ended and session released.*Display refresh failed: list unavailable/);
assert.doesNotMatch(node('notice').textContent,/No action was automatically resent/);

// Navigation interrupts later control writes and discards stale action errors.
await resetEnd();const staleEnd=deferred('POST /api/executions/orphan/cancel'),beforeStale=recoveries().length;
node('end-stopped').click();await tick();location.hash='#session=b';await settle();const betaNotice=node('notice').textContent;
executions.get('orphan').status='CANCELLING';staleEnd({execution_id:'orphan',cancelled:false});await settle();
assert.equal(recoveries().length,beforeStale);assert.equal(node('notice').textContent,betaNotice);

// Original resume capability remains a secondary action in the Recovery tab.
await resetEnd({},'STARTED');endMode=null;tab('recovery').click();await settle();
assert.match(node('inspector-content').textContent,/Resume stopped execution/);
globalThis.confirm=message=>{recoveryConfirmation=message;return false;};
node('resume-run').click();await settle();assert.match(recoveryConfirmation,/may call models or tools/);
globalThis.confirm=()=>true;
const releaseRecovery=deferred('POST /api/executions/orphan/recover');const beforeLegacy=recoveries().length;
node('resume-run').click();node('resume-run').click();await tick();assert.equal(recoveries().length,beforeLegacy+1);
executions.set('orphan',{...info('orphan',null),status:'SUCCEEDED'});releaseRecovery({execution_id:'orphan'});await settle();
assert.equal(node('resume-run').disabled,true);tab('overview').click();await settle();

// A cancellation conflict reads state without resending; a new click reuses its key.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});node('refresh').click();await settle();
await node('stop-run').click();await settle();
assert.equal(cancelAttempts,1);assert.match(node('notice').textContent,/Runtime status re-read: STARTED/);
assert.match(node('notice').textContent,/If still needed, choose Stop execution again/);
await node('stop-run').click();await settle();
assert.equal(cancelAttempts,2);
const cancels=cancellations().slice(-2);
assert.equal(cancels[0].body.request_id,cancels[1].body.request_id);
assert.match(node('notice').textContent,/terminal outcome is not yet confirmed/);

// Stop feedback follows canonical terminal and session-owner readback, not the request reply.
executions.get('orphan').session_id='a';sessions.get('a').active_execution_id='orphan';
executions.get('orphan').status='CANCELLED';node('refresh').click();await settle();
assert.match(node('notice').textContent,/Cancellation confirmed.*release is not yet confirmed/);
sessions.get('a').active_execution_id='a-new';node('refresh').click();await settle();
assert.match(node('notice').textContent,/Cancellation confirmed.*Another execution now owns/);
assert.equal(node('stop-run').hidden,true);

// Cancellation is a normal turn outcome rather than a raw API error.
const cancelledTurn={...timeline('a').items[0],status:'CANCELLED',error_code:'EXECUTION_CANCELLED',safe_error_details:{}};
timelineOverrides.set('a',{items:[cancelledTurn],next_cursor:null});location.hash='#session=a';await settle();
assert.match(node('conversation').textContent,/Execution cancelled\./);assert.doesNotMatch(node('conversation').textContent,/EXECUTION_CANCELLED|\{\}/);
timelineOverrides.delete('a');location.hash='#execution=orphan';await settle();

// A terminal snapshot also reconciles a pending Stop notice without manual refresh.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});sessions.get('a').active_execution_id='orphan';
const stopStream={};streamBlocks.set('orphan',stopStream);node('refresh').click();await settle();
await node('stop-run').click();await settle();
executions.get('orphan').status='CANCELLED';sessions.get('a').active_execution_id=null;
stopStream.controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('orphan'))+'\n\n'));
stopStream.controller.close();streamBlocks.delete('orphan');await settle();
assert.match(node('notice').textContent,/Cancellation confirmed.*Session released/);

// Unrelated errors are not replaced by later cancellation readback.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});
const unrelatedStream={};streamBlocks.set('orphan',unrelatedStream);node('refresh').click();await settle();
await node('stop-run').click();await settle();
const unrelated=deferred('GET /api/executions/orphan/models');tab('models').click();await tick();
unrelated.reject(new Error('new detail error'));await settle();
executions.get('orphan').status='CANCELLED';
const oldNotice=node('notice').textContent;
assert.match(oldNotice,/new detail error/);
tab('overview').click();await settle();
unrelatedStream.controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('orphan'))+'\n\n'));
unrelatedStream.controller.close();streamBlocks.delete('orphan');await settle();
assert.equal(node('notice').textContent,oldNotice);tab('overview').click();await settle();

// A failed cancellation readback must not post its error into a newer selection.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});node('refresh').click();await settle();
cancelAttempts=0;
const failedCancelRead=deferred('GET /api/executions/orphan');
node('stop-run').click();await tick();
location.hash='#session=b';await settle();
const newerNotice=node('notice').textContent;
failedCancelRead.reject(new Error('old readback unavailable'));await settle();
assert.equal(node('conversation-title').textContent,'Beta');
assert.equal(node('notice').textContent,newerNotice);

// A late failed Stop response cannot replace another conversation's notice.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});location.hash='#execution=orphan';await settle();
const failedStop=deferred('POST /api/executions/orphan/cancel');node('stop-run').click();await tick();
location.hash='#session=b';await settle();const otherConversationNotice=node('notice').textContent;
failedStop.reject(new Error('old Stop response lost'));await settle();
assert.equal(node('notice').textContent,otherConversationNotice);

// Newer detail errors own the notice even when Stop completes in the same view.
for(const success of [false,true]){
  executions.set('orphan',{...info('orphan',null),status:'STARTED'});location.hash='#execution=orphan';await settle();
  const lateStop=deferred('POST /api/executions/orphan/cancel');node('stop-run').click();await tick();
  const detailFailure=deferred('GET /api/executions/orphan/models');tab('models').click();await tick();
  detailFailure.reject(new Error('new detail error'));await settle();const ownedNotice=node('notice').textContent;
  if(success)lateStop({execution_id:'orphan',cancelled:false});else lateStop.reject(new Error('old Stop response lost'));
  await settle();assert.equal(node('notice').textContent,ownedNotice);tab('overview').click();await settle();
}

// Failed terminal owner readback cannot report an old Stop into another conversation.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});location.hash='#execution=orphan';await settle();
await node('stop-run').click();await settle();executions.get('orphan').status='CANCELLED';
const staleOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await tick();
location.hash='#session=b';await settle();const newOwnerNotice=node('notice').textContent;
staleOwnerRead.reject(new Error('old session release readback failed'));await settle();
assert.equal(node('notice').textContent,newOwnerNotice);

// A newer canonical release wins over a failed older owner readback.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});location.hash='#execution=orphan';await settle();
await node('stop-run').click();await settle();executions.get('orphan').status='CANCELLED';
const olderOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await tick();
const newerOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await tick();
newerOwnerRead({session:{...sessions.get('a'),active_execution_id:null},timeline:null});await settle();
assert.match(node('notice').textContent,/Cancellation confirmed.*Session released/);
const settledNotice=node('notice').textContent;
olderOwnerRead.reject(new Error('outdated owner readback failed'));await settle();
assert.equal(node('notice').textContent,settledNotice);

// Exact-ID lookup preserves opaque sessions and distinguishes standalone executions.
const opaque=' ../team/conversation?notes#你好 ';
sessions.set(opaque,{...sessions.get('a'),session_id:opaque,metadata:{title:'Opaque conversation'}});
timelineOverrides.set(opaque,{items:[],next_cursor:null});
node('open-kind').value='session';node('open-id').value=opaque;node('open-form').requestSubmit();await settle();
assert.equal(new URLSearchParams(location.hash.slice(1)).get('session'),opaque);
assert.equal(node('conversation-title').textContent,'Opaque conversation');
node('open-kind').value='execution';node('open-id').value='b-run';node('open-form').requestSubmit();await settle();
assert.equal(location.hash,'#execution=b-run');assert.equal(node('composer').hidden,true);

// A committed RUNNING identity can have an intentionally empty prompt envelope.
detailResponses.set('/api/executions/b-run/models',{items:[{execution_id:'b-run',agent_run_seq:1,depth:0,model_request_seq:1,purpose:'agent',status:'RUNNING',model:{},request:{},response:null,content_included:true}],next_cursor:null});
tab('models').click();await settle();
assert.match(node('inspector-content').textContent,/Prompt content is not available yet/);
assert.doesNotMatch(node('inspector-content').textContent,/Prompt architecture/);

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
assert.equal(calls.filter(call=>call.key==='/api/executions').at(-1).query.session_id,opaque);assert.match(node('execution-filter-summary').textContent,/1 applied/);
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
assert.equal(node('send').disabled,true);assert.equal(node('prompt').disabled,true);assert.equal(node('current-status').textContent,'Session CLOSED');
console.log('DOM contracts passed: stale navigation/detail/action, disabled stale controls, uncertain fork, interrupted dialog, repeated submit, metrics navigation');
