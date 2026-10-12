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
  set innerHTML(value){this.html=String(value);this.textContent=this.html.replace(/<[^>]*>/g,'').replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&amp;','&');}
  get innerHTML(){return this.html || '';}
  get scrollHeight(){return this.children.length*80;}
  connect(value){this.isConnected=value;if(this.id){if(value)nodes.set(this.id,this);else if(nodes.get(this.id)===this)nodes.delete(this.id);}this.children.forEach(child=>child.connect?.(value));if(!value && document.activeElement===this)document.activeElement=null;}
  append(...children){children.forEach(child=>{if(child.parentElement)child.parentElement.children=child.parentElement.children.filter(item=>item!==child);});this.children.push(...children);children.forEach(child=>{child.parentElement=this;child.connect?.(this.isConnected);});}
  replaceChildren(...children){this.children.forEach(child=>child.connect?.(false));this._text='';this.children=[];this.append(...children);}
  contains(node){return node===this || this.children.some(child=>child.contains?.(node));}
  querySelectorAll(selector){const matches=[];for(const child of this.children){if(selector==='[data-request-focus]' && child.dataset?.requestFocus)matches.push(child);matches.push(...(child.querySelectorAll?.(selector)||[]));}return matches;}
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
let hash='',hashChanged=Promise.resolve();
globalThis.location={get hash(){return hash;},set hash(value){hash=value;hashChanged=Promise.resolve().then(()=>Promise.all(handlers.get('hashchange')?.map(handler=>handler()) || []));}};
let counter=0;Object.defineProperty(globalThis,"crypto",{value:{randomUUID:()=>`request-${++counter}`},configurable:true});globalThis.confirm=()=>true;globalThis.prompt=()=> 'Retry prompt';
const networkFailures=new Set(),calls=[],delays=new Map(),timelineOverrides=new Map(),streamBlocks=new Map(),detailResponses=new Map(),unavailableTimelines=new Set();
const info=(id,session_id=id[0])=>({execution_id:id,agent_id:'default',session_id,status:'SUCCEEDED',binding_kind:'agent',lineage_kind:'ROOT',created_at:'2026-01-01T00:00:00Z',started_at:'2026-01-01T00:00:00Z',terminal_at:'2026-01-01T00:00:01Z'});
const sessions=new Map(['a','b'].map(id=>[id,{session_id:id,agent_id:'default',status:'OPEN',revision:0,history_quality:'complete',metadata:{title:id==='a'?'Alpha':'Beta'}}]));
const executions=new Map([['a-run',info('a-run')],['b-run',info('b-run')]]);
const timeline=id=>timelineOverrides.get(id) || ({items:[{execution_id:`${id}-run`,status:'SUCCEEDED',created_at:'2026-01-01T00:00:00Z',user_input:`${id} question`,conversation_committed:true,items:[{item_kind:'assistant',content:`${id} answer`}]}],next_cursor:null});
function response(value,status=200){return new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}});}
function deferred(path){
  let release,reject,started;
  const promise=new Promise((resolve,fail)=>{release=resolve;reject=fail;});
  const requested=new Promise(resolve=>{started=resolve;});
  delays.set(path,{promise,started});
  const resolve=(value,status=200)=>{delays.delete(path);release(response(value,status));};
  resolve.reject=error=>{delays.delete(path);reject(error);};resolve.requested=requested;
  return resolve;
}
let forkAttempts=0,cancelAttempts=0,endMode=null;
globalThis.fetch=async(path,options={})=>{
  const url=new URL(path,'http://127.0.0.1:8765'),key=url.pathname==='/api/session'?'/api/sessions/'+url.searchParams.get('session_id'):url.pathname.startsWith('/api/session/')?'/api/sessions/'+url.searchParams.get('session_id')+url.pathname.slice('/api/session'.length):url.pathname,method=options.method || 'GET',body=options.body?JSON.parse(options.body):null;
  calls.push({key,method,body,query:Object.fromEntries(url.searchParams)});
  if(networkFailures.has(method+' '+key))throw new Error('Runtime read unavailable');
  if(delays.has(method+' '+key)){const pending=delays.get(method+' '+key);pending.started();return pending.promise;}
  if(key==='/api/config')return response({asset_root:'/workspace/.linktools',read_only:readOnly,memory_scope:'default',capabilities:[{kind:'agent',id:'default',revision:1}],metric_names:[]});
  if(key==='/api/sessions'&&method==='GET')return response({items:[...sessions.values()],next_cursor:null});
  if(key==='/api/executions' && endMode?.listError)throw new Error('list unavailable');
  if(key==='/api/executions')return response({items:[...executions.values()],next_cursor:null,recent_scan:url.searchParams.get('recent')==='true'});
  if(detailResponses.has(key)){const value=detailResponses.get(key),result=typeof value==='function'?value(url):value;return result instanceof Response?result:response(result);}
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
const moduleURL=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const markdownURL=moduleURL(readFileSync(resolve(root,'markdown-it.js'),'utf8'));
const messageURL=moduleURL(readFileSync(resolve(root,'message.js'),'utf8').replace("'./markdown-it.js'",JSON.stringify(markdownURL)));
const source=readFileSync(resolve(root,'app.js'),'utf8').replace("'./console.js'",JSON.stringify(coreURL)).replace("'./message.js'",JSON.stringify(messageURL));
await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
// Stale callbacks without a returned action promise have no positive DOM change.
const settle=async()=>{for(let i=0;i<5;i++)await new Promise(resolve=>setTimeout(resolve,10));};
// Observe each handler without serializing later events behind it.
const select=async hash=>{location.hash=hash;await hashChanged;};
async function until(description,ready){
  const deadline=performance.now()+5000;
  while(!ready()){
    assert.ok(performance.now()<deadline,`Timed out waiting for ${description}`);
    await new Promise(resolve=>setImmediate(resolve));
  }
}
const node=id=>nodes.get(id),tab=name=>({async click(){if(node('inspector').hidden){const opening=node('inspector-toggle').click();if(node('detail-view').value===name)return opening;}node('detail-view').value=name;return node('detail-view').onchange();}}),view=name=>all.find(element=>element.dataset.view===name);
async function openOriginal(){
  if(node('inspector').hidden)await node('inspector-toggle').click();
  node('run-actions').open=true;await node('native-mode').click();
  assert.equal(node('inspector').hidden,true);assert.equal(node('run-actions').open,false);assert.equal(node('request-navigation').hidden,false);
}
async function refresh(){
  const previous=node('list').children;node('refresh').click();
  await until('refreshed list',()=>node('list').children!==previous);
}
async function detail(name,pattern){
  await tab(name).click();
  if(name==='trace' && pattern.test('Read SDK content')){const row=node('inspector-content').children.find(item=>item.className==='trace-item')?.children[0];if(row?.getAttribute('aria-expanded')!=='true')await row?.click();}
  await until(`${name} detail`,()=>pattern.test(node('inspector-content').textContent));
}
await until('initial session list',()=>node('list').children.length===2);
assert.equal(node('list').children.length,2);
await select('#session=a');
assert.equal(node('conversation-title').textContent,'Alpha');
assert.match(node('conversation').textContent,/a answer/);
assert.doesNotMatch(node('conversation').textContent,/Original request timeline/);
assert.equal(node('request-context').hidden,true);
assert.equal(node('request-navigation').hidden,true);
if(process.argv[3]==='minimal') {
  const descendants=node=>[node,...node.children.flatMap(descendants)];
  assert.ok(!descendants(node('conversation')).some(item=>item.tagName==='BUTTON' && ['Details','Model requests'].includes(item.textContent)),'Default turns have no duplicate diagnostic navigation');
  assert.equal(node('request-navigation').hidden,true);assert.equal(node('composer-options').hidden,true);
  const a=timeline('a').items[0],b={...a,execution_id:'a-two',created_at:'2026-01-02T00:00:00Z',user_input:'second'};
  timelineOverrides.set('a',{items:[a,b],next_cursor:null});executions.set('a-two',info('a-two','a'));
  await select('#session=a');await node('inspector-toggle').click();
  assert.equal(node('detail-turn').value,'a-two');assert.deepEqual(node('detail-turn').children.map(item=>item.value),['a-run','a-two']);
  node('detail-turn').value='a-run';await node('detail-turn').onchange();assert.match(node('execution-meta').textContent,/a-run/);
  const records=[{execution_id:'a-run',step_event_seq:9,payload:{kind:'TOOL_CALL',tool_name:'search',agent_run_seq:3,step_index:2,call_id:'call-a',child_execution_id:'child-run',status:'SUCCEEDED',duration_ns:2000000}},{execution_id:'child-run',step_event_seq:1,payload:{kind:'MODEL_RESPONSE',agent_run_seq:1,model_request_seq:4,status:'SUCCEEDED'}}];
  detailResponses.set('/api/executions/a-run/trace',url=>({items:url.searchParams.has('cursor')?[{execution_id:'a-run',step_event_seq:11,payload:{kind:'TOOL_RETURN',tool_name:'search'}}]:records,next_cursor:url.searchParams.has('cursor')?null:'trace-more'}));
  await tab('trace').click();
  const rows=()=>node('inspector-content').children.filter(item=>item.className==='trace-item');
  assert.deepEqual(rows().map(item=>item.children[0].textContent.match(/^#\d+/)[0]),['#9','#1'],'API order is preserved across execution identities');
  assert.ok(rows().every(item=>item.children.length===1),'Trace shows compact rows until an event is selected');
  rows()[0].children[0].focus();await rows()[0].children[0].click();assert.equal(document.activeElement.dataset.requestFocus,'trace:a-run:9');
  assert.match(rows()[0].textContent,/a-run.*run 3.*step 2/);assert.match(rows()[0].textContent,/Open subagent/);assert.match(rows()[0].textContent,/Original event record/);
  const beforePaging=calls.length;await node('detail-more').click();assert.equal(calls[beforePaging].query.cursor,'trace-more');
  assert.equal(rows()[0].children[0].getAttribute('aria-expanded'),'true');assert.equal(document.activeElement.dataset.requestFocus,'trace:a-run:9');assert.equal(rows().length,3);
  rows()[1].children[0].focus();await rows()[1].children[0].click();assert.equal(rows()[0].children.length,1);assert.equal(rows()[1].children[0].getAttribute('aria-expanded'),'true');
  await rows()[0].children[0].click();executions.set('child-run',{...info('child-run','a'),parent_execution_id:'a-run'});
  const childRead=deferred('GET /api/executions/child-run'),openingChild=descendants(rows()[0]).find(item=>item.tagName==='BUTTON' && item.textContent==='Open subagent').click();await childRead.requested;
  assert.equal(node('detail-turn').value,'child-run','Pending child read names its target immediately');
  childRead.reject(new Error('Child unavailable'));await openingChild;assert.equal(node('detail-turn').value,'child-run');
  node('detail-turn').value='a-run';await node('detail-turn').onchange();await rows()[0].children[0].click();
  await descendants(rows()[0]).find(item=>item.tagName==='BUTTON' && item.textContent==='Open subagent').click();
  assert.match(node('execution-meta').textContent,/child-run/);assert.equal(node('detail-turn').value,'child-run');assert.match(node('detail-turn').children.at(-1).textContent,/Execution child-run/);
  await descendants(node('execution-meta')).find(item=>item.tagName==='BUTTON' && item.textContent==='↑ Parent execution').click();assert.equal(node('detail-turn').value,'a-run');
  await openOriginal();assert.equal(node('request-turn').value,'a-run');assert.equal(document.activeElement,node('chat-mode'));node('chat-mode').click();assert.equal(document.activeElement,node('inspector-toggle'));
  console.log('Minimal navigation contracts passed: one entry, explicit turn identity, original Trace order, single selected event, paging/focus, real child/parent and full-width return');process.exit(0);
}
if(process.argv[3]==='details') {
  executions.set('a-run',{...info('a-run'),usage:{logical_requests:2,input_tokens:20,output_tokens:5,cache_read_tokens:0,cache_write_tokens:0,unknown_usage_requests:1}});
  await select('#session=a');node('inspector-toggle').click();await until('default original request details',()=>node('inspector-content').textContent.includes('No records'));
  assert.equal(node('detail-view').value,'models');assert.match(node('execution-meta').textContent,/SUCCEEDED/);assert.match(node('execution-meta').textContent,/1.000 s/);assert.match(node('execution-meta').textContent,/2 requests.*20 in.*5 out.*1 usage unknown/);
  assert.equal(node('session-details').hidden,true);assert.equal(node('recovery-controls').hidden,true);assert.equal(node('detail-attention').hidden,true);
  assert.ok(!calls.some(call=>call.key.endsWith('/history') || call.key.endsWith('/recovery')),'Opening default Details does not fetch diagnostics or recovery');
  node('run-actions').open=true;node('run-actions').onkeydown({key:'Escape'});assert.equal(node('run-actions').open,false);assert.equal(document.activeElement,node('run-actions-toggle'));
  await detail('overview',/Execution metadata/);assert.equal(node('session-details').hidden,false);assert.equal(node('detail-view').value,'overview');
  executions.set('orphan',{...info('orphan','a'),status:'RECOVERY_REQUIRED',terminal_at:null,error_code:'TOOL_EFFECT_OUTCOME_UNKNOWN'});sessions.get('a').active_execution_id='orphan';streamBlocks.set('orphan',{});
  await select('#session=a&execution=orphan');assert.equal(node('detail-attention').hidden,false);assert.equal(node('detail-recovery').hidden,false);assert.match(node('detail-attention-text').textContent,/RECOVERY REQUIRED.*TOOL_EFFECT_OUTCOME_UNKNOWN/);assert.match(node('execution-meta').textContent,/Duration unavailable/);assert.doesNotMatch(node('execution-meta').textContent,/In progress/);
  await node('detail-recovery').click();assert.equal(node('detail-view').value,'recovery');assert.equal(node('recovery-controls').hidden,false);assert.equal(node('resume-run').disabled,false);assert.equal(node('end-stopped').disabled,false);
  const posts=calls.filter(call=>call.method==='POST').length;globalThis.confirm=()=>false;await node('resume-run').click();await node('end-stopped').click();assert.equal(calls.filter(call=>call.method==='POST').length,posts,'Dismissed Resume/End confirmations have no side effects');globalThis.confirm=()=>true;
  executions.set('a-run',{...info('a-run'),status:'FAILED',error_code:'MODEL_ERROR'});await select('#session=a&execution=a-run');
  assert.equal(node('inspector-active-stop').hidden,false);assert.equal(node('stop-run').hidden,true);assert.match(node('detail-attention-text').textContent,/Failed.*MODEL_ERROR/);assert.equal(node('detail-recovery').hidden,true);assert.equal(node('retry').disabled,false);
  node('run-actions').open=true;await select('#session=b');assert.equal(node('run-actions').open,false,'Changing selected run closes its actions menu');assert.equal(node('detail-attention').hidden,true);
  node('run-actions').open=true;await select('#session=a&execution=a-run');assert.equal(node('run-actions').open,false,'Explicit session/execution navigation closes the previous target actions menu');
  node('run-actions').open=true;await select('#view=executions&execution=a-run');assert.equal(node('run-actions').open,false,'Same execution navigation into the independent browser closes the prior view action menu');await detail('overview',/Execution metadata/);assert.equal(node('session-details').hidden,true,'Standalone execution inspection cannot expose the previous conversation metadata');
  await select('#session=b&execution=a-run');assert.equal(node('session-details').hidden,true,'Another execution session is not the surrounding conversation');
  console.log('Details contracts passed: compact default source, summary, metadata, explicit recovery, confirmations, selected/active actions and menu focus');process.exit(0);
}
if(process.argv[3]==='guided') {
  const descendants=node=>[node,...node.children.flatMap(descendants)];
  assert.equal(node('conversation-heading').hidden,false);assert.equal(node('page-heading').hidden,true);
  assert.equal(node('inspector').hidden,true);assert.ok(!calls.some(call=>call.key.endsWith('/history')),'SDK records stay opt-in');
  assert.equal(calls.filter(call=>call.key.endsWith('/models')).length,0,'Default conversation does not fetch original or SDK request bodies');
  const source='{ "messages": [{"role":"system","content":"actual system first"},{"role":"system","content":"actual instructions second"}],"untouched":9007199254740993 }';
  const original={execution_id:'a-run',agent_run_seq:1,model_request_seq:1,purpose:'agent',status:'SUCCEEDED',model:{},client_requests_included:true,client_requests:[{api:'openai.chat.completions',body:Buffer.from(source).toString('base64'),body_encoding:'base64',response:null}]};
  detailResponses.set('/api/executions/a-run/history',{items:[
    {execution_id:'a-run',message_seq:1,agent_run_seq:1,part_index:null,item_kind:'instructions',content:'SDK instructions field first in projection'},
    {execution_id:'a-run',message_seq:1,agent_run_seq:1,part_index:0,item_kind:'system',content:'SDK system part follows field metadata'},
  ],next_cursor:null});
  detailResponses.set('/api/executions/a-run/models',{items:[original],next_cursor:null});
  await detail('history',/SDK instructions field first in projection/);
  assert.equal(node('detail-view').value,'history');
  assert.match(node('inspector-content').textContent,/not outbound messages/);assert.match(node('inspector-content').textContent,/Display order is not client message order/);
  assert.match(node('inspector-content').textContent,/SDK instructions field/);assert.match(node('inspector-content').textContent,/field metadata/);assert.match(node('inspector-content').textContent,/SDK system part/);
  assert.doesNotMatch(node('inspector-content').textContent,/Message 1/);
  for(const value of ['SDK instructions field first in projection','SDK system part follows field metadata'])assert.ok(descendants(node('inspector-content')).some(item=>item.tagName==='PRE' && item.textContent===value),'SDK field and part content remain unchanged');
  await descendants(node('inspector-content')).find(item=>item.tagName==='BUTTON' && item.textContent==='Open original model requests').click();
  assert.ok(descendants(node('inspector-content')).some(item=>item.tagName==='PRE' && item.textContent===source),'Original source follows captured client order, never SDK projection order');
  assert.doesNotMatch(node('inspector-content').textContent,/SDK instructions field first in projection/);
  node('inspector-close').click();await openOriginal();await until('independent model request mode',()=>node('conversation').textContent.includes('actual system first'));
  assert.equal(node('request-context').hidden,false);assert.equal(node('request-turn').value,'a-run');
  assert.doesNotMatch(node('conversation').textContent,/a question|a answer/,'Request mode does not interleave conversation turns with full outbound arrays');
  const request=descendants(node('conversation')).find(item=>item.className.includes('model-request') && item.tagName==='DETAILS');
  assert.equal(request.open,false);request.open=true;request.ontoggle();request.children[0].focus();
  await descendants(node('conversation')).find(item=>item.tagName==='BUTTON' && item.textContent==='Refresh original requests').click();
  assert.equal(descendants(node('conversation')).find(item=>item.className.includes('model-request') && item.tagName==='DETAILS').open,true,'Refreshing source preserves the expanded request');
  assert.equal(document.activeElement?.dataset.requestFocus,'a-run:1:1:request','Source refresh preserves keyboard focus on the same logical request');
  await detail('models',/actual system first/);
  const inspectorRequest=descendants(node('inspector-content')).find(item=>item.tagName==='DETAILS' && item.className.includes('model-request'));
  inspectorRequest.open=true;inspectorRequest.ontoggle();inspectorRequest.open=false;inspectorRequest.ontoggle();
  await openOriginal();await descendants(node('conversation')).find(item=>item.tagName==='BUTTON' && item.textContent==='Refresh original requests').click();
  assert.equal(descendants(node('conversation')).find(item=>item.className.includes('model-request') && item.tagName==='DETAILS').open,true,'Closing the inspector copy cannot collapse the original request being read in the conversation pane');
  timelineOverrides.set('a',{items:[...timeline('a').items,{...timeline('a').items[0],execution_id:'a-two',user_input:'Second question'}],next_cursor:null});
  await select('#session=a&execution=a-run');await openOriginal();
  networkFailures.add('GET /api/executions/a-two');node('request-turn').value='a-two';await node('request-turn').onchange();
  assert.equal(node('request-turn').value,'a-two');assert.match(node('conversation').textContent,/Runtime read unavailable/);
  assert.doesNotMatch(node('conversation').textContent,/actual system first/,'Failed turn selection cannot retain another turn original bytes');
  assert.ok(!descendants(node('conversation')).some(item=>item.tagName==='BUTTON' && item.textContent==='Download original bytes'));
  networkFailures.clear();await select('#session=a&execution=a-run');
  node('chat-mode').click();assert.match(node('conversation').textContent,/a answer/);assert.equal(node('request-context').hidden,true);
  await detail('transcript',/SDK text excerpts only/);assert.match(node('inspector-content').textContent,/System, instructions, tools/);
  node('new-session').click();assert.equal(document.activeElement,node('new-agent'));assert.equal(node('new-options').open,false);
  node('new-dialog').close();await openOriginal();node('new-session').click();node('new-title').value='Fresh chat';node('new-agent').value='default';
  const creation=deferred('POST /api/sessions'),creating=node('new-form').requestSubmit();await creation.requested;
  const createdId=calls.findLast(call=>call.method==='POST' && call.key==='/api/sessions').body.session_id;
  sessions.set(createdId,{session_id:createdId,agent_id:'default',status:'OPEN',revision:0,history_quality:'complete',metadata:{title:'Fresh chat'}});timelineOverrides.set(createdId,{items:[],next_cursor:null});
  creation({session_id:createdId});await creating;await until('fresh conversation composer focus',()=>node('conversation-title').textContent==='Fresh chat' && document.activeElement===node('prompt'));
  assert.equal(node('request-navigation').hidden,true,'A new conversation always starts in chat, even after inspecting original requests');
  node('menu').click();assert.equal(node('menu').getAttribute('aria-expanded'),'true');
  node('sidebar-close').click();assert.equal(node('menu').getAttribute('aria-expanded'),'false');assert.equal(document.activeElement,node('menu'));
  node('menu').click();view('metrics').click();await hashChanged;assert.equal(node('menu').getAttribute('aria-expanded'),'false');
  node('menu').click();await Promise.all(handlers.get('keydown').map(handler=>handler({key:'Escape'})));assert.equal(node('menu').getAttribute('aria-expanded'),'false');assert.equal(document.activeElement,node('menu'));
  console.log('Guided conversation contracts passed: default chat, independent original requests, unchanged bytes, SDK field/part provenance and discoverable controls');process.exit(0);
}
if(process.argv[3]==='native') {
  await openOriginal();await until('original request reading mode',()=>node('conversation').textContent.includes('Refresh original requests'));
  const descendants=node=>[node,...node.children.flatMap(descendants)];
  const clickText=label=>descendants(node('conversation')).find(item=>item.tagName==='BUTTON' && item.textContent===label).click();
  const bytes=value=>({body:Buffer.from(value).toString('base64'),body_encoding:'base64'});
  const messages=[{role:'developer',content:'<script>native only</script>',unknown:{preserve:null}},{role:'assistant',content:'',tool_calls:[{id:'call',function:{name:'read',arguments:'{"x":1}'}}]},{role:'tool',tool_call_id:'call',content:[{type:'text',text:'result'}]}];
  const source=JSON.stringify({messages,extra_body_field:{arbitrary:true}}).slice(0,-1)+',"large_integer":9007199254740993,"duplicate":1,"duplicate":2}';
  const responseText='data: {"choices":[{"delta":{"role":"assistant","content":"<img src=x onerror=alert(1)>"}}]}\n\n';
  const item={execution_id:'a-run',agent_run_seq:1,depth:0,model_request_seq:1,purpose:'agent',status:'CANCELLED',step_index:4,output_retry_index:2,started_at:'2026-01-01T00:00:00Z',finished_at:'2026-01-01T00:00:01Z',duration_ns:1000000000,model:{},content_included:false,request:{instructions:['DO NOT SYNTHESIZE SYSTEM']},client_requests_included:true,client_requests:[
    {api:'openai.chat.completions',...bytes(source),response:{...bytes('{"error":"retry"}'),status_code:429,format:'json',content_encoding:null,complete:true}},
    {api:'openai.chat.completions',...bytes(source),response:{...bytes(responseText),status_code:200,format:'sse',content_encoding:null,complete:false}},
  ]};
  detailResponses.set('/api/executions/a-run/models',{items:[item],next_cursor:'raw-next'});
  await clickText('Refresh original requests');
  assert.match(node('conversation').textContent,/Original outbound request body/);
  assert.ok(descendants(node('conversation')).some(item=>item.tagName==='PRE' && item.textContent===source),'Exact source retains large integers, duplicate keys, empty content, roles and all native fields');
  assert.match(node('conversation').textContent,/"large_integer":9007199254740993,"duplicate":1,"duplicate":2/);
  assert.match(node('conversation').textContent,/Logical request timing and identity/);
  assert.match(node('conversation').textContent,/"step_index": 4/);
  assert.match(node('conversation').textContent,/"output_retry_index": 2/);
  assert.match(node('conversation').textContent,/"duration_ns": 1000000000/);
  assert.match(node('conversation').textContent,/HTTP 429 · complete/);
  assert.match(node('conversation').textContent,/HTTP 200 · incomplete/);
  assert.match(node('conversation').textContent,/<img src=x onerror=alert\(1\)>/);
  assert.doesNotMatch(node('conversation').textContent,/DO NOT SYNTHESIZE SYSTEM/);
  const attempts=descendants(node('conversation')).filter(item=>item.dataset.clientAttempt);
  assert.ok(attempts.every(item=>item.tagName==='DETAILS' && !item.open),'Each HTTP attempt is independently expandable');
  attempts[1].open=true;attempts[1].ontoggle();attempts[1].children[0].focus();
  await clickText('Refresh original requests');
  const refreshedAttempts=descendants(node('conversation')).filter(item=>item.dataset.clientAttempt);
  assert.equal(refreshedAttempts[0].open,false);assert.equal(refreshedAttempts[1].open,true);assert.equal(document.activeElement.dataset.requestFocus,'a-run:1:1:1:attempt');
  assert.deepEqual(attempts.map(item=>item.dataset.clientAttempt),['a-run:1:1:0','a-run:1:1:1']);
  assert.ok(descendants(node('conversation')).filter(item=>item.tagName==='PRE').every(item=>!item.innerHTML),'Original body uses text nodes, never HTML rendering');
  let download;
  URL.createObjectURL=blob=>{download=blob;return 'blob:test';};URL.revokeObjectURL=()=>{};
  await clickText('Download original bytes');
  assert.equal(await download.text(),source);
  const beforeInspector=calls.filter(call=>call.key==='/api/executions/a-run/models').length;await detail('models',/Original outbound request body/);
  assert.equal(calls.filter(call=>call.key==='/api/executions/a-run/models').length,beforeInspector,'Opening the original inspector reuses the visible timeline body page');tab('overview').click();await openOriginal();
  networkFailures.add('GET /api/executions/a-run/models');
  await clickText('Load more model requests');
  assert.match(node('conversation').textContent,/Runtime read unavailable/);
  assert.ok(descendants(node('conversation')).some(item=>item.tagName==='PRE' && item.textContent===source),'Failed paging preserves loaded content');
  assert.ok(descendants(node('conversation')).some(item=>item.tagName==='BUTTON' && item.textContent==='Load more model requests'),'Failed paging retains its retry cursor');
  networkFailures.clear();
  const next=deferred('GET /api/executions/a-run/models'),before=calls.length;
  const moreButton=descendants(node('conversation')).find(item=>item.tagName==='BUTTON' && item.textContent==='Load more model requests');
  const more=moreButton.click();moreButton.click();await next.requested;
  assert.equal(calls.slice(before).filter(call=>call.query.cursor==='raw-next').length,1);
  next({items:[{...item,model_request_seq:2,client_requests:null},{...item,model_request_seq:3,client_requests:[]}],next_cursor:null});
  await more;
  assert.match(node('conversation').textContent,/were not recorded/);
  assert.match(node('conversation').textContent,/No supported client dispatch/);
  const stale=deferred('GET /api/executions/a-run/models'),staleRefresh=clickText('Refresh original requests');await stale.requested;
  await select('#session=b');
  stale({items:[item],next_cursor:null});await staleRefresh;
  assert.doesNotMatch(node('conversation').textContent,/native only/);
  assert.ok(calls.filter(call=>call.key.endsWith('/models')).every(call=>call.query.include_client_requests==='true' && !call.query.include_content));
  const id='native-live',path=`/api/executions/${id}/models`;
  sessions.get('a').active_execution_id=id;executions.set(id,{...info(id,'a'),status:'STARTED'});streamBlocks.set(id,{});
  const completed=seq=>({...item,execution_id:id,model_request_seq:seq,purpose:seq===22?'compaction':'agent',status:'SUCCEEDED'});
  detailResponses.set(path,url=>url.searchParams.has('model_request_seq')?{items:[completed(Number(url.searchParams.get('model_request_seq')))],next_cursor:null}:url.searchParams.has('cursor')?{items:Array.from({length:10},(_,i)=>completed(i+11)),next_cursor:null}:{items:Array.from({length:10},(_,i)=>completed(i+1)),next_cursor:'live-next'});
  await select('#session=a');
  const controller=streamBlocks.get(id).controller;assert.ok(controller);
  const emit=value=>controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify(value)+'\n\n'));
  const projection=(seq,status)=>({type:'model',item:{item:{...completed(seq),status,client_requests:null},visibility:'durable_history'}});
  const bodyCalls=()=>calls.filter(call=>call.key===path),bodyReads=()=>bodyCalls().length;
  // Negative timer assertions must span the 60 ms original-body debounce.
  const waitNativeRefresh=()=>new Promise(resolve=>setTimeout(resolve,80));
  emit(projection(20,'SUCCEEDED'));await until('completed original request 20',()=>node('conversation').textContent.includes('Request 20 · SUCCEEDED'));
  const oldPage=deferred(`GET ${path}`),oldPaging=clickText('Load more model requests');await oldPage.requested;
  oldPage({items:[...Array.from({length:9},(_,i)=>completed(i+11)),{...completed(20),status:'RUNNING',client_requests:null}],next_cursor:null});await oldPaging;
  assert.deepEqual(descendants(node('conversation')).filter(value=>value.tagName==='H3' && value.textContent.startsWith('Request ')).map(value=>Number(value.textContent.split(' ')[1])),Array.from({length:20},(_,i)=>i+1),'Later paging retains logical request order after a newer point read');
  assert.match(node('conversation').textContent,/Request 20 · SUCCEEDED/);assert.doesNotMatch(node('conversation').textContent,/Request 20 · RUNNING/);
  const beforeReplay=bodyReads();emit(projection(1,'SUCCEEDED'));await waitNativeRefresh();assert.equal(bodyReads(),beforeReplay,'Initial observation replay reuses already-loaded terminal originals');
  const beforeDeltas=bodyReads();emit(projection(21,'RUNNING'));
  for(let i=0;i<4;i++)emit({type:'event',item:{execution_id:id,depth:0,event:{event_type:'ASSISTANT_TEXT_DELTA',payload:{text:i?'!':'**live**'}}}});
  await until('live token batch',()=>node('live').textContent.includes('live!!!'));await waitNativeRefresh();assert.equal(bodyReads(),beforeDeltas,'Tokens and RUNNING metadata never read original bodies');
  assert.match(node('live').textContent,/LIVE RESPONSE · incomplete SDK display/);assert.match(node('live').textContent,/live!!!/);
  assert.ok(descendants(node('live')).some(value=>value.innerHTML.includes('<strong>live</strong>')),'Live Markdown remains visible in original mode');
  emit(projection(21,'SUCCEEDED'));emit(projection(22,'SUCCEEDED'));emit(projection(21,'SUCCEEDED'));await until('completed original requests 21 and 22',()=>node('conversation').textContent.includes('Request 21 · SUCCEEDED') && node('conversation').textContent.includes('Request 22 · SUCCEEDED'));
  assert.equal(bodyReads(),beforeDeltas+2,'Distinct completions read exactly their own requests and coalesce replays');
  assert.ok(bodyCalls().slice(beforeDeltas).every(call=>call.query.agent_run_seq==='1' && ['21','22'].includes(call.query.model_request_seq)),'Completion never rereads previous history pages');
  assert.match(node('conversation').textContent,/Request 19 · SUCCEEDED/);assert.match(node('conversation').textContent,/Request 22 · SUCCEEDED/);assert.match(node('conversation').textContent,/compaction/);assert.match(node('execution-meta').textContent,/STARTED/);
  emit(projection(21,'SUCCEEDED'));await waitNativeRefresh();assert.equal(bodyReads(),beforeDeltas+2,'Replayed completion does not reread bodies');
  networkFailures.add(`GET ${path}`);await clickText('Refresh original requests');
  assert.match(node('conversation').textContent,/Request 19 · SUCCEEDED/);assert.match(node('conversation').textContent,/Request 22 · SUCCEEDED/);assert.match(node('conversation').textContent,/Runtime read unavailable/);
  networkFailures.clear();
  const loading=deferred(`GET ${path}`),loadingRefresh=clickText('Refresh original requests');await loading.requested;emit(projection(23,'SUCCEEDED'));await until('completion during a body read',()=>node('inspector-content').textContent.includes('Request 23 · SUCCEEDED'));
  loading({items:Array.from({length:10},(_,i)=>completed(i+1)),next_cursor:'live-next'});await loadingRefresh;await until('original body for queued completion',()=>node('conversation').textContent.includes('Request 23 · SUCCEEDED'));
  assert.match(node('conversation').textContent,/Request 19 · SUCCEEDED/);assert.match(node('conversation').textContent,/Request 23 · SUCCEEDED/);
  assert.equal(bodyCalls().filter(call=>call.query.model_request_seq==='23').length,1,'Completion during a body read refreshes its request once after paging finishes');
  const beforeHidden=bodyReads();emit(projection(24,'SUCCEEDED'));await until('completion queued before hiding original view',()=>node('inspector-content').textContent.includes('Request 24 · SUCCEEDED'));node('chat-mode').click();await waitNativeRefresh();
  assert.equal(bodyReads(),beforeHidden,'A queued refresh cannot read bodies after the original view is hidden');
  openOriginal();await until('original body after reopening original view',()=>node('conversation').textContent.includes('Request 24 · SUCCEEDED'));assert.equal(bodyReads(),beforeHidden+1);assert.match(node('conversation').textContent,/Request 24 · SUCCEEDED/);
  networkFailures.add(`GET ${path}`);emit(projection(25,'SUCCEEDED'));await until('failed original point read',()=>node('conversation').textContent.includes('request 25: Runtime read unavailable'));
  assert.match(node('conversation').textContent,/Request 24 · SUCCEEDED/);const failedReads=bodyReads();await waitNativeRefresh();assert.equal(bodyReads(),failedReads,'Failed point reads do not retry in a loop');
  networkFailures.clear();await clickText('Refresh original requests');await until('retried original point read',()=>node('conversation').textContent.includes('Request 25 · SUCCEEDED'));assert.match(node('conversation').textContent,/Request 25 · SUCCEEDED/);
  const child={...completed(1),execution_id:'native-child'};
  detailResponses.set('/api/executions/native-child/models',{items:[child],next_cursor:null});
  emit({type:'model',item:{item:{...child,client_requests:null},visibility:'durable_history'}});await until('child original body',()=>descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='native-child:1:1:0'));
  assert.ok(calls.some(call=>call.key==='/api/executions/native-child/models' && call.query.agent_run_seq==='1' && call.query.model_request_seq==='1'),'Child completion reads its actual execution identity');
  assert.ok(descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='native-live:1:1:0'));
  assert.ok(descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='native-child:1:1:0'));
  const deniedChild={...completed(1),execution_id:'denied-child'},deniedPath='/api/executions/denied-child/models';
  detailResponses.set(deniedPath,()=>response({error_code:'AUTHORIZATION_DENIED'},403));
  const corruptChild={...completed(1),execution_id:'corrupt-child'},corruptPath='/api/executions/corrupt-child/models';
  detailResponses.set(corruptPath,()=>response({error_code:'STORAGE_INTEGRITY_ERROR'},500));
  const readyChild={...child,model_request_seq:2};detailResponses.set('/api/executions/native-child/models',{items:[readyChild],next_cursor:null});
  emit({type:'model',item:{item:{...deniedChild,client_requests:null},visibility:'durable_history'}});
  emit({type:'model',item:{item:{...corruptChild,client_requests:null},visibility:'durable_history'}});
  emit({type:'model',item:{item:{...readyChild,client_requests:null},visibility:'durable_history'}});
  emit(projection(26,'SUCCEEDED'));await until('root completion after failed child reads',()=>node('conversation').textContent.includes('Request 26 · SUCCEEDED'));
  assert.match(node('conversation').textContent,/Request 26 · SUCCEEDED/,'One failed child body read cannot block another completed root request');
  assert.match(node('conversation').textContent,/denied-child · request 1: AUTHORIZATION_DENIED/);
  assert.match(node('conversation').textContent,/corrupt-child · request 1: STORAGE_INTEGRITY_ERROR/);
  assert.ok(descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='native-child:1:2:0'));
  const deniedReads=()=>calls.filter(call=>call.key===deniedPath).length;
  assert.equal(deniedReads(),1);emit(projection(27,'SUCCEEDED'));await until('next independent root completion',()=>node('conversation').textContent.includes('Request 27 · SUCCEEDED'));
  assert.match(node('conversation').textContent,/Request 27 · SUCCEEDED/);assert.equal(deniedReads(),1,'Unrelated completions do not automatically retry a failed body read');
  assert.match(node('conversation').textContent,/denied-child · request 1: AUTHORIZATION_DENIED/);
  detailResponses.set(deniedPath,{items:[deniedChild],next_cursor:null});detailResponses.set(corruptPath,{items:[corruptChild],next_cursor:null});
  await clickText('Refresh original requests');await until('retried child original bodies',()=>descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='denied-child:1:1:0'));assert.equal(deniedReads(),2,'Explicit refresh retries the failed identity');
  assert.ok(descendants(node('conversation')).some(value=>value.dataset.clientAttempt==='denied-child:1:1:0'));
  assert.doesNotMatch(node('conversation').textContent,/AUTHORIZATION_DENIED|STORAGE_INTEGRITY_ERROR/);
  console.log('Original timeline contracts passed: native arrays, attempts, response provenance, safe text, byte download, paging and navigation');
  process.exit(0);
}
node('chat-mode').click();
assert.match(node('conversation').textContent,/a answer/);

// Progressive disclosure keeps explicit controls, focus, and authoritative ownership.
assert.equal(node('inspector').hidden,true);assert.equal(node('current-status').textContent,'Session available');
assert.equal(node('inspector-toggle').getAttribute('aria-expanded'),'false');
const beforeDetails=calls.length;node('inspector-toggle').click();
assert.equal(node('inspector').hidden,false);assert.equal(document.activeElement,node('inspector-close'));
assert.equal(node('inspector-toggle').getAttribute('aria-expanded'),'true');
assert.equal(node('detail-view').value,'models','Details opens directly to original model requests');
assert.ok(calls.slice(beforeDetails).some(call=>call.key.endsWith('/models') && call.query.include_client_requests==='true'));
const afterDetails=calls.length;node('inspector-close').click();assert.equal(document.activeElement,node('inspector-toggle'));
assert.equal(node('inspector').hidden,true);assert.equal(calls.length,afterDetails,'Closing Details does not fetch another view');
node('action-menu').open=true;node('action-menu').onkeydown({key:'Escape'});
assert.equal(node('action-menu').open,false);assert.equal(document.activeElement,node('actions-toggle'));
node('options-toggle').click();assert.equal(node('composer-options').hidden,false);
assert.equal(node('options-toggle').getAttribute('aria-expanded'),'true');node('options-toggle').click();
for(const id of ['planning','thinking','memory','files','rename-session','fork-session','close-session','retry','fork-run','end-stopped','export','detail-more','turns-more','list-more'])assert.ok(node(id),`missing retained control ${id}`);
assert.match(html,/<details id="action-menu"[\s\S]*?id="rename-session"[\s\S]*?id="close-session"[\s\S]*?<\/details>/);
assert.match(html,/<details id="open-tools"[\s\S]*?id="open-record"[\s\S]*?<\/details>/);
assert.deepEqual(all.filter(item=>item.dataset.tab).map(item=>item.dataset.tab).sort(),['overview','history','transcript','models','sdkmodels','trace','recovery'].sort(),'All inspection functions remain reachable through the single view picker');
if(readOnly){
  for(const id of ['send','prompt','planning','thinking','memory','files','new-session','retry','fork-run','end-stopped','rename-session','fork-session','close-session'])assert.equal(node(id).disabled,true,id);
  assert.equal(node('connection').hidden,false);assert.equal(node('connection').textContent,'Read-only');
  node('inspector-toggle').click();assert.equal(node('inspector').hidden,false);assert.equal(node('export').disabled,false);
  await detail('history',/History marker/);assert.match(node('inspector-content').textContent,/History marker/);
  await detail('recovery',/require an execution Runtime/);assert.match(node('inspector-content').textContent,/require an execution Runtime/);
  node('open-tools').open=true;node('open-kind').value='execution';node('open-id').value='b-run';node('open-form').requestSubmit();await hashChanged;
  assert.equal(node('inspector').hidden,false);assert.equal(node('open-record').disabled,false);
  assert.equal(calls.filter(call=>call.method==='POST').length,0);
  console.log('Read-only disclosure contracts passed');process.exit(0);
}
if(process.argv[3]==='navigation'){
  await detail('models',/No records/);node('inspector-close').click();const hiddenReads=calls.length;
  await select('#session=b');assert.equal(node('inspector').hidden,true);
  assert.ok(!calls.slice(hiddenReads).some(item=>item.key.endsWith('/models')),'A hidden inspector cannot request another execution body');
  node('inspector-toggle').click();await until('reopened original request inspector',()=>node('inspector-content').textContent.includes('No records'));assert.ok(calls.slice(hiddenReads).some(item=>item.key==='/api/executions/b-run/models' && item.query.include_client_requests==='true'));
  node('inspector-close').click();const hiddenRefresh=calls.length;await refresh();
  assert.ok(!calls.slice(hiddenRefresh).some(item=>item.key.endsWith('/models')),'Refreshing a conversation does not refresh hidden detail bodies');
  const closedDetail=deferred('GET /api/executions/b-run/models');node('inspector-toggle').click();await closedDetail.requested;node('inspector-close').click();
  closedDetail.reject(new Error('closed detail failed'));await settle();assert.equal(node('notice').textContent,'');
  tab('overview').click();node('inspector-close').click();
  const oldMetrics=deferred('GET /api/metrics');view('metrics').click();await oldMetrics.requested;
  await select('#session=b');oldMetrics.reject(new Error('old metrics read failed'));await settle();
  assert.equal(node('notice').textContent,'','A previous page cannot replace the current conversation notice');
  const olderQuery=deferred('GET /api/metrics');view('metrics').click();await olderQuery.requested;
  delays.delete('GET /api/metrics');await node('metrics-form').requestSubmit();
  olderQuery.reject(new Error('superseded metrics query failed'));await settle();assert.equal(node('notice').textContent,'');
  const currentQuery=deferred('GET /api/metrics'),currentMetrics=node('metrics-form').requestSubmit();await currentQuery.requested;
  currentQuery.reject(new Error('current metrics query failed'));await currentMetrics;assert.match(node('notice').textContent,/current metrics query failed/);
  for(const [action,control] of [['update','rename-session'],['close','close-session'],['fork','fork-session']]){
    await select('#session=a');const oldAction=deferred(`POST /api/sessions/a/${action}`);
    const actionResult=node(control).click();await oldAction.requested;await select('#session=b');
    oldAction.reject(new Error(`old ${action} failed`));await actionResult;assert.equal(node('notice').textContent,'');
  }
  const oldRename=deferred('POST /api/sessions/b/update'),oldRenaming=node('rename-session').click();await oldRename.requested;
  const newDetail=deferred('GET /api/executions/b-run/models');tab('models').click();await newDetail.requested;
  newDetail.reject(new Error('new detail read failed'));await until('new detail error',()=>node('notice').textContent.includes('new detail read failed'));const latestNotice=node('notice').textContent;
  oldRename.reject(new Error('older rename failed'));await oldRenaming;assert.equal(node('notice').textContent,latestNotice);
  const currentRename=deferred('POST /api/sessions/b/update'),currentRenaming=node('rename-session').click();await currentRename.requested;
  currentRename.reject(new Error('current rename failed'));await currentRenaming;assert.match(node('notice').textContent,/current rename failed/);
  tab('overview').click();
  sessions.get('a').active_execution_id='page-old';executions.set('page-old',{...info('page-old','a'),status:'STARTED'});streamBlocks.set('page-old',{});
  timelineOverrides.set('a',{...timeline('a'),next_cursor:'older'});await select('#session=a');
  const oldSession=structuredClone(sessions.get('a')),olderPage=deferred('GET /api/sessions/a');
  const pageRequests=()=>calls.filter(item=>item.key==='/api/sessions/a' && item.query.cursor==='older').length;
  const paging=node('turns-more').click();node('turns-more').click();await olderPage.requested;assert.equal(pageRequests(),1);
  delays.delete('GET /api/sessions/a');executions.get('page-old').status='SUCCEEDED';sessions.get('a').active_execution_id='page-new';
  executions.set('page-new',{...info('page-new','a'),status:'STARTED'});streamBlocks.set('page-new',{});
  const oldController=streamBlocks.get('page-old').controller;
  oldController.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('page-old'))+'\n\n'));oldController.close();await until('current owner after older page request',()=>node('cancel').getAttribute('aria-label')?.includes('page-new'));
  assert.match(node('cancel').getAttribute('aria-label'),/page-new/);
  olderPage({session:oldSession,timeline:{items:[{...timeline('a').items[0],execution_id:'earlier-run',user_input:'Earlier question'}],next_cursor:null}});await paging;
  assert.match(node('conversation').textContent,/Earlier question/);assert.match(node('cancel').getAttribute('aria-label'),/page-new/);
  const newCancel=deferred('POST /api/executions/page-new/cancel'),cancelling=node('cancel').click();await newCancel.requested;
  assert.equal(calls.filter(item=>item.method==='POST').at(-1).key,'/api/executions/page-new/cancel');
  executions.get('page-new').status='CANCELLING';newCancel({execution_id:'page-new',cancelled:false});await cancelling;
  const currentNotice=node('notice').textContent;assert.match(currentNotice,/Cancellation requested/);
  await select('#session=a');const failedPage=deferred('GET /api/sessions/a'),failedPaging=node('turns-more').click();await failedPage.requested;
  timelineOverrides.set('b',{...timeline('b'),next_cursor:'beta-older'});await select('#session=b');
  assert.equal(node('turns-more').disabled,false,'Paging another conversation is independent');
  failedPage.reject(new Error('older page failed'));await failedPaging;assert.equal(node('notice').textContent,'');

  sessions.get('a').active_execution_id='reopen-old';executions.set('reopen-old',{...info('reopen-old','a'),status:'STARTED'});streamBlocks.set('reopen-old',{});
  await select('#session=a');const previousSession=structuredClone(sessions.get('a')),reopen=deferred('GET /api/sessions/a');
  const reopening=select('#session=a&execution=a-run');await reopen.requested;delays.delete('GET /api/sessions/a');
  executions.get('reopen-old').status='SUCCEEDED';sessions.get('a').active_execution_id='reopen-new';
  executions.set('reopen-new',{...info('reopen-new','a'),status:'STARTED'});streamBlocks.set('reopen-new',{});
  const reopeningController=streamBlocks.get('reopen-old').controller;
  reopeningController.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('reopen-old'))+'\n\n'));reopeningController.close();await until('new current owner during session reopen',()=>node('cancel').getAttribute('aria-label')?.includes('reopen-new'));
  assert.match(node('cancel').getAttribute('aria-label'),/reopen-new/);
  reopen({session:previousSession,timeline:timeline('a')});await reopening;
  assert.match(node('cancel').getAttribute('aria-label'),/reopen-new/);assert.equal(node('cancel').hidden,false);
  assert.match(node('execution-meta').textContent,/a-run/);assert.doesNotMatch(node('execution-meta').textContent,/Loading/);

  await select('#execution=a-run');const recent=deferred('GET /api/executions');
  node('list-action').value='recent';node('filter-form').requestSubmit();await recent.requested;
  await select('#session=b');recent.reject(new Error('old recent scan failed'));await settle();assert.equal(node('notice').textContent,'');
  const creation=deferred('POST /api/sessions');node('new-session').click();const creating=node('new-form').requestSubmit();await creation.requested;
  node('new-dialog').close();await select('#session=a');creation.reject(new Error('dismissed creation failed'));await creating;assert.equal(node('notice').textContent,'');
  const priorCreation=deferred('POST /api/sessions');node('new-session').click();const priorSession=node('new-dialog').dataset.sessionId;
  const priorCreating=node('new-form').requestSubmit();await priorCreation.requested;assert.equal(node('create-session').disabled,true);
  node('new-dialog').close();node('new-session').click();assert.equal(node('create-session').disabled,false);
  const activeCreation=deferred('POST /api/sessions'),activeCreating=node('new-form').requestSubmit();await activeCreation.requested;assert.equal(node('create-session').disabled,true);
  priorCreation({session_id:priorSession});await priorCreating;assert.equal(node('create-session').disabled,true,'A prior creation cannot enable the pending dialog submit');
  activeCreation.reject(new Error('current creation failed'));await activeCreating;assert.equal(node('create-session').disabled,false);assert.match(node('notice').textContent,/current creation failed/);node('new-dialog').close();

  detailResponses.set('/api/executions/effect-run/recovery',{items:[{tool_name:'write',operation_id:'effect-operation',fence:1,tool_call_id:'call'}],next_cursor:null});
  for(const success of [false,true]){
    await select('#execution=effect-run');await detail('recovery',/Resume stopped execution/);
    const resolution=deferred('POST /api/executions/effect-run/resolve');
    const resolving=node('inspector-content').children.find(child=>child.className==='detail-card').children.find(child=>child.textContent==='not applied').click();await resolution.requested;
    await select('#session=b');await detail('history',/History marker/);
    const historyReads=()=>calls.filter(item=>item.key==='/api/executions/b-run/history').length,beforeResolution=historyReads();
    if(success)resolution({resolved:true});else resolution.reject(new Error('old resolution failed'));
    await resolving;assert.equal(node('notice').textContent,'');assert.equal(historyReads(),beforeResolution,'An old resolution cannot reload a newer panel');
  }
  sessions.get('a').active_execution_id=null;await select('#session=a');
  const recoverySelection=deferred('GET /api/executions/a-run'),openingRecovery=node('open-recovery').click();await recoverySelection.requested;
  await select('#session=b');await detail('history',/History marker/);const beforeLateRecovery=calls.length;
  recoverySelection(info('a-run'));await openingRecovery;
  assert.ok(!calls.slice(beforeLateRecovery).some(call=>call.key==='/api/executions/b-run/history'),'A stale recovery selection cannot reload the new SDK drilldown');
  assert.equal(node('detail-view').value,'history');
  console.log('Navigation contracts passed: stale responses, current owner during paging/reopen and independent repeated paging');process.exit(0);
}
if(process.argv[3]==='races'){
  await select('#execution=a-run');
  const oldRefresh=deferred('GET /api/executions/a-run');node('refresh').click();await oldRefresh.requested;
  const loading=deferred('GET /api/executions/b-run'),loadingSelection=select('#execution=b-run');await loading.requested;
  oldRefresh(info('a-run'));await settle();assert.doesNotMatch(node('notice').textContent,/null|execution_id/);
  loading(info('b-run'));await loadingSelection;assert.match(node('execution-meta').textContent,/b-run/);assert.equal(node('notice').textContent,'');
  const activate=async id=>{
    sessions.get('a').active_execution_id=id;executions.set(id,{...info(id,'a'),status:'STARTED'});streamBlocks.set(id,{});
    await select('#session=a');
    if(node('inspector').hidden)node('inspector-toggle').click();node('detail-turn').value='a-run';await node('detail-turn').onchange();
    return streamBlocks.get(id).controller;
  };
  const snapshot=(controller,id)=>{controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get(id))+'\n\n'));controller.close();};
  let controller=await activate('owner-one');
  node('filter-form').requestSubmit();await until('running sidebar before terminal',()=>node('list').textContent.includes('Running'));
  const staleInfo={...executions.get('owner-one')},late=deferred('GET /api/executions/owner-one');
  node('refresh').click();await late.requested;delays.delete('GET /api/executions/owner-one');
  const staleList=deferred('GET /api/sessions');node('filter-form').requestSubmit();await staleList.requested;
  const listReadsBefore=calls.filter(call=>call.key==='/api/sessions').length;
  node('list').children.find(item=>item.textContent.includes('Beta')).focus();
  executions.get('owner-one').status='CANCELLED';sessions.get('a').active_execution_id=null;snapshot(controller,'owner-one');await until('released session after terminal snapshot',()=>node('current-status').textContent==='Session available');
  assert.doesNotMatch(node('list').textContent,/Running/,'Canonical terminal release updates the sidebar without Refresh');
  assert.match(document.activeElement?.textContent || '',/Beta/,'Background completion preserves the focused sidebar row');
  assert.equal(calls.filter(call=>call.key==='/api/sessions').length,listReadsBefore,'Projecting accepted session metadata adds no list request');
  const rowsBefore=node('list').children;node('prompt').focus();
  staleList({items:[{...sessions.get('a'),active_execution_id:'owner-one'},sessions.get('b')],next_cursor:null});await until('stale list response rendered',()=>node('list').children!==rowsBefore);
  assert.doesNotMatch(node('list').children.find(item=>item.className.includes('active')).textContent,/Running/,'A late older list read cannot overwrite accepted session release');
  assert.equal(document.activeElement,node('prompt'),'Background list refresh does not steal composer focus');
  const newOwnerList=deferred('GET /api/sessions');node('filter-form').requestSubmit();await newOwnerList.requested;
  newOwnerList({items:[{...sessions.get('a'),active_execution_id:'external-run'},sessions.get('b')],next_cursor:null});await until('newer same-revision list owner visible',()=>node('list').textContent.includes('Running'));
  const renamedList=deferred('GET /api/sessions');node('filter-form').requestSubmit();await renamedList.requested;
  renamedList({items:[{...sessions.get('a'),revision:1,metadata:{title:'Renamed elsewhere'},active_execution_id:'external-run'},sessions.get('b')],next_cursor:null});await until('newer revision list title visible',()=>node('list').textContent.includes('Renamed elsewhere'));
  late(staleInfo);await settle();assert.equal(node('cancel').hidden,true);assert.equal(node('end-active').hidden,true);assert.equal(node('current-status').textContent,'Session available');

  // Snapshot delivery and response EOF are distinct; an older Refresh cannot undo terminal truth.
  controller=await activate('owner-held');node('show-active').click();await until('selected active owner details',()=>node('execution-meta').textContent.includes('owner-held') && !node('execution-meta').textContent.includes('Loading'));
  const started={...executions.get('owner-held')},refresh=deferred('GET /api/executions/owner-held');node('refresh').click();await refresh.requested;
  executions.get('owner-held').status='CANCELLED';controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('owner-held'))+'\n\n'));await until('terminal snapshot before EOF',()=>node('execution-meta').textContent.includes('CANCELLED'));
  refresh(started);await settle();assert.equal(node('cancel').hidden,true);assert.match(node('execution-meta').textContent,/CANCELLED/);assert.doesNotMatch(node('current-status').textContent,/STARTED/);
  sessions.get('a').active_execution_id=null;controller.close();await until('completed terminal readback',()=>node('connection').textContent==='Observation: Complete');

  controller=await activate('owner-two');
  executions.set('owner-three',{...info('owner-three','a'),status:'STARTED'});streamBlocks.set('owner-three',{});
  executions.get('owner-two').status='SUCCEEDED';sessions.get('a').active_execution_id='owner-three';controller.error(new Error('closed without snapshot'));await until('next owner observation',()=>Boolean(streamBlocks.get('owner-three').controller));
  controller=streamBlocks.get('owner-three').controller;assert.ok(controller);assert.equal(node('cancel').hidden,false);assert.match(node('cancel').getAttribute('aria-label'),/owner-three/);
  assert.equal(node('end-active').hidden,false);assert.equal(calls.filter(item=>item.method==='POST').length,0);
  for(const success of [false,true]){
    const end=deferred('POST /api/executions/owner-three/cancel'),ending=node('end-active').click();await end.requested;
    const detail=deferred('GET /api/executions/a-run/models');tab('models').click();await detail.requested;detail.reject(new Error('new detail error'));await until('new detail error',()=>node('notice').textContent.includes('new detail error'));
    const latest=node('notice').textContent;assert.match(latest,/new detail error/);
    if(success){executions.get('owner-three').status='CANCELLED';sessions.get('a').active_execution_id=null;end({execution_id:'owner-three',cancelled:true});}
    else end.reject(new Error('old End response lost'));
    await ending;assert.equal(node('notice').textContent,latest);tab('overview').click();
  }
  snapshot(controller,'owner-three');await until('completed next owner readback',()=>node('connection').textContent==='Observation: Complete');

  controller=await activate('owner-four');
  executions.get('owner-four').status='CANCELLED';networkFailures.add('GET /api/sessions/a');snapshot(controller,'owner-four');await until('failed terminal session readback',()=>node('connection').textContent.includes('read unavailable'));
  assert.match(node('connection').textContent,/read unavailable/);assert.equal(node('connection').hidden,false);
  networkFailures.clear();streamBlocks.delete('owner-four');executions.set('owner-five',{...info('owner-five','a'),status:'STARTED'});streamBlocks.set('owner-five',{});sessions.get('a').active_execution_id='owner-five';
  await until('observation after Runtime read recovery',()=>Boolean(streamBlocks.get('owner-five').controller));controller=streamBlocks.get('owner-five').controller;assert.ok(controller);assert.match(node('cancel').getAttribute('aria-label'),/owner-five/);
  // A finite nonterminal watch is reattached, while a recovery boundary remains explicit.
  snapshot(controller,'owner-five');streamBlocks.set('owner-five',{});await until('reattached finite nonterminal observation',()=>Boolean(streamBlocks.get('owner-five').controller));
  assert.notEqual(streamBlocks.get('owner-five').controller,controller);controller=streamBlocks.get('owner-five').controller;assert.ok(controller);
  executions.get('owner-five').status='RECOVERY_REQUIRED';snapshot(controller,'owner-five');await until('explicit recovery boundary',()=>node('connection').textContent.includes('Recovery required'));
  const watches=calls.filter(item=>item.key==='/api/executions/owner-five/events').length;await new Promise(resolve=>setTimeout(resolve,650));
  assert.equal(calls.filter(item=>item.key==='/api/executions/owner-five/events').length,watches);assert.match(node('connection').textContent,/Recovery required/);
  // Successful competing readbacks must not turn a final snapshot into a reconnect.
  const partialId='partial-cancel',partialText='Received partial content stays incomplete';
  const previousTurn=timeline('a').items[0];
  sessions.get('a').active_execution_id=partialId;
  executions.set(partialId,{...info(partialId,'a'),status:'STARTED'});
  timelineOverrides.set('a',{items:[previousTurn,{execution_id:partialId,status:'STARTED',created_at:'2026-01-02T00:00:00Z',user_input:'slow',conversation_committed:false,items:[]}],next_cursor:null});
  streamBlocks.set(partialId,{});await select('#session=a&execution=a-run');
  node('chat-mode').click();controller=streamBlocks.get(partialId).controller;
  for(const [event_type,payload] of [['ASSISTANT_TEXT_DELTA',{text:partialText}],['ASSISTANT_THINKING_DELTA',{text:'Partial thinking'}]]){
    controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:partialId,depth:0,event:{event_type,payload}}})+'\n\n'));
  }
  await until('partial live text and thinking',()=>node('live').textContent.includes(partialText) && node('live').textContent.includes('Partial thinking'));assert.match(node('live').textContent,/Received partial content stays incomplete/);
  const stop=deferred(`POST /api/executions/${partialId}/cancel`);
  const stopping=node('cancel').click();await stop.requested;
  executions.get(partialId).status='CANCELLED';sessions.get('a').active_execution_id=null;
  timelineOverrides.get('a').items[1].status='CANCELLED';
  const finishFinalRead=deferred('GET /api/sessions/a');
  snapshot(controller,partialId);await finishFinalRead.requested;delays.delete('GET /api/sessions/a');
  stop({execution_id:partialId,cancelled:true});await stopping;
  finishFinalRead({session:sessions.get('a'),timeline:timeline('a')});await until('terminal observation after competing Stop readback',()=>node('connection').textContent==='Observation: Complete');
  assert.match(node('live').textContent,/Received partial content stays incomplete/);
  assert.match(node('live').textContent,/Partial thinking/);
  assert.match(node('live').textContent,/incomplete/);
  assert.equal(node('current-status').textContent,'Session available');
  assert.match(node('notice').textContent,/Cancellation confirmed.*Session released/);
  assert.equal(calls.filter(item=>item.key===`/api/executions/${partialId}/events`).length,1);
  console.log('Observation and End notice race contracts passed');process.exit(0);
}

if(process.argv[3]==='workspace'){
  const descendants=node=>[node,...node.children.flatMap(child=>child.children?descendants(child):[])];
  const markdown='## Question\n\n- **bold**\n\n```js\nconst x = "<tag>";\n```';
  const old=timeline('a').items[0];
  timelineOverrides.set('a',{items:[{...old,user_input:{version:1,prompt:{kind:'text',text:markdown},files:[{path:'src/example.py',size:12,media_type:'text/plain'}],attachments:[]},items:[{item_kind:'assistant',content:'| A | B |\n| --- | --- |\n| 1 | 2 |'}]}],next_cursor:null});
  await select('#session=a');
  const rendered=descendants(node('conversation')).filter(item=>item.className==='markdown').map(item=>item.innerHTML).join('');
  assert.match(rendered,/<h2>Question/);assert.match(rendered,/<table>/);
  assert.match(node('conversation').textContent,/src\/example.py/);
  let copied;Object.defineProperty(globalThis,'navigator',{value:{clipboard:{writeText:async value=>{copied=value;}}},configurable:true});
  await descendants(node('conversation')).find(item=>item.textContent==='Copy Markdown' && item.tagName==='BUTTON').click();assert.equal(copied,markdown);
  node('prompt').value='Unsent **Alpha**';node('files').value='src/a.py';node('planning').checked=true;node('memory').value='alpha-memory';
  await select('#session=b');assert.equal(node('prompt').value,'');node('prompt').value='Beta draft';
  await select('#session=a');
  assert.equal(node('prompt').value,'Unsent **Alpha**');assert.equal(node('files').value,'src/a.py');assert.equal(node('planning').checked,true);assert.equal(node('memory').value,'alpha-memory');
  const rejected=deferred('POST /api/sessions/a/messages'),rejectedSend=node('composer').requestSubmit();await rejected.requested;
  rejected({code:'SESSION_BUSY'},409);await rejectedSend;
  assert.equal(node('prompt').value,'Unsent **Alpha**');assert.equal(node('files').value,'src/a.py');
  const sends=calls.filter(item=>item.key==='/api/sessions/a/messages').length;
  sessions.get('a').active_execution_id='a-live';executions.set('a-live',{...info('a-live','a'),status:'STARTED'});
  timelineOverrides.set('a',{items:[old,{...old,execution_id:'a-live',status:'STARTED',conversation_committed:false,user_input:'Current question',items:[]}],next_cursor:null});
  streamBlocks.set('a-live',{});await select('#session=a');
  const observations=()=>calls.filter(item=>item.key==='/api/executions/a-live/events');
  if(node('inspector').hidden)node('inspector-toggle').click();node('detail-turn').value='a-run';await node('detail-turn').onchange();
  assert.match(node('execution-meta').textContent,/a-run/);assert.equal(node('cancel').hidden,false);assert.equal(node('end-active').hidden,false);
  let controller=streamBlocks.get('a-live').controller;
  const delta=(value,cursor='cursor-one')=>controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',cursor,item:{execution_id:'a-live',agent_id:'default',depth:0,event:{event_type:'ASSISTANT_TEXT_DELTA',payload:{text:value}}}})+'\n\n'));
  delta('**Live');await until('initial live response',()=>node('live').textContent.includes('Live'));assert.match(node('live').textContent,/Live/);
  const observed=observations().length;
  view('metrics').click();await hashChanged;assert.equal(node('metrics-view').hidden,false);
  delta(' response**');await until('background live response',()=>node('live').textContent.includes('Live response'));view('sessions').click();await hashChanged;
  assert.equal(node('metrics-view').hidden,true);assert.equal(observations().length,observed);
  assert.match(node('live').textContent,/Live response/);assert.match(node('execution-meta').textContent,/a-run/);
  assert.equal(node('prompt').value,'Unsent **Alpha**');assert.equal(calls.filter(item=>item.key==='/api/sessions/a/messages').length,sends);
  assert.ok(descendants(node('live')).some(item=>item.className==='markdown' && item.innerHTML.includes('<strong>Live response</strong>')));
  // A failed canonical read during disconnection cannot end the reconnect loop.
  const failedRead=deferred('GET /api/executions/a-live');controller.error(new Error('transport interrupted'));await failedRead.requested;
  failedRead.reject(new Error('Runtime temporarily unavailable'));await until('failed canonical reconnect read',()=>node('connection').textContent.includes('read unavailable'));
  streamBlocks.set('a-live',{});await until('reattached interrupted stream',()=>Boolean(streamBlocks.get('a-live').controller));
  controller=streamBlocks.get('a-live').controller;assert.ok(controller);assert.equal(observations().at(-1).query.cursor,'cursor-one');
  delta('Reattached **response**');await until('reattached live response',()=>node('live').textContent.includes('Reattached response'));assert.match(node('live').textContent,/Reattached response/);assert.doesNotMatch(node('live').textContent,/Live response/);
  assert.equal(calls.filter(item=>item.method==='POST' && item.key.endsWith('/recover')).length,0);
  // Owner actions stay on the current turn while the inspector shows an old one.
  const stop=deferred('POST /api/executions/a-live/cancel'),stopping=node('cancel').click();node('cancel').click();await stop.requested;
  assert.equal(calls.filter(item=>item.key==='/api/executions/a-live/cancel').length,1);
  executions.get('a-live').status='CANCELLING';stop({execution_id:'a-live',cancelled:false});await stopping;
  assert.match(node('execution-meta').textContent,/a-run/);assert.match(node('notice').textContent,/terminal outcome is not yet confirmed/);
  globalThis.confirm=()=>false;await node('end-active').click();assert.equal(calls.filter(item=>item.key==='/api/executions/a-live/recover').length,0);
  globalThis.confirm=()=>true;const recover=deferred('POST /api/executions/a-live/recover'),recovering=node('end-active').click();await recover.requested;
  executions.get('a-live').status='CANCELLED';sessions.get('a').active_execution_id=null;recover({execution_id:'a-live'});await recovering;
  assert.match(node('notice').textContent,/ended and session released/);assert.match(node('execution-meta').textContent,/a-run/);
  controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('a-live'))+'\n\n'));controller.close();streamBlocks.delete('a-live');await until('completed stopped live stream',()=>node('connection').textContent==='Observation: Complete');
  view('executions').click();await hashChanged;assert.equal(node('conversation-column').hidden,true);assert.equal(node('execution-list').hidden,false);assert.equal(node('browse-panel').parentElement,node('execution-list'));assert.equal(node('execution-live').textContent,'');
  assert.equal(node('composer').hidden,true);view('sessions').click();await hashChanged;assert.equal(node('browse-panel').parentElement,node('sidebar-browse'));assert.equal(node('prompt').value,'Unsent **Alpha**');
  // Successful admission while another page is open must not resurrect its prompt.
  node('prompt').value='Accepted while browsing';const admitted=deferred('POST /api/sessions/a/messages');
  const admission=node('composer').requestSubmit();await admitted.requested;view('metrics').click();await hashChanged;
  admitted({execution_id:'a-run'});await admission;view('sessions').click();await hashChanged;assert.equal(node('prompt').value,'');
  node('prompt').value='Accepted before next edit';const nextAdmission=deferred('POST /api/sessions/a/messages');
  const nextSend=node('composer').requestSubmit();await nextAdmission.requested;view('executions').click();await hashChanged;node('prompt').value='Newer draft';
  nextAdmission({execution_id:'a-run'});await nextSend;view('sessions').click();await hashChanged;assert.equal(node('prompt').value,'Newer draft');
  console.log('Workspace contracts passed: Markdown, copy source, drafts, inspection, Metrics, reattachment and active-owner controls');process.exit(0);
}

sessions.get('a').active_execution_id='active-other';executions.set('active-other',{...info('active-other','a'),status:'STARTED'});
await select('#session=a&execution=a-run');
assert.match(node('current-status').textContent,/Session occupied/);assert.equal(node('show-active').hidden,false);
assert.doesNotMatch(node('current-status').textContent,/SUCCEEDED/);
node('show-active').click();await until('active execution details',()=>node('execution-meta').textContent.includes('active-other') && !node('execution-meta').textContent.includes('Loading'));assert.match(node('current-status').textContent,/Session occupied · STARTED/);
assert.equal(node('inspector').hidden,false);assert.equal(node('cancel').hidden,false);
node('inspector-close').click();assert.equal(node('cancel').hidden,false);
sessions.get('a').active_execution_id=null;await select('#session=a');

// A session response cannot overwrite a newer navigation.
const releaseA=deferred('GET /api/sessions/a');
const oldSelection=select('#session=a');await releaseA.requested;await select('#session=b');
releaseA({session:sessions.get('a'),timeline:timeline('a')});await oldSelection;
assert.equal(node('conversation-title').textContent,'Beta');

// Changing inspector tabs invalidates the previous request.
const releaseModels=deferred('GET /api/executions/b-run/models');
tab('models').click();await releaseModels.requested;await detail('history',/History marker/);
releaseModels({items:[{model_request_seq:999,status:'SUCCEEDED',model:{},execution_id:'b-run'}],next_cursor:null});await settle();
assert.match(node('inspector-content').textContent,/History marker/);
assert.doesNotMatch(node('inspector-content').textContent,/999/);

// A child trace read cannot apply its selectors after the user changes selection.
detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'trace-child',step_event_seq:1,payload:{agent_run_seq:7,model_request_seq:9}}],next_cursor:null});
await detail('trace',/Read SDK content/);
const readChild=node('inspector-content').children[0].children[1].children.find(child=>child.tagName==='BUTTON' && child.textContent==='Read SDK content');
const releaseTraceChild=deferred('GET /api/executions/trace-child');
const childRead=readChild.click();await releaseTraceChild.requested;await select('#session=a');await detail('models',/No records|No durable detail/);
const afterTraceNavigation=calls.length;
releaseTraceChild(info('trace-child','b'));await childRead;
assert.equal(location.hash,'#session=a');
assert.equal(node('detail-view').value,'models');
assert.ok(!calls.slice(afterTraceNavigation).some(call=>call.key==='/api/executions/a-run/history'));
detailResponses.delete('/api/executions/b-run/trace');

for(const destination of ['tab','execution','current']){
  await select('#session=b');
  detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'trace-child',step_event_seq:1,payload:{agent_run_seq:7,model_request_seq:9}}],next_cursor:null});
  await detail('trace',/Read SDK content/);
  const read=node('inspector-content').children[0].children[1].children.find(child=>child.tagName==='BUTTON' && child.textContent==='Read SDK content');
  const release=deferred('GET /api/executions/trace-child'),childRead=read.click();await release.requested;
  if(destination==='tab')await detail('models',/No records|No durable detail/);
  if(destination==='execution')await select('#session=b&execution=b-run');
  const beforeReadback=calls.length;release(info('trace-child','b'));await childRead;
  const content=calls.slice(beforeReadback).filter(call=>call.key.endsWith('/history'));
  if(destination==='current'){
    assert.equal(content.length,1);assert.equal(content[0].key,'/api/executions/trace-child/history');
    assert.equal(content[0].query.agent_run_seq,'7');assert.equal(content[0].query.model_request_seq,'9');
  }else{
    assert.equal(content.length,0,destination);
    if(destination==='tab')assert.equal(node('detail-view').value,'models');
    else assert.match(node('execution-meta').textContent,/b-run/);
  }
  detailResponses.delete('/api/executions/b-run/trace');
}

// A pending execution action cannot navigate away from a newer selection.
await select('#session=a');
const releaseRetry=deferred('POST /api/executions/a-run/retry');
const retry=node('retry').click();await releaseRetry.requested;await select('#session=b');
releaseRetry({execution_id:'a-retried'});await retry;
assert.equal(location.hash,'#session=b');
assert.equal(node('conversation-title').textContent,'Beta');

// While metadata for a newly selected execution is loading, old controls are disabled.
const releaseExecution=deferred('GET /api/executions/a-run'),executionSelection=select('#session=a');
await releaseExecution.requested;
assert.equal(node('retry').disabled,true);
releaseExecution(info('a-run'));await executionSelection;assert.equal(node('retry').disabled,false);

// An uncertain fork preserves both destination identity and idempotency key.
await node('fork-session').click();await hashChanged;
await node('fork-session').click();await hashChanged;
const forks=calls.filter(call=>call.key==='/api/sessions/a/fork');
assert.equal(forks.length,2);
assert.equal(forks[0].body.new_session_id,forks[1].body.new_session_id);
assert.equal(forks[0].body.request_id,forks[1].body.request_id);

// Dismissing an in-flight creation cannot close or navigate a newer dialog.
const releaseCreate=deferred('POST /api/sessions');
node('new-session').click();node('new-title').value='<img src=x onerror=alert(1)>';
node('new-agent').value='default';const create=node('new-form').requestSubmit();await releaseCreate.requested;
const oldSession=node('new-dialog').dataset.sessionId;
node('new-dialog').close();node('new-session').click();const newerSession=node('new-dialog').dataset.sessionId;
const priorHash=location.hash;releaseCreate({session_id:oldSession});await create;
assert.equal(node('new-dialog').open,true);assert.equal(node('new-dialog').dataset.sessionId,newerSession);assert.equal(location.hash,priorHash);
node('new-dialog').close();

// Repeated submit during one in-flight request invokes start only once.
await select('#session=a');
const releaseSend=deferred('POST /api/sessions/a/messages');node('prompt').value='Repeated message';
const send=node('composer').requestSubmit();node('composer').requestSubmit();await releaseSend.requested;
assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').length,1);
node('prompt').value='Next draft';releaseSend({execution_id:'a-run'});await send;assert.equal(node('prompt').value,'Next draft');
assert.equal(node('inspector').hidden,true,'Send must preserve the conversation-only view');
for(const open of [false,true]){
  if(node('inspector').hidden===open)node('inspector-toggle').click();
  const renamed=deferred('POST /api/sessions/a/update'),rename=node('rename-session').click();await renamed.requested;renamed({session_id:'a'});await rename;
  assert.equal(node('inspector').hidden,!open,'Rename must preserve the panel choice');
}
node('inspector-close').click();

// Known admission rejection gets a new key only on the next explicit send.
for(const code of ['SESSION_BUSY','SESSION_CONFLICT']) {
  node('prompt').value=`Rejected ${code}`;
  const rejected=deferred('POST /api/sessions/a/messages'),send=node('composer').requestSubmit();
  await rejected.requested;
  rejected({code,operation_id:'web',safe_details:{reason:'<script>safe detail</script>'},exception_message:'private diagnostic'},409);await send;
  const first=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  assert.equal(node('prompt').value,`Rejected ${code}`);
  assert.match(node('notice').textContent,/this message was not started/);
  assert.match(node('notice').textContent,/POST \/api\/session\/messages/);
  assert.match(node('notice').textContent,/<script>safe detail<\/script>/);
  assert.doesNotMatch(node('notice').textContent,/private diagnostic/);
  assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1),first);
  const retried=deferred('POST /api/sessions/a/messages'),retry=node('composer').requestSubmit();
  await retried.requested;
  const next=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  assert.notEqual(next.body.request_id,first.body.request_id);
  assert.equal(next.body.prompt,first.body.prompt);
  retried({execution_id:'a-run'});await retry;
}

// Unknown commit and network outcomes preserve the identity of the original send.
for(const failure of ['STORAGE_COMMIT_UNKNOWN','network']) {
  node('prompt').value=`Uncertain ${failure}`;
  const uncertain=deferred('POST /api/sessions/a/messages'),send=node('composer').requestSubmit();
  await uncertain.requested;
  const first=calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1);
  if(failure==='network')uncertain.reject(new TypeError('Network unavailable'));
  else uncertain({code:failure},503);
  await send;
  const repeated=deferred('POST /api/sessions/a/messages'),retry=node('composer').requestSubmit();
  await repeated.requested;
  assert.equal(calls.filter(call=>call.key==='/api/sessions/a/messages').at(-1).body.request_id,first.body.request_id);
  repeated({execution_id:'a-run'});await retry;
}

// A rejected send cannot place its error in a newer conversation.
node('prompt').value='Old selection draft';
const oldSend=deferred('POST /api/sessions/a/messages');
const oldSubmission=node('composer').requestSubmit();await oldSend.requested;await select('#session=b');
oldSend({code:'SESSION_BUSY'},409);await oldSubmission;
assert.equal(node('conversation-title').textContent,'Beta');assert.equal(node('notice').textContent,'');
await select('#session=a');

// Refreshing an old turn must not clear a new live turn selected in the same session.
executions.set('a-new',{...info('a-new'),status:'RUNNING'});
const older=timeline('a').items[0];
streamBlocks.set('a-new',{});sessions.get('a').active_execution_id='a-new';
timelineOverrides.set('a',{items:[older,{...older,execution_id:'a-new',user_input:'new question',conversation_committed:false,items:[]}],next_cursor:null});
await select('#session=a&execution=a-run');
const releaseRefresh=deferred('GET /api/sessions/a');node('refresh').click();await releaseRefresh.requested;
const newTurn=node('conversation').children.find(child=>child.textContent.includes('new question'));
if(node('inspector').hidden)node('inspector-toggle').click();node('detail-turn').value='a-new';await node('detail-turn').onchange();await until('live stream opened',()=>Boolean(streamBlocks.get('a-new').controller));
const stream=streamBlocks.get('a-new');
stream.controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:'a-new',agent_id:'default',depth:0,event:{event_type:'ASSISTANT_TEXT_DELTA',payload:{text:'New execution text'}}},cursor:'resume-cursor'})+'\n\n'));
await until('live assistant text',()=>node('live').textContent.includes('New execution text'));assert.match(node('live').textContent,/New execution text/);
const liveEvent=(event_type,seq,depth=0,payload={})=>stream.controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:depth?'child':'a-new',agent_id:'default',depth,event:{event_type,durable_seq:seq,payload}},cursor:null})+'\n\n'));
liveEvent('TOOL_CALL_STARTED',1);await until('tool start event',()=>node('live-activity').textContent.includes('tool call started'));assert.equal(node('live-activity').open,false);
node('live-activity').open=true;node('live-activity').ontoggle();node('live-activity').children[0].focus();liveEvent('MODEL_REQUEST_STARTED',2);await until('model start event',()=>node('live-activity').textContent.includes('model request started'));assert.equal(node('live-activity').open,true);assert.equal(document.activeElement,node('live-activity').children[0]);
liveEvent('EXECUTION_FAILED',3,1);
liveEvent('MODEL_REQUEST_FINISHED',4,0,{agent_run_seq:1,model_request_seq:1,status:'FAILED'});
liveEvent('TOOL_CALL_FINISHED',5,0,{agent_run_seq:1,call_id:'failed-call',tool_name:'query',status:'FAILED',error_code:'TOOL_RETRY_REQUIRED'});
liveEvent('MODEL_REQUEST_FINISHED',6,0,{agent_run_seq:1,model_request_seq:2,status:'CANCELLED'});
liveEvent('TOOL_CALL_FINISHED',7,0,{agent_run_seq:1,call_id:'successful-call',tool_name:'query',status:'SUCCEEDED'});
for(const [index,event] of ['EXECUTION_START_UNKNOWN','EXECUTION_RECOVERY_REQUIRED','CANCEL_REQUESTED','EXECUTION_CANCELLED'].entries())liveEvent(event,8+index);
await until('final event in stream batch',()=>node('live').textContent.includes('execution cancelled'));
for(const name of ['execution failed','model request finished · FAILED','tool call finished · FAILED · TOOL_RETRY_REQUIRED','model request finished · CANCELLED','execution start unknown','execution recovery required','cancel requested','execution cancelled'])assert.ok(node('live').children.some(child=>child.className.includes('live-event') && child.textContent.includes(name)),name);
assert.ok(node('live-activity').children.some(child=>child.className.includes('live-event') && child.textContent.includes('tool call finished · SUCCEEDED')));
assert.ok(!node('live').children.some(child=>child.className.includes('live-event') && child.textContent.includes('SUCCEEDED')));
const focusedChild=node('live').children.find(child=>child.className.includes('live-event') && child.textContent.includes('execution failed')).children.find(child=>child.tagName==='BUTTON');focusedChild.focus();liveEvent('TOOL_CALL_STARTED',12,0,{agent_run_seq:1,call_id:'next-call'});await until('next tool event',()=>node('live-activity').textContent.includes('next-call'));assert.equal(document.activeElement.dataset.liveFocus,focusedChild.dataset.liveFocus);
releaseRefresh({session:sessions.get('a'),timeline:timeline('a')});await settle();
assert.match(node('live').textContent,/New execution text/);assert.equal(node('live-activity').open,true);
const focusBeforeRefresh=document.activeElement.dataset.liveFocus;
await refresh();assert.equal(node('live-activity').open,true);
assert.equal(document.activeElement.dataset.liveFocus,focusBeforeRefresh);
// A disconnected running observation reconnects from the last Runtime cursor.
const disconnected=stream.controller;
disconnected.close();
await until('reconnected live stream',()=>stream.controller!==disconnected);
assert.equal(calls.filter(call=>call.key==='/api/executions/a-new/events').at(-1).query.cursor,'resume-cursor');
stream.controller.enqueue(new TextEncoder().encode('data: '+JSON.stringify({type:'event',item:{execution_id:'a-new',agent_id:'default',depth:0,event:{event_type:'ASSISTANT_TEXT_DELTA',payload:{text:'Reconnected execution text'}}},cursor:'next-cursor'})+'\n\n'));
await until('reconnected assistant text',()=>node('live').textContent.includes('Reconnected execution text'));
assert.doesNotMatch(node('live').textContent,/New execution text/);
stream.controller.close();streamBlocks.delete('a-new');timelineOverrides.delete('a');sessions.get('a').active_execution_id=null;
executions.set('a-new',{...info('a-new'),status:'SUCCEEDED'});
await until('closed stream canonical readback',()=>node('connection').textContent==='Observation: SUCCEEDED');

// A late session read cannot re-open the conversation pane over Metrics.
const releaseAgain=deferred('GET /api/sessions/a'),oldMetricsSelection=select('#session=a');await releaseAgain.requested;const metricRows=node('metric-results').children;view('metrics').click();await until('metric results',()=>node('metric-results').children!==metricRows);
releaseAgain({session:sessions.get('a'),timeline:timeline('a')});await oldMetricsSelection;
assert.equal(node('metrics-view').hidden,false);assert.equal(node('conversation-view').hidden,true);
assert.equal(node('metric-name').tagName,'INPUT');
executions.set('orphan',{...info('orphan',null),status:'RECOVERY_REQUIRED'});await select('#execution=orphan');
assert.equal(node('composer').hidden,true);assert.equal(node('inspector').hidden,false);assert.equal(document.activeElement,node('execution-meta'));assert.equal(node('inspector-close').hidden,true);assert.equal(node('stop-run').hidden,false);assert.ok(node('execution-live'));

// Status alone never initiates cleanup; one stopped-executor confirmation gates it.
const recoveries=()=>calls.filter(call=>call.key==='/api/executions/orphan/recover');
const cancellations=()=>calls.filter(call=>call.key==='/api/executions/orphan/cancel');
await detail('recovery',/Resume stopped execution/);
for(const status of ['START_UNKNOWN','WAITING_DEFERRED','WAITING_RETRY','FINALIZING','SUCCEEDED']){
  executions.set('orphan',{...info('orphan',null),status});await refresh();
  assert.equal(node('resume-run').disabled,true,status);
}
for(const status of ['PENDING_START','STARTED','CANCELLING','RECOVERY_REQUIRED']){
  executions.set('orphan',{...info('orphan',null),status});await refresh();
  assert.equal(node('end-stopped').disabled,false);
  assert.equal(node('resume-run').disabled,false,status);
  assert.equal(recoveries().length,0);assert.equal(cancellations().length,0);
}
let recoveryConfirmation='';
globalThis.confirm=message=>{recoveryConfirmation=message;return false;};
await node('end-stopped').click();
assert.match(recoveryConfirmation,/previous executor has stopped/);assert.equal(recoveries().length,0);assert.equal(cancellations().length,0);
globalThis.confirm=()=>true;
const resetEnd=async(mode={},status='STARTED')=>{
  endMode=mode;sessions.get('a').active_execution_id='orphan';
  executions.set('orphan',{...info('orphan','a'),status});await select('#execution=orphan');
  await until('stopped execution status',()=>node('execution-meta').textContent.includes(status));
};
await resetEnd();
const releaseEndCancel=deferred('POST /api/executions/orphan/cancel');
const end=node('end-stopped').click();node('end-stopped').click();await releaseEndCancel.requested;
assert.equal(cancellations().length,1);assert.equal(node('end-stopped').disabled,true);
for(const id of ['stop-run','retry','fork-run'])assert.equal(node(id).disabled,true);
executions.get('orphan').status='CANCELLING';releaseEndCancel({execution_id:'orphan',cancelled:false});await end;
assert.equal(recoveries().length,1);assert.equal(executions.get('orphan').status,'CANCELLED');
assert.equal(sessions.get('a').active_execution_id,null);assert.match(node('notice').textContent,/ended and session released/);
assert.equal(calls.filter(call=>call.key==='/api/sessions/a' && call.query.include_timeline==='false').length>0,true);

// Unknown cancellation stays on the same key; a committed cancellation skips resend.
for(const commitCancel of [false,true]){
  await resetEnd({cancelError:'network outcome unknown',commitCancel});
  const start=cancellations().length,recoverStart=recoveries().length;
  await node('end-stopped').click();
  assert.equal(recoveries().length,recoverStart);assert.match(node('notice').textContent,/network outcome unknown/);
  await node('end-stopped').click();
  const sent=cancellations().slice(start);assert.equal(sent.length,commitCancel?1:2);
  if(!commitCancel)assert.equal(sent[0].body.request_id,sent[1].body.request_id);
  assert.match(node('notice').textContent,/ended and session released/);
}
await resetEnd({keepRecovery:true},'RECOVERY_REQUIRED');const beforeRecoveryRequired=recoveries().length;
assert.equal(node('end-stopped').hidden,false);assert.match(node('end-stopped').textContent,/free session/);
await node('end-stopped').click();
assert.equal(recoveries().length,beforeRecoveryRequired+1);assert.match(node('notice').textContent,/ended and session released/);

// Interruption after durable cancel is safe: a fresh attempt reads CANCELLING first.
await resetEnd({},'CANCELLING');const beforeResumeCancel=cancellations().length;
await node('end-stopped').click();assert.equal(cancellations().length,beforeResumeCancel);
assert.match(node('notice').textContent,/ended and session released/);

await resetEnd({recoverError:'recovery reply lost'});const beforeUnknownRecovery=recoveries().length;
await node('end-stopped').click();assert.equal(recoveries().length,beforeUnknownRecovery+1);
assert.match(node('notice').textContent,/recovery reply lost/);
// If that request completed, the next explicit attempt must not recover again.
executions.get('orphan').status='CANCELLED';sessions.get('a').active_execution_id=null;
await refresh();assert.equal(node('end-stopped').disabled,false);
await node('end-stopped').click();assert.equal(recoveries().length,beforeUnknownRecovery+1);
assert.match(node('notice').textContent,/ended and session released/);

// Uncertain recovery keeps its key, while Resume and End use separate operations.
await resetEnd({recoverError:'unknown end response'});
await node('end-stopped').click();const uncertainEnd=recoveries().at(-1);
await node('end-stopped').click();
assert.equal(recoveries().at(-1).body.request_id,uncertainEnd.body.request_id);
assert.match(node('notice').textContent,/ended and session released/);

await resetEnd({recoverError:'unknown end response'});await detail('recovery',/Resume stopped execution/);
await node('end-stopped').click();const endBeforeResume=recoveries().at(-1);
await node('resume-run').click();
assert.notEqual(recoveries().at(-1).body.request_id,endBeforeResume.body.request_id);

await resetEnd();await detail('recovery',/Resume stopped execution/);
const failedResume=deferred('POST /api/executions/orphan/recover');
const failedResuming=node('resume-run').click();await failedResume.requested;const resumeBeforeEnd=recoveries().at(-1);
failedResume.reject(new Error('unknown resume response'));await failedResuming;
await node('end-stopped').click();
assert.notEqual(recoveries().at(-1).body.request_id,resumeBeforeEnd.body.request_id);
assert.match(node('notice').textContent,/ended and session released/);

for(const [mode,status,message] of [
  [{blockCancel:true},'RECOVERY_REQUIRED',/STORAGE_CONFLICT/],
  [{keepCancelling:true},'STARTED',/still CANCELLING/],
  [{keepOwner:true},'STARTED',/session release is not confirmed/],
  [{effectError:true},'STARTED',/TOOL_EFFECT_OUTCOME_UNKNOWN/],
]){
  await resetEnd(mode,status);const before=recoveries().length;
  await node('end-stopped').click();assert.match(node('notice').textContent,message);
  assert.doesNotMatch(node('notice').textContent,/ended and session released/);
  if(mode.blockCancel)assert.equal(recoveries().length,before);
}
await resetEnd({directTerminal:true},'PENDING_START');const beforeDirect=recoveries().length;
await node('end-stopped').click();assert.equal(recoveries().length,beforeDirect);
assert.match(node('notice').textContent,/ended and session released/);
await resetEnd({nextOwner:'a-new'});await node('end-stopped').click();
assert.match(node('notice').textContent,/another execution now owns/);

// Failed owner readback remains retryable after a terminal refresh, without controls.
await resetEnd();const failedOwnerRead=deferred('GET /api/sessions/a');
const failedEnd=node('end-stopped').click();await failedOwnerRead.requested;failedOwnerRead.reject(new Error('owner read unavailable'));await failedEnd;
assert.match(node('notice').textContent,/owner read unavailable/);
await refresh();assert.equal(node('end-stopped').disabled,false);
const beforeOwnerRetry=recoveries().length;await node('end-stopped').click();
assert.equal(recoveries().length,beforeOwnerRetry);assert.match(node('notice').textContent,/ended and session released/);

// Auxiliary rendering failures do not overturn confirmed terminal/owner facts.
await resetEnd();endMode.listError=true;await node('end-stopped').click();
assert.match(node('notice').textContent,/ended and session released.*Display refresh failed: list unavailable/);
assert.doesNotMatch(node('notice').textContent,/No action was automatically resent/);

// Navigation interrupts later control writes and discards stale action errors.
await resetEnd();const staleEnd=deferred('POST /api/executions/orphan/cancel'),beforeStale=recoveries().length;
const oldEnd=node('end-stopped').click();await staleEnd.requested;await select('#session=b');const betaNotice=node('notice').textContent;
executions.get('orphan').status='CANCELLING';staleEnd({execution_id:'orphan',cancelled:false});await oldEnd;
assert.equal(recoveries().length,beforeStale);assert.equal(node('notice').textContent,betaNotice);

// Original resume capability remains a secondary action in the Recovery tab.
await resetEnd({},'STARTED');endMode=null;await detail('recovery',/Resume stopped execution/);
assert.match(node('inspector-content').textContent,/Resume stopped execution/);
globalThis.confirm=message=>{recoveryConfirmation=message;return false;};
await node('resume-run').click();assert.match(recoveryConfirmation,/may call models or tools/);
globalThis.confirm=()=>true;
const releaseRecovery=deferred('POST /api/executions/orphan/recover');const beforeLegacy=recoveries().length;
const resume=node('resume-run').click();node('resume-run').click();await releaseRecovery.requested;assert.equal(recoveries().length,beforeLegacy+1);
executions.set('orphan',{...info('orphan',null),status:'SUCCEEDED'});releaseRecovery({execution_id:'orphan'});await resume;await hashChanged;await until('resumed execution readback',()=>node('resume-run')?.disabled);
assert.equal(node('resume-run').disabled,true);tab('overview').click();

// A cancellation conflict reads state without resending; a new click reuses its key.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});await refresh();
await node('stop-run').click();
assert.equal(cancelAttempts,1);assert.match(node('notice').textContent,/Runtime status re-read: STARTED/);
assert.match(node('notice').textContent,/If still needed, choose Stop execution again/);
await node('stop-run').click();
assert.equal(cancelAttempts,2);
const cancels=cancellations().slice(-2);
assert.equal(cancels[0].body.request_id,cancels[1].body.request_id);
assert.match(node('notice').textContent,/terminal outcome is not yet confirmed/);

// Stop feedback follows canonical terminal and session-owner readback, not the request reply.
executions.get('orphan').session_id='a';sessions.get('a').active_execution_id='orphan';
executions.get('orphan').status='CANCELLED';await refresh();
assert.match(node('notice').textContent,/Cancellation confirmed.*release is not yet confirmed/);
sessions.get('a').active_execution_id='a-new';await refresh();
assert.match(node('notice').textContent,/Cancellation confirmed.*Another execution now owns/);
assert.equal(node('stop-run').hidden,true);

// Cancellation is a normal turn outcome rather than a raw API error.
const cancelledTurn={...timeline('a').items[0],status:'CANCELLED',error_code:'EXECUTION_CANCELLED',safe_error_details:{}};
timelineOverrides.set('a',{items:[cancelledTurn],next_cursor:null});await select('#session=a');
assert.match(node('conversation').textContent,/Execution cancelled\./);assert.doesNotMatch(node('conversation').textContent,/EXECUTION_CANCELLED|\{\}/);
timelineOverrides.delete('a');await select('#execution=orphan');

// A terminal snapshot also reconciles a pending Stop notice without manual refresh.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});sessions.get('a').active_execution_id='orphan';
const stopStream={};streamBlocks.set('orphan',stopStream);await refresh();
await node('stop-run').click();
executions.get('orphan').status='CANCELLED';sessions.get('a').active_execution_id=null;
stopStream.controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('orphan'))+'\n\n'));
stopStream.controller.close();streamBlocks.delete('orphan');await until('confirmed terminal Stop notice',()=>node('notice').textContent.includes('Session released'));
assert.match(node('notice').textContent,/Cancellation confirmed.*Session released/);

// Unrelated errors are not replaced by later cancellation readback.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});
const unrelatedStream={};streamBlocks.set('orphan',unrelatedStream);await refresh();
await node('stop-run').click();
const unrelated=deferred('GET /api/executions/orphan/models');tab('models').click();await unrelated.requested;
unrelated.reject(new Error('new detail error'));await until('new detail error after Stop',()=>node('notice').textContent.includes('new detail error'));
executions.get('orphan').status='CANCELLED';
const oldNotice=node('notice').textContent;
assert.match(oldNotice,/new detail error/);
tab('overview').click();
unrelatedStream.controller.enqueue(new TextEncoder().encode('event: snapshot\ndata: '+JSON.stringify(executions.get('orphan'))+'\n\n'));
unrelatedStream.controller.close();streamBlocks.delete('orphan');await until('terminal readback preserves unrelated notice',()=>node('connection').textContent==='Observation: Complete');
assert.equal(node('notice').textContent,oldNotice);tab('overview').click();

// A failed cancellation readback must not post its error into a newer selection.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});await refresh();
cancelAttempts=0;
const failedCancelRead=deferred('GET /api/executions/orphan');
const oldCancel=node('stop-run').click();await failedCancelRead.requested;
await select('#session=b');
const newerNotice=node('notice').textContent;
failedCancelRead.reject(new Error('old readback unavailable'));await oldCancel;
assert.equal(node('conversation-title').textContent,'Beta');
assert.equal(node('notice').textContent,newerNotice);

// A late failed Stop response cannot replace another conversation's notice.
executions.set('orphan',{...info('orphan',null),status:'STARTED'});await select('#execution=orphan');
const failedStop=deferred('POST /api/executions/orphan/cancel'),failedStopping=node('stop-run').click();await failedStop.requested;
await select('#session=b');const otherConversationNotice=node('notice').textContent;
failedStop.reject(new Error('old Stop response lost'));await failedStopping;
assert.equal(node('notice').textContent,otherConversationNotice);

// Newer detail errors own the notice even when Stop completes in the same view.
for(const success of [false,true]){
  executions.set('orphan',{...info('orphan',null),status:'STARTED'});await select('#execution=orphan');
  const lateStop=deferred('POST /api/executions/orphan/cancel'),lateStopping=node('stop-run').click();await lateStop.requested;
  const detailFailure=deferred('GET /api/executions/orphan/models');tab('models').click();await detailFailure.requested;
  detailFailure.reject(new Error('new detail error'));await until('current detail failure while Stop pending',()=>node('notice').textContent.includes('new detail error'));const ownedNotice=node('notice').textContent;
  if(success)lateStop({execution_id:'orphan',cancelled:false});else lateStop.reject(new Error('old Stop response lost'));
  await lateStopping;assert.equal(node('notice').textContent,ownedNotice);tab('overview').click();
}

// Failed terminal owner readback cannot report an old Stop into another conversation.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});await select('#execution=orphan');
await node('stop-run').click();executions.get('orphan').status='CANCELLED';
const staleOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await staleOwnerRead.requested;
await select('#session=b');const newOwnerNotice=node('notice').textContent;
staleOwnerRead.reject(new Error('old session release readback failed'));await settle();
assert.equal(node('notice').textContent,newOwnerNotice);

// A newer canonical release wins over a failed older owner readback.
executions.set('orphan',{...info('orphan','a'),status:'STARTED'});await select('#execution=orphan');
await node('stop-run').click();executions.get('orphan').status='CANCELLED';
const olderOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await olderOwnerRead.requested;
const newerOwnerRead=deferred('GET /api/sessions/a');node('refresh').click();await newerOwnerRead.requested;
newerOwnerRead({session:{...sessions.get('a'),active_execution_id:null},timeline:null});await until('newer canonical release readback',()=>node('notice').textContent.includes('Session released'));
assert.match(node('notice').textContent,/Cancellation confirmed.*Session released/);
const settledNotice=node('notice').textContent;
olderOwnerRead.reject(new Error('outdated owner readback failed'));await settle();
assert.equal(node('notice').textContent,settledNotice);

// Exact-ID lookup preserves opaque sessions and distinguishes standalone executions.
const opaque=' ../team/conversation?notes#你好 ';
sessions.set(opaque,{...sessions.get('a'),session_id:opaque,metadata:{title:'Opaque conversation'}});
timelineOverrides.set(opaque,{items:[],next_cursor:null});
node('open-kind').value='session';node('open-id').value=opaque;node('open-form').requestSubmit();await hashChanged;
assert.equal(new URLSearchParams(location.hash.slice(1)).get('session'),opaque);
assert.equal(node('conversation-title').textContent,'Opaque conversation');
node('open-kind').value='execution';node('open-id').value='b-run';node('open-form').requestSubmit();await hashChanged;
assert.equal(new URLSearchParams(location.hash.slice(1)).get('execution'),'b-run');assert.equal(node('page-title').textContent,'Executions');assert.equal(node('composer').hidden,true);

// A committed RUNNING identity can have an intentionally empty prompt envelope.
detailResponses.set('/api/executions/b-run/models',{items:[{execution_id:'b-run',agent_run_seq:1,depth:0,model_request_seq:1,purpose:'agent',status:'RUNNING',model:{},request:{},response:null,content_included:true}],next_cursor:null});
await detail('sdkmodels',/Prompt content is not available yet/);
assert.match(node('inspector-content').textContent,/Prompt content is not available yet/);
assert.doesNotMatch(node('inspector-content').textContent,/SDK prompt architecture/);

// Former history/trace summaries remain human-readable without an eager trace fetch.
const beforeTrace=calls.filter(call=>call.key.endsWith('/trace')).length;
detailResponses.set('/api/executions/b-run/models',{items:[{execution_id:'b-run',agent_run_seq:1,depth:0,model_request_seq:1,purpose:'agent',status:'SUCCEEDED',model:{model_name:'Recorded model'},duration_ns:1000,usage:{input_tokens:10,output_tokens:4,cache_read_tokens:3,cache_write_tokens:2},request:{messages:[{parts:[{part_kind:'system-prompt',content:'real system prompt'}]}],parameters:{instruction_parts:[{name:'workspace',content:'fixed',dynamic:false}],output_mode:'text'}},content_included:true}],next_cursor:null});
await detail('sdkmodels',/Recorded model · 1.000 us/);
assert.match(node('inspector-content').textContent,/Recorded model · 1.000 us/);
assert.match(node('inspector-content').textContent,/3 cache read \/ 2 cache write/);
assert.match(node('inspector-content').textContent,/SDK prompt architecture/);
assert.match(node('inspector-content').textContent,/System Prompt1 part\(s\) · ~18 chars/);
assert.match(node('inspector-content').textContent,/Fixed Instructions \(F0\/F1\)/);
assert.equal(calls.filter(call=>call.key.endsWith('/trace')).length,beforeTrace);
detailResponses.set('/api/executions/b-run/trace',{items:[{execution_id:'b-run',step_event_seq:1,payload:{kind:'MODEL_RESPONSE',status:'SUCCEEDED',scope:'root',agent_run_seq:1,step_index:0,model_request_seq:1,purpose:'agent',duration_ns:2000000,token_usage:{input_tokens:10,output_tokens:4}}}],next_cursor:null});
await detail('trace',/Read SDK content/);
assert.match(node('inspector-content').textContent,/request #1/);
assert.match(node('inspector-content').textContent,/agent · 2.000 ms · 10 in \/ 4 out/);

// Exact newest ordering is an explicit scan action, never a refresh side effect.
const executionRows=node('list').children;view('executions').click();await until('execution list',()=>node('list').children!==executionRows);
const scans=()=>calls.filter(call=>call.query.recent==='true').length;
assert.equal(scans(),0);
node('list-action').value='recent';node('list-action').onchange();
assert.equal(node('filter-session').disabled,true);
node('filter-form').requestSubmit();node('filter-form').requestSubmit();await until('explicit newest scan',()=>node('list-scope').textContent.includes('scanned visible execution metadata'));assert.equal(scans(),1);
assert.match(node('list-scope').textContent,/scanned visible execution metadata/);
await refresh();assert.equal(scans(),1);
assert.equal(node('list-action').value,'paged');assert.equal(node('filter-session').disabled,false);
node('filter-session').value=opaque;node('filter-form').requestSubmit();await until('session filter results',()=>node('execution-filter-summary').textContent.includes('1 applied'));
assert.equal(calls.filter(call=>call.key==='/api/executions').at(-1).query.session_id,opaque);assert.match(node('execution-filter-summary').textContent,/1 applied/);
node('settings').click();assert.match(node('settings-content').textContent,/Asset root\/workspace\/\.linktools/);node('settings-dialog').close();

// Unavailable history cannot hide readable metadata or make refresh show stale values.
unavailableTimelines.add('a');sessions.set('a',{...sessions.get('a'),revision:3,cwd:'old/path',history_quality:'partial',active_execution_id:null});
node('open-kind').value='session';node('open-id').value='a';node('open-form').requestSubmit();await hashChanged;
assert.equal(node('conversation-title').textContent,'Alpha');
assert.match(node('session-details').textContent,/"revision": 3/);
assert.match(node('conversation').textContent,/SESSION_HISTORY_UNAVAILABLE/);
assert.doesNotMatch(node('conversation').textContent,/Ready when you are/);
assert.ok(calls.some(call=>call.key==='/api/sessions/a' && call.query.include_timeline==='false'));
sessions.set('a',{...sessions.get('a'),revision:4,cwd:'new/path',active_execution_id:'other-run'});
await refresh();
assert.match(node('session-details').textContent,/"revision": 4/);
assert.match(node('session-details').textContent,/new\/path/);
assert.match(node('session-details').textContent,/other-run/);
assert.doesNotMatch(node('session-details').textContent,/old\/path/);

// Refresh reconciles controls even when an empty session has no execution.
sessions.set(opaque,{...sessions.get(opaque),status:'OPEN',active_execution_id:null});
node('open-kind').value='session';node('open-id').value=opaque;node('open-form').requestSubmit();await hashChanged;
assert.equal(node('send').disabled,false);
sessions.set(opaque,{...sessions.get(opaque),status:'CLOSED'});
await refresh();
assert.match(node('conversation-meta').textContent,/CLOSED/);
assert.equal(node('send').disabled,true);assert.equal(node('prompt').disabled,true);assert.equal(node('current-status').textContent,'Session CLOSED');
console.log('DOM contracts passed: stale navigation/detail/action, disabled stale controls, uncertain fork, interrupted dialog, repeated submit, metrics navigation');
