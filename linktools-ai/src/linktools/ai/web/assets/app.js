import {terminal, eventKey, upsertModel, mergePage, readSSE, metricValue, duration, modelLabel, usageLabel, promptLayers} from './console.js';

const $ = id => document.getElementById(id);
const state = {config:null, view:'sessions', list:[], listCursor:null, session:null, turns:[], turnCursor:null, hasEarlierTurns:false, timelineError:'',
  execution:null, executionGeneration:0, tab:'overview', details:[], detailCursor:null, models:new Map(), events:new Map(),
  generation:0, listGeneration:0, detailGeneration:0, metricsGeneration:0, stream:null, cursor:null,
  liveText:'', liveThinking:'', pending:new Map(), actionsPending:new Set(), pendingForks:new Map(), recoveryReadbackId:null, selectedSession:null, selectedExecution:null};
const json = value => JSON.stringify(value, null, 2);
const text = value => typeof value === 'string' ? value : json(value);
const short = value => value ? String(value).slice(0, 12) : '—';
const date = value => value ? new Date(value).toLocaleString() : '—';
const enc = encodeURIComponent;
const sessionURL=(id,action="",paging={})=>`/api/session${action?"/"+action:""}?`+new URLSearchParams({session_id:id,...paging});
function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content != null) node.textContent = String(content);
  return node;
}
function button(label, action, className='quiet') {
  const node = element('button', className, label); node.type='button';
  node.onclick = () => action().catch?.(showError); return node;
}
function clear(id) { $(id).replaceChildren(); }
function notice(message='') { $('notice').textContent=message; $('notice').hidden=!message; }
function showError(error) {
  if(error.name==='AbortError')return;
  notice(error.message || String(error));
  if(error.context)$('notice').append(element('div','',error.context));
  if(error.details?.operation_id || Object.keys(error.details?.safe_details || {}).length) {
    $('notice').append(rawDetail({operation_id:error.details.operation_id,safe_details:error.details.safe_details},'Error details'));
  }
}
function connection(label, failed=false) { $('connection').textContent=label; $('connection').classList.toggle('error', failed); }
async function api(path, {body, signal}={}) {
  const response = await fetch(path, {method:body ? 'POST':'GET', signal, cache:'no-store',
    headers:body ? {'Content-Type':'application/json', 'X-LinkTools-Console':'1'} : {},
    body:body ? JSON.stringify(body) : undefined});
  const payload = await response.json();
  if (!response.ok) {
    const code = payload.error_code || payload.code || 'REQUEST_FAILED';
    const error = new Error(`${code}${payload.message ? ': '+payload.message : ''}`);
    error.code=code; error.details=payload; error.context=`${body ? 'POST':'GET'} ${path.split('?')[0]}`; throw error;
  }
  return payload;
}
async function mutate(path, payload, {newAttemptOn=[]}={}) {
  const key = path + JSON.stringify(payload);
  let operation = state.pending.get(key);
  if (operation?.running) return operation.running;
  if (!operation) { operation={id:crypto.randomUUID()}; state.pending.set(key, operation); }
  operation.running=api(path, {body:{...payload, request_id:operation.id}});
  try { const result=await operation.running; state.pending.delete(key); return result; }
  catch(error) {
    if(newAttemptOn.includes(error.code))state.pending.delete(key);
    throw error;
  }
  finally { operation.running=null; }
}
function setDisabled() {
  const readonly=!state.config || state.config.read_only;
  document.querySelectorAll('[data-view]').forEach(node=>node.disabled=!state.config);
  $('settings').disabled=!state.config;$('open-record').disabled=!state.config;
  ['new-session','welcome-new'].forEach(id => $(id).disabled=readonly);
  ['send','planning','thinking','memory','files','prompt'].forEach(id => $(id).disabled=readonly || state.session?.status === 'CLOSED' || !state.selectedSession);
  ['rename-session','fork-session','close-session'].forEach(id => $(id).disabled=readonly || !state.session);
  ['retry','fork-run','recover'].forEach(id => $(id).disabled=readonly || !state.execution);
  $('recover').disabled=readonly || !['RECOVERY_REQUIRED','PENDING_START','STARTED','CANCELLING'].includes(state.execution?.status) || state.recoveryReadbackId===state.selectedExecution || state.actionsPending.has(`${state.selectedExecution}:recover`);
  $('export').disabled=!state.execution || !terminal(state.execution.status);
  const stoppable=!readonly && state.execution && !terminal(state.execution.status);
  $('stop-run').hidden=!stoppable;
  $('cancel').hidden=!stoppable || state.session?.active_execution_id!==state.selectedExecution;
  $('fork-session').disabled=readonly || !state.session || state.actionsPending.has(`session:${state.selectedSession}:fork`);
}
function setView(view, refreshSelection=true) {
  state.view=view;
  document.querySelectorAll('[data-view]').forEach(node => node.classList.toggle('active', node.dataset.view===view));
  $('page-title').textContent={sessions:'Conversations',executions:'Executions',metrics:'Metrics'}[view];
  $('execution-filters').hidden=view !== 'executions'; $('filter-form').hidden=view === 'metrics';$('open-form').hidden=view==='metrics';
  if(view!=='metrics')$('open-kind').value=view==='sessions'?'session':'execution';
  $('filter-label').textContent=view === 'executions' ? 'Filter loaded executions':'Filter loaded conversations';
  $('filter').placeholder=view === 'executions' ? 'Execution ID or status':'Title or session ID';
  $('metrics-view').hidden=view !== 'metrics';
  $('conversation-view').hidden=view === 'metrics' || (!state.selectedSession && !state.selectedExecution);
  $('welcome').hidden=view === 'metrics' || Boolean(state.selectedSession || state.selectedExecution);
  if (view === 'metrics') { ++state.generation; stopStream(); loadMetrics().catch(showError); }
  else {
    loadList().catch(showError);
    if (refreshSelection && (state.selectedSession || state.selectedExecution)) openSelection(state.selectedSession,state.selectedExecution).catch(showError);
  }
}
async function loadList(more=false) {
  const generation=++state.listGeneration, view=state.view;
  if (view === 'metrics') return;
  $('list-action').value='paged';setListFilterAvailability();
  const params=new URLSearchParams({limit:'50'});
  if (more && state.listCursor) params.set('cursor',state.listCursor);
  if (view === 'executions') {
    if ($('filter-agent').value.trim()) params.set('agent_id',$('filter-agent').value);
    if ($('filter-session').value.trim()) params.set('session_id',$('filter-session').value);
    if ($('filter-parent').value.trim()) params.set('parent_execution_id',$('filter-parent').value);
  }
  const payload=await api(`/api/${view}?${params}`);
  if (generation !== state.listGeneration || view !== state.view) return;
  const key=item => view==='sessions' ? item.session_id:item.execution_id;
  state.list=more ? mergePage(state.list,payload.items,key):payload.items;
  state.listCursor=payload.next_cursor; $('list-more').hidden=!state.listCursor; $('list-scope').hidden=!payload.recent_only;
  $('list-scope').textContent='Showing recent sessions in read-only mode';
  renderList();
}
function setListFilterAvailability() {
  ['filter-agent','filter-session','filter-parent'].forEach(id=>$(id).disabled=$('list-action').value==='recent');
}
async function loadRecentExecutions() {
  if(state.actionsPending.has('recent-scan'))return;
  state.actionsPending.add('recent-scan');const generation=++state.listGeneration;
  try {
    const payload=await api('/api/executions?recent=true&limit=20');
    if(generation!==state.listGeneration || state.view!=='executions')return;
    state.list=payload.items;state.listCursor=null;$('list-more').hidden=true;
    $('list-scope').textContent='Newest 20 by creation time · scanned visible execution metadata';$('list-scope').hidden=false;
    renderList();
  } finally {state.actionsPending.delete('recent-scan');}
}
function renderList() {
  clear('list'); const filter=$('filter').value.trim().toLowerCase();
  const rows=state.list.filter(item => JSON.stringify(item).toLowerCase().includes(filter));
  if (!rows.length) $('list').append(element('p','empty',filter ? 'No matches in loaded rows.':'Nothing here yet.'));
  rows.forEach(item => {
    const session=state.view==='sessions', id=session ? item.session_id:item.execution_id;
    const active=session ? id===state.selectedSession:id===state.selectedExecution;
    const node=button('',async () => navigate(session ? id:item.session_id,session ? null:id),`list-item${active?' active':''}`);
    node.append(element('span','list-title',session ? item.metadata?.title || short(id) : short(id)));
    const sub=element('span','list-subtitle');
    sub.append(element('i',`status-dot ${item.status==='FAILED'?'failed':item.active_execution_id || !terminal(item.status) && !session ? 'running':''}`));
    sub.append(element('span','',`${item.status} · ${item.agent_id || 'task'} · ${short(id)}`));
    node.append(sub); $('list').append(node);
  });
}
function stopStream() { state.stream?.abort(); state.stream=null; }
function navigate(sessionId, executionId) {
  const params=new URLSearchParams();
  if (sessionId) params.set('session',sessionId);
  if (executionId) params.set('execution',executionId);
  const next='#'+params;
  if (location.hash===next) return openSelection(sessionId,executionId);
  location.hash=next;
}
function openExact(event) {
  event.preventDefault();if(!state.config)return;
  const id=$('open-id').value, session=$('open-kind').value==='session';
  if(!id.trim())return;
  setView(session?'sessions':'executions',false);
  return navigate(session?id:null,session?null:id);
}
async function readSession(id) {
  try {return await api(sessionURL(id,'',{limit:'50'}));}
  catch(error) {
    if(error.code!=='SESSION_HISTORY_UNAVAILABLE')throw error;
    const payload=await api(sessionURL(id,'',{include_timeline:'false'}));
    return {...payload,timeline_error:error.message};
  }
}
function renderSessionHeader() {
  if(!state.session)return;
  $('conversation-title').textContent=state.session.metadata?.title || 'Conversation';
  $('conversation-meta').textContent=`${state.selectedSession} · ${state.session.agent_id} · ${state.session.status} · history ${state.session.history_quality}`;
  $('session-details').replaceChildren(rawDetail(state.session,'Session metadata'));
}
async function openSelection(sessionId, executionId) {
  const generation=++state.generation;
  stopStream(); state.cursor=null; state.liveText=''; state.liveThinking=''; state.models.clear(); state.events.clear();
  state.selectedSession=sessionId || null; state.selectedExecution=executionId || null;
  state.session=null; state.execution=null; state.turns=[]; state.turnCursor=null; state.hasEarlierTurns=false;state.timelineError='';
  $('sidebar').classList.remove('open'); $('metrics-view').hidden=true; $('welcome').hidden=Boolean(sessionId || executionId);
  $('conversation-view').hidden=!sessionId && !executionId;
  $('composer').hidden=!sessionId; $('conversation-title').textContent='Loading…';
  $('conversation-meta').textContent=''; clear('session-details'); clear('conversation'); clear('inspector-content');
  setDisabled(); renderList(); notice(state.config?.read_only ? 'Read-only mode. Start ai web with a configured model to run agents.':'');
  if (!sessionId && !executionId) return;
  try {
    if (sessionId) {
      const payload=await readSession(sessionId);
      if (generation!==state.generation) return;
      state.session=payload.session;state.turns=payload.timeline?.items || [];state.turnCursor=payload.timeline?.next_cursor || null;state.timelineError=payload.timeline_error || '';
      renderSessionHeader();
      executionId=executionId || state.session.active_execution_id || state.turns.at(-1)?.execution_id;
      renderConversation();
    } else {
      $('conversation-title').textContent='Execution'; $('conversation-meta').textContent=executionId;renderConversation();
    }
    if (executionId) await selectExecution(executionId,generation);
    setDisabled(); renderList();
  } catch(error) { if (generation===state.generation) showError(error); }
}
async function moreTurns() {
  const generation=state.generation;
  const payload=await api(sessionURL(state.selectedSession,"",{limit:"50",cursor:state.turnCursor}));
  if (generation!==state.generation) return;
  state.session=payload.session;renderSessionHeader();
  state.turns=mergePage(payload.timeline.items,state.turns,item=>item.execution_id); state.turnCursor=payload.timeline.next_cursor; state.hasEarlierTurns=true;
  renderConversation({prepend:true});
}
function renderConversation({prepend=false}={}) {
  const container=$('conversation'), previousTop=container.scrollTop, previousHeight=container.scrollHeight;
  const follows=previousTop+container.clientHeight>=previousHeight-40;
  clear('conversation'); $('turns-more').hidden=!state.turnCursor;
  if(state.timelineError)$('conversation').append(element('p','empty',`${state.timelineError} · Session metadata is available. Browse its executions for retained detail; previously loaded turns may be incomplete.`));
  state.turns.forEach(turn => {
    const section=element('article','turn');
    const heading=element('div','turn-heading'); heading.append(element('span','',date(turn.created_at)),element('span','badge',turn.status));
    heading.append(button('Inspect ↗',async()=>selectExecution(turn.execution_id,state.generation)));
    section.append(heading);
    section.append(element('div','message user',typeof turn.user_input==='string' ? turn.user_input : text(turn.user_input)));
    const replies=turn.items.filter(item=>item.item_kind!=='user');
    replies.forEach(item => {
      if (item.item_kind==='assistant') {
        const label=element('div','message-label'); label.append(element('span','avatar','lt'),element('span','',state.session?.agent_id || 'AGENT'));
        section.append(label,element('div','message',text(item.content)));
      } else if (['tool_call','tool_result','thinking'].includes(item.item_kind)) {
        const detail=element('details','tool-message'); detail.append(element('summary','',`${item.item_kind.replaceAll('_',' ')} · ${item.tool_name || ''}`),element('pre','',text(item.content))); section.append(detail);
      }
    });
    if (!turn.conversation_committed) section.append(element('p','muted','History is incomplete; the Runtime has not committed this turn.'));
    if (turn.error_code) section.append(element('p','muted',`${turn.error_code} · ${text(turn.safe_error_details)}`));
    $('conversation').append(section);
  });
  if (!state.turns.length && !state.timelineError) $('conversation').append(element('p','empty',state.selectedSession ? 'Ready when you are. Send a message to start this conversation.':'Inspect this execution using the panels on the right.'));
  const live=element('section','turn'); live.id='live'; $('conversation').append(live); renderLive();
  container.scrollTop=prepend?previousTop+container.scrollHeight-previousHeight:follows?container.scrollHeight:previousTop;
}
function renderLive() {
  const live=$('live'); if (!live) return;
  const container=$('conversation'), follows=container.scrollTop+container.clientHeight>=container.scrollHeight-40;
  live.replaceChildren();
  if (state.liveThinking) { const detail=element('details','tool-message'); detail.append(element('summary','','Thinking'),element('pre','',state.liveThinking)); live.append(detail); }
  if (state.liveText) { live.append(element('div','message-label',`LIVE · ${state.execution?.agent_id || 'AGENT'} · ${short(state.selectedExecution)}`),element('div','message',state.liveText)); }
  [...state.events.values()].slice(-12).forEach(item => {
    const payload=item.event.payload || {};
    const row=element('div',`live-event${item.depth?' child':''}`,`${item.depth?'↳ ':''}${item.agent_id || 'task'} · ${item.event.event_type.replaceAll('_',' ').toLowerCase()}${payload.tool_name?' · '+payload.tool_name:''}${payload.call_id?' #'+short(payload.call_id):''}`);
    if(item.depth)row.append(button('Inspect subagent',async()=>selectExecution(item.execution_id)));
    live.append(row);
  });
  if(follows)container.scrollTop=container.scrollHeight;
}
async function selectExecution(id, generation=state.generation) {
  const executionGeneration=++state.executionGeneration;
  stopStream(); state.cursor=null; state.liveText=''; state.liveThinking=''; state.models.clear(); state.events.clear();
  state.selectedExecution=id; state.execution=null;setDisabled();clear('inspector-content');renderLive();
  $('execution-meta').textContent=`Loading ${id}…`;
  const info=await api(`/api/executions/${enc(id)}`);
  if (generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution) return;
  state.execution=info; renderExecution(); await loadDetail(false); setDisabled();
  if (generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution && !terminal(info.status) && !state.config.read_only) watchExecution(id,generation).catch(showError);
}
function renderExecution() {
  const info=state.execution; if (!info) return;
  $('execution-meta').replaceChildren(element('strong','',`${info.agent_id || info.task_id || 'Execution'} · ${info.status}`),element('div','mono',info.execution_id));
  if (info.parent_execution_id) $('execution-meta').append(button('↑ Parent execution',async()=>navigate(null,info.parent_execution_id)));
}
function properties(values) {
  const node=element('dl','kv');
  Object.entries(values).forEach(([key,value]) => node.append(element('dt','',key),element('dd','',typeof value==='object'?text(value):value ?? '—')));
  return node;
}
function rawDetail(value, title='Full details') {
  const detail=element('details'); detail.append(element('summary','',title),element('pre','json',json(value))); return detail;
}
async function loadDetail(more=false, selectors={}) {
  if (!state.selectedExecution) return;
  const generation=++state.detailGeneration, selection=state.generation, id=state.selectedExecution, tab=state.tab;
  if (!more) { state.details=[]; state.detailCursor=null; clear('inspector-content'); }
  $('detail-more').hidden=true;
  document.querySelectorAll('[data-tab]').forEach(node=>node.classList.toggle('active',node.dataset.tab===tab));
  if (tab==='overview') { renderOverview(); return; }
  if (tab==='recovery' && state.config.read_only) { $('inspector-content').append(element('p','empty','Recovery controls require an execution Runtime.')); return; }
  const params=new URLSearchParams({limit:'50',...selectors});
  if (['history','transcript','models'].includes(tab)) params.set('include_content','true');
  if (more && state.detailCursor) params.set('cursor',state.detailCursor);
  try {
    const payload=await api(`/api/executions/${enc(id)}/${tab}?${params}`);
    if (generation!==state.detailGeneration || selection!==state.generation || id!==state.selectedExecution || tab!==state.tab) return;
    const items=Array.isArray(payload)?payload:payload.items;
    state.details.push(...items); state.detailCursor=payload.next_cursor || null; state.detailSelectors=selectors;
    renderDetails(); $('detail-more').hidden=!state.detailCursor;
  } catch(error) { if(generation===state.detailGeneration) showError(error); }
}
function renderOverview() {
  clear('inspector-content'); const info=state.execution; if (!info) return;
  $('inspector-content').append(properties({Agent:info.agent_id,Kind:info.binding_kind,Session:info.session_id,Lineage:info.lineage_kind,Created:date(info.created_at),Started:date(info.started_at),Finished:date(info.terminal_at),Error:info.error_code}));
  if(info.usage) { const card=element('div','detail-card'); card.append(element('h3','','Usage'),properties({Requests:info.usage.logical_requests,'Input tokens':info.usage.input_tokens,'Output tokens':info.usage.output_tokens,'Unknown usage':info.usage.unknown_usage_requests}),rawDetail(info.usage)); $('inspector-content').append(card); }
  $('inspector-content').append(rawDetail(info,'Execution metadata'));
  if(state.models.size) {
    $('inspector-content').append(element('h3','','Live model requests'));
    state.models.forEach(item=>{const card=element('div','detail-card');card.append(element('h3','',`Request ${item.model_request_seq} · ${item.status}`),element('p','',`${item.execution_id} · run ${item.agent_run_seq} · ${item.purpose}`),rawDetail(item));$('inspector-content').append(card);});
  }
}
function renderDetails() {
  clear('inspector-content');
  if(!state.details.length) $('inspector-content').append(element('p','empty',terminal(state.execution?.status)?'No records for this selection.':'No durable detail is available yet. Live progress can precede committed history; refresh to check again.'));
  state.details.forEach(item=>{
    const card=element('div','detail-card');
    if(state.tab==='models') {
      card.append(element('h3','',`Request ${item.model_request_seq} · ${item.status}`),element('p','',`Run ${item.agent_run_seq} · depth ${item.depth} · ${item.purpose}`),element('p','',`${short(item.execution_id)} · ${modelLabel(item.model)} · ${duration(item.duration_ns)}`));
      if(item.usage)card.append(element('p','',usageLabel(item.usage)));
      if(item.request && Object.keys(item.request).length && item.content_included!==false){const layers=element('details');layers.append(element('summary','','Prompt architecture'),properties(Object.fromEntries(promptLayers(item.request))));card.append(layers);}
      else card.append(element('p','muted','Prompt content is not available yet.'));
      card.append(rawDetail(item,'Prompt, response & metadata'));
      if(item.execution_id!==state.selectedExecution)card.append(button('Open subagent',async()=>selectExecution(item.execution_id)));
    } else if(state.tab==='trace') {
      const row=item.payload || {}; card.append(element('h3','',`#${item.step_event_seq} · ${row.kind || 'Step'} · ${row.status || ''}`),element('p','',`${row.scope || ''}/${row.agent_run_seq || ''} · step ${row.step_index ?? '—'} · ${row.tool_name || 'request #'+(row.model_request_seq ?? '—')}`),element('p','',`${row.purpose || '—'} · ${duration(row.duration_ns)}${row.token_usage?' · '+usageLabel(row.token_usage):''}`),rawDetail(item));
      const selectors={}; ['agent_run_seq','model_request_seq','step_index','tool_call_id'].forEach(key=>{if(row[key]!=null)selectors[key]=row[key];});
      if(row.call_id)selectors.tool_call_id=row.call_id;
      card.append(button('Read content',async()=>{if(item.execution_id!==state.selectedExecution)await selectExecution(item.execution_id);state.tab='history';await loadDetail(false,selectors);}));
      if(row.child_execution_id)card.append(button('Open subagent',async()=>navigate(null,row.child_execution_id)));
    } else if(state.tab==='recovery') {
      card.append(element('h3','',item.tool_name),element('p','',`Effect outcome unknown · ${item.tool_call_id}`),rawDetail(item));
      ['not_applied','applied','failed'].forEach(resolution=>card.append(button(resolution.replaceAll('_',' '),async()=>{
        if(!confirm(`Confirm the external effect was ${resolution.replaceAll('_',' ')}? This controls recovery.`))return;
        let result=null; if(resolution==='applied'){const raw=prompt('Recorded tool result (JSON)','null');if(raw===null)return;result=JSON.parse(raw);}
        await mutate(`/api/executions/${enc(state.selectedExecution)}/resolve`,{operation_id:item.operation_id,expected_fence:item.fence,resolution,result});await loadDetail();
      })));
    } else {
      card.append(element('h3','',`${item.item_kind || 'Text'} ${item.tool_name?'· '+item.tool_name:''}`),element('p','',`Message ${item.message_seq} · run ${item.agent_run_seq ?? '—'}${item.execution_id?' · '+short(item.execution_id):''}`),element('pre','json',text(item.content ?? item.text)),rawDetail(item));
    }
    if(state.tab==='history' && item.execution_id!==state.selectedExecution)card.append(button('Open subagent',async()=>selectExecution(item.execution_id)));
    $('inspector-content').append(card);
  });
}
async function refreshSelected() {
  const generation=state.generation, executionGeneration=state.executionGeneration, sessionId=state.selectedSession, id=state.selectedExecution;
  if(sessionId) {
    const payload=await readSession(sessionId);
    if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    state.session=payload.session;state.timelineError=payload.timeline_error || '';renderSessionHeader();
    if(payload.timeline){state.turns=mergePage(state.turns,payload.timeline.items,item=>item.execution_id);if(!state.hasEarlierTurns)state.turnCursor=payload.timeline.next_cursor;}
    else{state.turnCursor=null;state.hasEarlierTurns=false;}
    const selected=state.turns.find(item=>item.execution_id===id);
    if(payload.timeline && selected?.conversation_committed){state.liveText='';state.liveThinking='';state.events.clear();}
    renderConversation();setDisabled();
  }
  if(id) {
    const info=await api(`/api/executions/${enc(id)}`);
    if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    state.execution=info;if(state.recoveryReadbackId===id)state.recoveryReadbackId=null;renderExecution();await loadDetail();setDisabled();
  }
  await loadList();
}
async function watchExecution(id,generation) {
  const controller=new AbortController();state.stream=controller;
  let delay=500;
  try {
  while(!controller.signal.aborted && generation===state.generation && id===state.selectedExecution) {
    try {
      const params=new URLSearchParams();if(state.cursor)params.set('cursor',state.cursor);
      const response=await fetch(`/api/executions/${enc(id)}/events?${params}`,{signal:controller.signal,cache:'no-store'});
      if(!response.ok){const payload=await response.json();const error=new Error(payload.error_code || payload.code || 'Observation unavailable');error.reconnect=response.status>=500;throw error;}
      connection('Live');
      let ended=false;
      await readSSE(response,async frame=>{
        if(generation!==state.generation || controller.signal.aborted || id!==state.selectedExecution)return;
        if(frame.event==='snapshot') {state.execution=frame.data;ended=true;renderExecution();setDisabled();return;}
        if(frame.event==='observation_error') {const error=new Error(`${frame.data.error_code}: observation interrupted; checking Runtime state`);error.reconnect=frame.data.origin==='stream' && !frame.data.safe_details?.cleanup_pending;throw error;}
        const envelope=frame.data,item=envelope.item;
        if(envelope.cursor)state.cursor=envelope.cursor;
        if(envelope.type==='model') {upsertModel(state.models,item.item);if(state.tab==='overview')renderOverview();}
        else {
          const event=item.event,payload=event.payload || {};
          if(item.depth===0 && event.event_type==='ASSISTANT_TEXT_DELTA')state.liveText+=payload.text || '';
          else if(item.depth===0 && event.event_type==='ASSISTANT_THINKING_DELTA')state.liveThinking+=payload.text || '';
          else if(!event.event_type.endsWith('_DELTA')) {
            state.events.set(eventKey(item),item);
            if(state.events.size>100)state.events.delete(state.events.keys().next().value);
          }
          renderLive();
        }
      },controller.signal);
      if(ended) {await refreshSelected();connection(terminal(state.execution?.status)?'Complete':state.execution?.status || 'Local');return;}
      throw new Error('Observation connection closed');
    } catch(error) {
      if(controller.signal.aborted || generation!==state.generation || id!==state.selectedExecution)return;
      connection('Reconnecting',true);
      // Transient text is not a replay checkpoint. Re-read canonical content rather than append duplicates.
      state.liveText='';state.liveThinking='';await refreshSelected();
      if(terminal(state.execution?.status)){connection(state.execution.status);return;}
      if(error.reconnect===false){connection('Observation paused',true);notice(`${error.message}. Refresh to read the current state and restart observation.`);return;}
      notice(`${error.message}. Execution continues; reconnecting from the Runtime cursor.`);
      await new Promise(resolve=>{const timer=setTimeout(resolve,delay);controller.signal.addEventListener('abort',()=>{clearTimeout(timer);resolve();},{once:true});});
      delay=Math.min(delay*2,5000);
    }
  }
  } finally {if(state.stream===controller)state.stream=null;}
}
async function sendMessage(event) {
  event.preventDefault(); const sessionId=state.selectedSession,generation=state.generation,promptValue=$('prompt').value;
  if(!sessionId || !promptValue.trim())return;
  $('send').disabled=true;notice('');
  try {
    // Rejected admission is terminal for its key; another explicit send is a new attempt.
    const result=await mutate(sessionURL(sessionId,"messages"),{prompt:promptValue,planning:$('planning').checked,thinking:$('thinking').checked,memory_scope:$('memory').value,files:$('files').value.split('\n').map(s=>s.trim()).filter(Boolean)}, {newAttemptOn:['SESSION_BUSY','SESSION_CONFLICT']});
    if(generation!==state.generation)return;
    if($('prompt').value===promptValue)$('prompt').value=''; await openSelection(sessionId,result.execution_id); await loadList();
  } catch(error){
    if(generation!==state.generation || sessionId!==state.selectedSession)return;
    if(['SESSION_BUSY','SESSION_CONFLICT'].includes(error.code))error.message+='; this message was not started. Check the current execution, then send again when the session is available.';
    showError(error);
  } finally {if(generation===state.generation)setDisabled();}
}
function openNew() {
  if(!state.config || state.config.read_only)return;
  $('new-dialog').dataset.sessionId=crypto.randomUUID();$('new-dialog').showModal();$('new-title').focus();
}
async function createSession(event) {
  event.preventDefault();const button=event.submitter;button.disabled=true;
  const sessionId=$('new-dialog').dataset.sessionId;
  try {
    const session=await mutate('/api/sessions',{session_id:sessionId,title:$('new-title').value,agent_id:$('new-agent').value,cwd:$('new-cwd').value || null});
    if($('new-dialog').open && $('new-dialog').dataset.sessionId===sessionId){$('new-dialog').close();setView('sessions',false);navigate(session.session_id,null);}
    await loadList();
  } catch(error){showError(error);} finally {button.disabled=false;}
}
async function sessionAction(action) {
  const id=state.selectedSession,generation=state.generation;if(!id || !state.session)return;
  const key=`session:${id}:${action}`;
  if(state.actionsPending.has(key))return;
  let payload={};
  if(action==='close' && !confirm('Close this session? Its history will remain available.'))return;
  if(action==='fork'){
    if(!state.pendingForks.has(id))state.pendingForks.set(id,crypto.randomUUID());
    payload.new_session_id=state.pendingForks.get(id);
  }
  if(action==='update') {
    const title=prompt('Conversation title',state.session.metadata?.title || 'Conversation');if(!title?.trim())return;
    payload={expected_revision:state.session.revision,metadata:{...state.session.metadata,title},cwd:state.session.cwd};
  }
  state.actionsPending.add(key);setDisabled();
  try {
    const result=await mutate(sessionURL(id,action),payload);
    if(action==='fork')state.pendingForks.delete(id);
    if(generation!==state.generation || id!==state.selectedSession){await loadList();return;}
    if(action==='fork')navigate(result.session_id,null);else await openSelection(id,state.selectedExecution);
    await loadList();
  } finally {state.actionsPending.delete(key);setDisabled();}
}

async function executionAction(action) {
  const id=state.selectedExecution,generation=state.generation,executionGeneration=state.executionGeneration;
  if(!id || !state.execution)return;
  const key=`${id}:${action}`;
  if(state.actionsPending.has(key))return;
  const payload={};
  if(['retry','fork'].includes(action)){const value=prompt(`${action==='retry'?'Retry':'Fork'} with this prompt`,$('prompt').value);if(!value?.trim())return;payload.prompt=value;}
  if(action==='recover' && !confirm('Recover this execution only after confirming its previous executor has stopped. Unresolved external effects must be resolved first. Proceed?'))return;
  state.actionsPending.add(key);setDisabled();
  try {
    const result=await mutate(`/api/executions/${enc(id)}/${action}`,payload);
    if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    if(action==='cancel'){notice(result.cancelled?'Cancellation confirmed.':'Cancellation requested; terminal outcome is not yet confirmed.');await refreshSelected();}
    else {
      const info=await api(`/api/executions/${enc(result.execution_id)}`);
      if(generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution)navigate(info.session_id,result.execution_id);
    }
  } catch(error) {
    if(action==='cancel' && error.code==='STORAGE_CONFLICT' && generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution){
      try {
        await refreshSelected();
        if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
        const status=state.execution?.status;
        error.message+=`; Runtime status re-read: ${status || 'unknown'}. ${terminal(status)?'Execution is terminal; no cancellation retry is needed.':'Cancellation is not confirmed. If still needed, choose Stop execution again.'}`;
      } catch(readbackError) {
        if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
        error.message+=`; cancellation outcome is unresolved (${readbackError.message}). Refresh to check the Runtime before another action.`;
      }
    }
    if(action==='recover' && generation===state.generation && id===state.selectedExecution){
      state.recoveryReadbackId=id;
      try {await refreshSelected();error.message+='; canonical state re-read. Recovery was not resent.';}
      catch(readbackError){error.message+=`; outcome unresolved (${readbackError.message}). Refresh before another recovery action.`;}
    }
    throw error;
  } finally {state.actionsPending.delete(key);setDisabled();}
}

async function loadMetrics(event) {
  event?.preventDefault();const generation=++state.metricsGeneration;
  const params=new URLSearchParams();
  [['metric-name','metric'],['metric-groups','group_by'],['metric-bucket','bucket_seconds'],['metric-filters','filters'],['metric-correlations','correlation_filters'],['metric-aggregation','aggregation'],['metric-percentile','percentile']].forEach(([id,key])=>{if($(id).value)params.set(key,$(id).value);});
  if($('metric-start').value)params.set('start',new Date($('metric-start').value).toISOString());
  if($('metric-end').value)params.set('end',new Date($('metric-end').value).toISOString());
  const payload=await api('/api/metrics?'+params);if(generation!==state.metricsGeneration)return;
  clear('metric-results');
  if(payload.unavailable)$('metric-results').append(element('p','empty','No metrics store exists yet.'));
  payload.items.forEach(result=>{
    const card=element('article','metric-card');card.append(element('h3','',result.metric.replace('linktools.','').replaceAll('.',' / ').replaceAll('_',' ')));
    if(result.points.length===1)card.append(element('div','value',metricValue(result,result.points[0])));
    else {const table=element('table');result.points.forEach(point=>{const row=element('tr');row.append(element('td','',point.dimensions.map(d=>d.join('=')).join(', ') || date(point.bucket_start)),element('td','',metricValue(result,point)));table.append(row);});const wrap=element('div','metric-points');wrap.append(table);card.append(wrap);}
    card.append(element('p','',`${result.aggregation} · ${result.unit} · ${result.points.reduce((sum,item)=>sum+item.sample_count,0)} samples`),element('p','',`${date(result.window_start)} → ${date(result.window_end)}`),rawDetail(result));$('metric-results').append(card);
  });
}
async function exportResult() {
  const id=state.selectedExecution,result=await api(`/api/executions/${enc(id)}/result`);
  const url=URL.createObjectURL(new Blob([json(result)],{type:'application/json'}));const anchor=document.createElement('a');anchor.href=url;anchor.download=`${id}.json`;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function showSettings() {
  clear('settings-content'); const config=state.config;if(!config)return;
  $('settings-content').append(properties({Workspace:config.workspace,'Asset root':config.asset_root,Namespace:config.namespace,Model:config.model || 'Not configured',Vision:config.vision,'API key':config.api_key_configured?'Configured':'Not configured','Base URL':config.base_url_configured?'Configured':'Default','Runtime DB':config.runtime_db,'Object store':config.object_store,'Metrics DB':config.metrics_db,Mode:config.read_only?'Read-only':'Local execution'}));
  const card=element('div','detail-card');card.append(element('h3','','Captured capabilities'));config.capabilities.forEach(item=>card.append(element('p','',`${item.kind} · ${item.id} · revision ${item.revision}`)));if(!config.capabilities.length)card.append(element('p','','Capability composition is loaded when an execution Runtime starts.'));$('settings-content').append(card);$('settings-dialog').showModal();
}
async function bootstrap() {
  state.config=await api('/api/config');$('workspace-label').textContent=state.config.workspace || 'Local Runtime';$('memory').value=state.config.memory_scope;
  state.config.capabilities.filter(item=>item.kind==='agent').forEach(item=>{const option=element('option','',item.id);option.value=item.id;$('new-agent').append(option);});
  state.config.metric_names.forEach(name=>{const option=element('option','',name.replace('linktools.',''));option.value=name;$('metric-names').append(option);});
  if(state.config.read_only){connection('Read-only');notice('Read-only mode. Configure a model when starting ai web to enable conversations.');}
  setDisabled();await loadList();await fromHash();
}
async function fromHash(){if(!state.config)return;const params=new URLSearchParams(location.hash.slice(1));if(state.view==='metrics')setView(params.has('session')?'sessions':'executions',false);await openSelection(params.get('session'),params.get('execution'));}
window.addEventListener('hashchange',()=>fromHash().catch(showError));window.addEventListener('beforeunload',stopStream);
$('new-session').onclick=openNew;$('welcome-new').onclick=openNew;$('new-form').onsubmit=createSession;
$('composer').onsubmit=sendMessage;$('prompt').onkeydown=event=>{if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();$('composer').requestSubmit();}};
$('menu').onclick=()=>$('sidebar').classList.toggle('open');
$('options-toggle').onclick=()=>$('composer-options').hidden=!$('composer-options').hidden;
$('open-form').onsubmit=event=>{Promise.resolve(openExact(event)).catch(showError);};
$('filter').oninput=renderList;$('list-action').onchange=setListFilterAvailability;
$('filter-form').onsubmit=event=>{event.preventDefault();(state.view==='executions' && $('list-action').value==='recent'?loadRecentExecutions():loadList()).catch(showError);};
$('list-more').onclick=()=>loadList(true).catch(showError);$('turns-more').onclick=()=>moreTurns().catch(showError);
$('detail-more').onclick=()=>loadDetail(true,state.detailSelectors).catch(showError);
$('refresh').onclick=()=>{notice('');(async()=>{if(state.view==='metrics'){await loadMetrics();return;}await refreshSelected();if(state.execution && !terminal(state.execution.status) && !state.stream && !state.config.read_only){state.cursor=null;watchExecution(state.selectedExecution,state.generation).catch(showError);}})().catch(showError);};
$('settings').onclick=showSettings;$('export').onclick=()=>exportResult().catch(showError);
$('rename-session').onclick=()=>sessionAction('update').catch(showError);$('fork-session').onclick=()=>sessionAction('fork').catch(showError);$('close-session').onclick=()=>sessionAction('close').catch(showError);
$('cancel').onclick=()=>executionAction('cancel').catch(showError);$('stop-run').onclick=()=>executionAction('cancel').catch(showError);$('retry').onclick=()=>executionAction('retry').catch(showError);$('fork-run').onclick=()=>executionAction('fork').catch(showError);$('recover').onclick=()=>executionAction('recover').catch(showError);
$('metrics-form').onsubmit=event=>loadMetrics(event).catch(showError);
document.querySelectorAll('[data-view]').forEach(node=>node.onclick=()=>setView(node.dataset.view));
document.querySelectorAll('[data-tab]').forEach(node=>node.onclick=()=>{state.tab=node.dataset.tab;loadDetail().catch(showError);});
document.querySelectorAll('[data-close]').forEach(node=>node.onclick=()=>$(node.dataset.close).close());
setDisabled();
bootstrap().catch(showError);
