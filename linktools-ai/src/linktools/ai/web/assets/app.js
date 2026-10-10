import {terminal, eventKey, upsertModel, mergePage, readSSE, metricValue, duration, modelLabel, usageLabel, promptLayers} from './console.js';
import {renderMarkdown, inputPresentation} from './message.js';

const $ = id => document.getElementById(id);
const state = {config:null, view:'sessions', drafts:new Map(), conversationExecution:null, browserExecution:null, filters:new Map(), list:[], listCursor:null, session:null, turns:[], turnCursor:null, hasEarlierTurns:false, timelineError:'',
  execution:null, executionGeneration:0, observedExecution:null, observedInfo:null, observationGeneration:0, observationReadGeneration:0, tab:'overview', details:[], detailCursor:null, models:new Map(), events:new Map(),
  generation:0, refreshGeneration:0, listGeneration:0, detailGeneration:0, metricsGeneration:0, stream:null, cursor:null,
  liveText:'', liveThinking:'', liveActivityOpen:false, pending:new Map(), actionsPending:new Set(), pendingForks:new Map(), recoveryReadbackId:null, endReadbackId:null, cancellationNoticeId:null, noticeRevision:0, selectedSession:null, selectedExecution:null};
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
function markdownMessage(source, className='') {
  const node=element('div',`message ${className}`), body=element('div','markdown');
  const original=element('details','message-source');
  original.append(element('summary','','Markdown source'),element('pre','',source));
  node.append(body,button('Copy Markdown',async()=>{
    if(!navigator.clipboard)throw new Error('Clipboard unavailable. Open Markdown source to copy the original text.');
    await navigator.clipboard.writeText(source);
  },'quiet copy-message'),original);
  body.innerHTML=renderMarkdown(source);
  return node;
}
function userMessage(value) {
  const input=inputPresentation(value), node=element('div','user-input');
  if(input.text)node.append(markdownMessage(input.text,'user'));
  else node.append(element('p','muted',input.attachments.length?'Attached input':'Structured input · open original input for details'));
  if(input.attachments.length) {
    const files=element('ul','attachments');
    input.attachments.forEach(item=>files.append(element('li','',item)));
    node.append(files);
  }
  if(input.structured)node.append(rawDetail(value,'Original input'));
  return node;
}
function clear(id) { $(id).replaceChildren(); }
function notice(message='', executionId=null, revision=state.noticeRevision+1) { state.noticeRevision=revision; state.cancellationNoticeId=executionId; $('notice').textContent=message; $('notice').hidden=!message; return revision; }
function showError(error) {
  if(error.name==='AbortError')return;
  notice(error.message || String(error));
  if(error.context)$('notice').append(element('div','',error.context));
  if(error.details?.operation_id || Object.keys(error.details?.safe_details || {}).length) {
    $('notice').append(rawDetail({operation_id:error.details.operation_id,safe_details:error.details.safe_details},'Error details'));
  }
}
function connection(label, failed=false) { $('connection').textContent=label==='Read-only'?label:`Observation: ${label}`; $('connection').hidden=!failed && label!=='Read-only'; $('connection').classList.toggle('error', failed); }
function showInspector(open, focus=false) {
  if(state.view==='sessions')state.conversationInspectorOpen=open;
  if(state.view==='executions')open=true;
  $('inspector').hidden=!open;$('inspector-toggle').setAttribute('aria-expanded',String(open));
  if(focus){const target=open?(state.view==='executions'?$('execution-meta'):$('inspector-close')):$('inspector-toggle');target.focus();if(open)target.scrollIntoView({block:'nearest'});}
}
function renderCurrentState() {
  const owner=state.view==='sessions'?state.session?.active_execution_id:null,info=owner===state.observedExecution?state.observedInfo:state.execution;
  let label='No execution selected';
  if(state.session && state.view==='sessions') {
    if(state.session.status!=='OPEN')label=`Session ${state.session.status}${owner?' · occupied':''}`;
    else if(owner)label=`Session occupied${owner===info?.execution_id?' · '+info.status:''}`;
    else label='Session available';
  } else if(info)label=`Selected execution: ${info.status}`;
  else if(state.selectedExecution || state.selectedSession)label='Loading…';
  $('current-status').textContent=label;
  $('current-status').classList.toggle('error',state.session?.status==='CLEANUP_REQUIRED' || Boolean(info && (!state.session || owner===info.execution_id) && ['FAILED','START_UNKNOWN','RECOVERY_REQUIRED'].includes(info.status)));
  $('show-active').hidden=!owner || owner===state.selectedExecution;
  $('stop-run').setAttribute('aria-label',`Stop selected execution ${state.selectedExecution || ''}`);
}
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
async function mutate(path, payload, {newAttemptOn=[],scope=''}={}) {
  const key = scope + path + JSON.stringify(payload);
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
  renderCurrentState();
  const readonly=!state.config || state.config.read_only;
  document.querySelectorAll('[data-view]').forEach(node=>node.disabled=!state.config);
  $('settings').disabled=!state.config;$('open-record').disabled=!state.config;
  ['new-session','welcome-new'].forEach(id => $(id).disabled=readonly);
  ['send','planning','thinking','memory','files','prompt'].forEach(id => $(id).disabled=readonly || state.session?.status === 'CLOSED' || !state.selectedSession);
  ['rename-session','fork-session','close-session'].forEach(id => $(id).disabled=readonly || !state.session);
  const ending=state.actionsPending.has(`${state.selectedExecution}:end-stopped`);
  const recoverable=['RECOVERY_REQUIRED','PENDING_START','STARTED','CANCELLING'].includes(state.execution?.status);
  ['retry','fork-run','end-stopped'].forEach(id => $(id).disabled=readonly || !state.execution || ending);
  $('end-stopped').disabled ||= !recoverable && state.endReadbackId!==state.selectedExecution;
  $('end-stopped').hidden=readonly || !state.execution || !recoverable && state.endReadbackId!==state.selectedExecution;
  $('end-stopped').textContent=state.execution?.session_id ? 'End previous execution and free session' : 'End stopped execution';
  if($('resume-run'))$('resume-run').disabled=readonly || !recoverable || ending || state.recoveryReadbackId===state.selectedExecution || state.actionsPending.has(`${state.selectedExecution}:recover`);
  $('export').disabled=!state.execution || !terminal(state.execution.status);
  const active=state.session?.active_execution_id===state.observedExecution?state.observedInfo:null;
  $('end-active').hidden=readonly || !active || !['RECOVERY_REQUIRED','PENDING_START','STARTED','CANCELLING'].includes(active.status) && state.endReadbackId!==active.execution_id;
  $('end-active').disabled=Boolean(active && [...state.actionsPending].some(key=>key.startsWith(`${active.execution_id}:`)));
  const stoppable=!readonly && state.execution && !terminal(state.execution.status);
  $('stop-run').hidden=!stoppable;
  $('cancel').hidden=readonly || !active || terminal(active.status);
  $('cancel').setAttribute('aria-label',`Stop current execution ${active?.execution_id || ''}`);
  $('stop-run').disabled=ending;$('cancel').disabled=Boolean(active && state.actionsPending.has(`${active.execution_id}:end-stopped`));
  $('fork-session').disabled=readonly || !state.session || state.actionsPending.has(`session:${state.selectedSession}:fork`);
}
function renderPage() {
  const sessions=state.view==='sessions', executions=state.view==='executions', metrics=state.view==='metrics';
  $('metrics-view').hidden=!metrics;
  $('conversation-view').hidden=metrics || sessions && !state.selectedSession;
  $('conversation-column').hidden=!sessions;
  $('welcome').hidden=!sessions || Boolean(state.selectedSession);
  $('execution-list').hidden=!executions;
  $('browse-panel').hidden=metrics;
  (executions?$('execution-list'):$('sidebar-browse')).append($('browse-panel'));
  $('composer').hidden=!sessions || !state.selectedSession;
  $('session-details').hidden=!sessions;
  $('inspector-close').hidden=executions;
  $('inspector').classList.toggle('execution-page',executions);
  if(executions)showInspector(true);
  $('execution-live').hidden=!executions;
  if($('live'))clear('live');clear('execution-live');renderLive();
}
function setView(view, refreshSelection=true) {
  if(view!==state.view){saveDraft();++state.generation;state.filters.set(state.view,$('filter').value);$('filter').value=state.filters.get(view) || '';state.list=[];clear('list');}
  state.view=view;
  document.querySelectorAll('[data-view]').forEach(node => node.classList.toggle('active', node.dataset.view===view));
  $('page-title').textContent={sessions:'Conversations',executions:'Executions',metrics:'Metrics'}[view];
  $('execution-filters').hidden=view !== 'executions'; $('filter-form').hidden=view === 'metrics';$('open-tools').hidden=view==='metrics';
  if(view!=='metrics')$('open-kind').value=view==='sessions'?'session':'execution';
  $('filter-label').textContent=view === 'executions' ? 'Filter loaded executions':'Filter loaded conversations';
  $('filter').placeholder=view === 'executions' ? 'Execution ID or status':'Title or session ID';
  renderPage();
  if(view==='metrics')loadMetrics().catch(showError);
  else loadList().catch(showError);
  if(refreshSelection)setDisabled();
}
function navigateView(view) {
  const params=new URLSearchParams({view});
  if(view==='sessions' && state.selectedSession){params.set('session',state.selectedSession);if(state.conversationInspectorOpen && state.conversationExecution)params.set('execution',state.conversationExecution);}
  if(view==='executions' && state.browserExecution)params.set('execution',state.browserExecution);
  if(location.hash==='#'+params)return setView(view);
  location.hash='#'+params;
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
  const payload=await api(`/api/${view}?${params}`).catch(error=>{if(generation===state.listGeneration && view===state.view)throw error;return null;});
  if(!payload)return;
  if (generation !== state.listGeneration || view !== state.view) return;
  if(view==='executions'){const count=['agent_id','session_id','parent_execution_id'].filter(key=>params.has(key)).length;$('execution-filter-summary').textContent=`Execution filters${count?' · '+count+' applied':''}`;}
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
    $('execution-filter-summary').textContent='Execution filters · Newest 20';
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
    const node=button('',async () => navigate(session ? id:null,session ? null:id),`list-item${active?' active':''}`);
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
  if(!sessionId && executionId)params.set('view','executions');
  if (sessionId) params.set('session',sessionId);
  if (executionId) params.set('execution',executionId);
  const next='#'+params;
  if (location.hash===next) return fromHash();
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
  $('conversation-meta').textContent=`${state.session.agent_id}${state.session.status==='CLOSED'?' · CLOSED':''}${state.session.history_quality!=='complete'?' · history '+state.session.history_quality:''}`;
  $('session-details').replaceChildren(rawDetail(state.session,'Session metadata'));
}
function saveDraft() {
  if(!state.selectedSession)return;
  state.drafts.set(state.selectedSession,{prompt:$('prompt').value,files:$('files').value,planning:$('planning').checked,thinking:$('thinking').checked,memory:$('memory').value});
}
function restoreDraft(sessionId) {
  const draft=state.drafts.get(sessionId) || {prompt:'',files:'',planning:false,thinking:false,memory:state.config.memory_scope};
  for(const id of ['prompt','files','memory'])$(id).value=draft[id];
  for(const id of ['planning','thinking'])$(id).checked=draft[id];
}
async function openSelection(sessionId, executionId, {preserveInspector=false}={}) {
  const preserveObservation=Boolean(sessionId && sessionId===state.selectedSession);saveDraft();
  const generation=++state.generation;
  if(!preserveObservation){stopStream(); ++state.observationGeneration; state.observedExecution=null;state.observedInfo=null; state.cursor=null; state.liveText=''; state.liveThinking=''; state.liveActivityOpen=false;state.models.clear(); state.events.clear();}
  state.selectedSession=sessionId || null; state.selectedExecution=executionId || null;
  restoreDraft(state.selectedSession);
  state.session=null; state.execution=null; state.turns=[]; state.turnCursor=null; state.hasEarlierTurns=false;state.timelineError='';
  const explicitExecution=Boolean(executionId) || state.view==='executions';$('action-menu').open=false;
  $('sidebar').classList.remove('open');renderPage();
  if(!preserveInspector)showInspector(explicitExecution,explicitExecution && Boolean(sessionId || executionId));
  $('composer').hidden=!sessionId; $('conversation-title').textContent='Loading…';
  $('conversation-meta').textContent=''; clear('session-details'); if(!preserveObservation)clear('conversation'); clear('inspector-content');
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
  const liveFocus=document.activeElement?.dataset?.liveFocus;
  const container=$('conversation'), previousTop=container.scrollTop, previousHeight=container.scrollHeight;
  const follows=previousTop+container.clientHeight>=previousHeight-40;
  clear('conversation'); $('turns-more').hidden=!state.turnCursor;
  if(state.timelineError)$('conversation').append(element('p','empty',`${state.timelineError} · Session metadata is available. Browse its executions for retained detail; previously loaded turns may be incomplete.`));
  state.turns.forEach(turn => {
    const section=element('article','turn');
    const heading=element('div','turn-heading'); heading.append(element('span','',date(turn.created_at)),element('span','badge',turn.status));
    heading.append(button('Details',async()=>{showInspector(true,true);await selectExecution(turn.execution_id,state.generation);}));
    section.append(heading);
    section.append(userMessage(turn.user_input));
    const replies=turn.items.filter(item=>item.item_kind!=='user');
    replies.forEach(item => {
      if (item.item_kind==='assistant') {
        const label=element('div','message-label'); label.append(element('span','avatar','lt'),element('span','',state.session?.agent_id || 'AGENT'));
        section.append(label,typeof item.content==='string'?markdownMessage(item.content):rawDetail(item.content,'Structured output'));
      } else if (['tool_call','tool_result','thinking'].includes(item.item_kind)) {
        const detail=element('details','tool-message'); detail.append(element('summary','',`${item.item_kind.replaceAll('_',' ')} · ${item.tool_name || ''}`),element('pre','',text(item.content))); section.append(detail);
      }
    });
    if (!turn.conversation_committed) section.append(element('p','muted','History is incomplete; the Runtime has not committed this turn.'));
    if (turn.error_code) section.append(element('p','muted',turn.error_code==='EXECUTION_CANCELLED' ? 'Execution cancelled.' : `${turn.error_code}${Object.keys(turn.safe_error_details || {}).length?' · '+text(turn.safe_error_details):''}`));
    $('conversation').append(section);
  });
  if (!state.turns.length && !state.timelineError) $('conversation').append(element('p','empty',state.selectedSession ? 'Ready when you are. Send a message to start this conversation.':'Inspect this execution using the panels on the right.'));
  const live=element('section','turn'); live.id='live'; $('conversation').append(live); renderLive(liveFocus);
  container.scrollTop=prepend?previousTop+container.scrollHeight-previousHeight:follows?container.scrollHeight:previousTop;
}
let liveRenderTimer=null;
function scheduleLiveRender() {
  if(liveRenderTimer===null)liveRenderTimer=setTimeout(()=>{liveRenderTimer=null;renderLive();},60);
}
function renderLive(liveFocus=document.activeElement?.dataset?.liveFocus) {
  let live=$('live');
  if(state.view==='executions'){live=$('execution-live');if(state.selectedExecution!==state.observedExecution){live.replaceChildren();return;}}
  if (!live) return;
  const container=$('conversation'), follows=container.scrollTop+container.clientHeight>=container.scrollHeight-40;
  live.replaceChildren();
  if (state.liveThinking) { const detail=element('details','tool-message'); detail.append(element('summary','','Thinking'),element('pre','',state.liveThinking)); live.append(detail); }
  if (state.liveText) {
    if(live.markdownSource!==state.liveText){live.markdownSource=state.liveText;live.markdownMessage=markdownMessage(state.liveText);}
    live.append(element('div','message-label',`LIVE · ${state.observedInfo?.agent_id || 'AGENT'} · ${short(state.observedExecution)}`),live.markdownMessage);
  }
  const events=[...state.events.values()].slice(-12),activity=element('details','tool-message');activity.id='live-activity';activity.open=state.liveActivityOpen;
  activity.ontoggle=()=>{if($('live-activity')===activity)state.liveActivityOpen=activity.open;};
  const attention=item=>['EXECUTION_START_UNKNOWN','EXECUTION_RECOVERY_REQUIRED','EXECUTION_FAILED','CANCEL_REQUESTED','EXECUTION_CANCELLED','APPROVAL_REQUESTED','EXTERNAL_REQUESTED'].includes(item.event.event_type)
    || ['MODEL_REQUEST_FINISHED','TOOL_CALL_FINISHED'].includes(item.event.event_type) && ['FAILED','CANCELLED'].includes(item.event.payload?.status);
  const ordinary=events.filter(item=>!attention(item)).length;
  const summary=element('summary','',`Activity · ${ordinary} recent events`);summary.dataset.liveFocus='summary';activity.append(summary);
  let restoreFocus=liveFocus==='summary' && ordinary ? summary:null;
  events.forEach(item => {
    const payload=item.event.payload || {};
    const row=element('div',`live-event${item.depth?' child':''}`,`${item.depth?'↳ ':''}${item.agent_id || 'task'} · ${item.event.event_type.replaceAll('_',' ').toLowerCase()}${payload.status?' · '+payload.status:''}${payload.error_code?' · '+payload.error_code:''}${payload.tool_name?' · '+payload.tool_name:''}${payload.call_id?' #'+short(payload.call_id):''}`);
    if(item.depth){const inspect=button('Inspect subagent',async()=>{showInspector(true,true);await selectExecution(item.execution_id);});inspect.dataset.liveFocus=eventKey(item);row.append(inspect);if(liveFocus===inspect.dataset.liveFocus)restoreFocus=inspect;}
    (attention(item)?live:activity).append(row);
  });
  if(ordinary)live.append(activity);
  restoreFocus?.focus({preventScroll:true});
  if(follows)container.scrollTop=container.scrollHeight;
}
function observedInfo(info, readGeneration=state.observationReadGeneration) {
  if(info.execution_id!==state.observedExecution)return info;
  const previous=state.observedInfo;
  if(previous && readGeneration!==state.observationReadGeneration)return previous;
  state.observedInfo=info;return info;
}
async function selectExecution(id, generation=state.generation) {
  const executionGeneration=++state.executionGeneration;
  const readGeneration=id===state.observedExecution?++state.observationReadGeneration:state.observationReadGeneration;
  state.selectedExecution=id;if(state.view==='sessions')state.conversationExecution=id;else if(state.view==='executions')state.browserExecution=id; state.execution=null;setDisabled();clear('inspector-content');renderLive();
  $('execution-meta').textContent=`Loading ${id}…`;
  try {
  let info=await api(`/api/executions/${enc(id)}`);
  if (generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution) return;
  info=observedInfo(info,readGeneration);
  state.execution=info; renderExecution(); await loadDetail(false); setDisabled();
  if (generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution) await syncObservation();
  } catch(error){if(generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution && (id!==state.observedExecution || readGeneration===state.observationReadGeneration))throw error;}
}
function renderExecution() {
  const info=state.execution; if (!info) return;
  renderCurrentState();
  $('execution-meta').replaceChildren(element('strong','',`${info.agent_id || info.task_id || 'Execution'} · ${info.status}`),element('div','mono',info.execution_id));
  if (info.parent_execution_id) $('execution-meta').append(button('↑ Parent execution',async()=>state.view==='sessions'?selectExecution(info.parent_execution_id):navigate(null,info.parent_execution_id)));
  if(info.session_id && state.view==='executions')$('execution-meta').append(button('Open conversation',async()=>navigate(info.session_id,null)));
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
  } catch(error) { if(generation===state.detailGeneration && selection===state.generation && id===state.selectedExecution && tab===state.tab) showError(error); }
}
function renderOverview() {
  clear('inspector-content'); const info=state.execution; if (!info) return;
  $('inspector-content').append(properties({Agent:info.agent_id,Kind:info.binding_kind,Session:info.session_id,Lineage:info.lineage_kind,Created:date(info.created_at),Started:date(info.started_at),Finished:date(info.terminal_at),Error:info.error_code}));
  if(info.usage) { const card=element('div','detail-card'); card.append(element('h3','','Usage'),properties({Requests:info.usage.logical_requests,'Input tokens':info.usage.input_tokens,'Output tokens':info.usage.output_tokens,'Unknown usage':info.usage.unknown_usage_requests}),rawDetail(info.usage)); $('inspector-content').append(card); }
  $('inspector-content').append(rawDetail(info,'Execution metadata'));
  if(state.models.size && state.selectedExecution===state.observedExecution) {
    $('inspector-content').append(element('h3','','Live model requests'));
    state.models.forEach(item=>{const card=element('div','detail-card');card.append(element('h3','',`Request ${item.model_request_seq} · ${item.status}`),element('p','',`${item.execution_id} · run ${item.agent_run_seq} · ${item.purpose}`),rawDetail(item));$('inspector-content').append(card);});
  }
}
function renderDetails() {
  clear('inspector-content');
  if(state.tab==='recovery' && !state.config.read_only) {
    const resume=button('Resume stopped execution',()=>executionAction('recover'));resume.id='resume-run';
    $('inspector-content').append(element('p','empty','Resume continues the original work and may call models or tools. To end it instead, use End stopped execution.'),resume);setDisabled();
  }
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
      card.append(button('Read content',async()=>{
        if(item.execution_id!==state.selectedExecution){
          const generation=state.generation,selection=selectExecution(item.execution_id),executionGeneration=state.executionGeneration;
          await selection;
          if(generation!==state.generation || executionGeneration!==state.executionGeneration || item.execution_id!==state.selectedExecution || state.tab!=='trace')return;
        }
        state.tab='history';await loadDetail(false,selectors);
      }));
      if(row.child_execution_id)card.append(button('Open subagent',async()=>state.view==='sessions'?selectExecution(row.child_execution_id):navigate(null,row.child_execution_id)));
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
  const refreshGeneration=++state.refreshGeneration, generation=state.generation, executionGeneration=state.executionGeneration, sessionId=state.view==='sessions'?state.selectedSession:null, id=state.selectedExecution, noticeRevision=state.noticeRevision, noticeId=state.cancellationNoticeId;
  try {
  if(sessionId) {
    const payload=await readSession(sessionId);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    state.session=payload.session;state.timelineError=payload.timeline_error || '';renderSessionHeader();
    if(payload.timeline){state.turns=mergePage(state.turns,payload.timeline.items,item=>item.execution_id);if(!state.hasEarlierTurns)state.turnCursor=payload.timeline.next_cursor;}
    else{state.turnCursor=null;state.hasEarlierTurns=false;}
    const selected=state.turns.find(item=>item.execution_id===state.observedExecution);
    if(payload.timeline && selected?.conversation_committed){state.liveText='';state.liveThinking='';state.events.clear();}
    renderConversation();setDisabled();
  }
  if(id) {
    const readGeneration=id===state.observedExecution?++state.observationReadGeneration:state.observationReadGeneration;
    let info=await api(`/api/executions/${enc(id)}`);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    info=observedInfo(info,readGeneration);state.execution=info;if(state.recoveryReadbackId===id)state.recoveryReadbackId=null;renderExecution();
    await loadDetail();setDisabled();
  }
  if(noticeId) {
    const info=noticeId===id?state.execution:await api(`/api/executions/${enc(noticeId)}`);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    if(terminal(info.status)) {
      const session=info.session_id ? (await api(sessionURL(info.session_id,'',{include_timeline:'false'}))).session : null;
      if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
      if(session && sessionId===info.session_id){state.session=session;renderSessionHeader();}
      reconcileCancellation(info,session,noticeRevision);setDisabled();
    }
  }
  await loadList();
  } catch(error) {
    if(refreshGeneration===state.refreshGeneration && generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution && noticeRevision===state.noticeRevision)throw error;
  }
}
async function syncObservation({refresh=false}={}) {
  const id=state.view==='executions' ? state.selectedExecution : state.selectedSession ? state.session?.active_execution_id || state.turns.at(-1)?.execution_id : state.selectedExecution;
  if(state.view==='executions' && !id)return;
  if(id!==state.observedExecution) {
    stopStream();++state.observationGeneration;state.observedExecution=id || null;state.observedInfo=null;
    state.cursor=null;state.liveText='';state.liveThinking='';state.liveActivityOpen=false;state.models.clear();state.events.clear();renderLive();
  }
  if(!id)return;
  const generation=state.observationGeneration,view=state.view;
  let info=state.observedInfo;
  if(refresh || !info){
    const readGeneration=++state.observationReadGeneration;
    try {info=id===state.selectedExecution?state.execution:await api(`/api/executions/${enc(id)}`);}
    catch(error){if(generation===state.observationGeneration && id===state.observedExecution && readGeneration===state.observationReadGeneration && view===state.view)throw error;return;}
    if(generation!==state.observationGeneration || id!==state.observedExecution || readGeneration!==state.observationReadGeneration)return;
  }
  if(!info)return;
  info=observedInfo(info);setDisabled();
  if(info && !terminal(info.status) && !state.config.read_only && !state.stream)watchExecution(id,generation).catch(showError);
}
function reconcileCancellation(info, session, revision) {
  if(state.cancellationNoticeId!==info.execution_id || revision!==state.noticeRevision || !terminal(info.status))return;
  const outcome=info.status==='CANCELLED'?'Cancellation confirmed.':`Execution finished: ${info.status}.`;
  const occupied=session?.active_execution_id===info.execution_id;
  notice(`${outcome}${occupied?' Session release is not yet confirmed.':session?.active_execution_id?' Another execution now owns the session.':session?' Session released.':''}`,occupied?info.execution_id:null,revision);
}
async function refreshObservation(id, generation) {
  const sessionId=state.observedInfo?.session_id===state.selectedSession?state.selectedSession:null, refreshGeneration=++state.refreshGeneration, readGeneration=++state.observationReadGeneration, noticeRevision=state.noticeRevision;
  const current=()=>generation===state.observationGeneration && id===state.observedExecution && refreshGeneration===state.refreshGeneration && readGeneration===state.observationReadGeneration;
  let info=await api(`/api/executions/${enc(id)}`);
  if(!current())return;
  const payload=sessionId ? await readSession(sessionId) : null;
  if(!current())return;
  info=observedInfo(info,readGeneration);
  if(id===state.selectedExecution){state.execution=info;renderExecution();}
  if(payload && sessionId===state.selectedSession) {
    state.session=payload.session;state.timelineError=payload.timeline_error || '';renderSessionHeader();
    if(payload.timeline){state.turns=mergePage(state.turns,payload.timeline.items,item=>item.execution_id);if(!state.hasEarlierTurns)state.turnCursor=payload.timeline.next_cursor;}
    const turn=state.turns.find(item=>item.execution_id===id);
    if(payload.timeline && turn?.conversation_committed){state.liveText='';state.liveThinking='';state.events.clear();}
    renderConversation();
  }
  if(state.cancellationNoticeId===id && terminal(info.status)) {
    const session=info.session_id ? (payload && info.session_id===sessionId ? payload.session : (await api(sessionURL(info.session_id,'',{include_timeline:'false'}))).session) : null;
    if(!current())return;
    reconcileCancellation(info,session,noticeRevision);
  }
  setDisabled();return true;
}
async function watchExecution(id,generation) {
  const controller=new AbortController();state.stream=controller;
  const current=()=>!controller.signal.aborted && generation===state.observationGeneration && id===state.observedExecution;
  let delay=500;
  const backoff=()=>new Promise(resolve=>{
    const done=()=>{clearTimeout(timer);controller.signal.removeEventListener('abort',done);resolve();};
    const timer=setTimeout(done,delay);controller.signal.addEventListener('abort',done,{once:true});
  });
  try {
  while(current()) {
    try {
      const params=new URLSearchParams();if(state.cursor)params.set('cursor',state.cursor);
      const response=await fetch(`/api/executions/${enc(id)}/events?${params}`,{signal:controller.signal,cache:'no-store'});
      if(!response.ok){const payload=await response.json();const error=new Error(payload.error_code || payload.code || 'Observation unavailable');error.reconnect=response.status>=500;throw error;}
      if(!current())return;
      connection('Live');
      let ended=false;
      await readSSE(response,async frame=>{
        if(!current())return;
        if(frame.event==='snapshot') {
          ++state.observationReadGeneration;const info=observedInfo(frame.data);ended=true;
          if(id===state.selectedExecution){state.execution=info;renderExecution();}
          setDisabled();return;
        }
        if(frame.event==='observation_error') {const error=new Error(`${frame.data.error_code}: observation interrupted; checking Runtime state`);error.reconnect=frame.data.origin==='stream' && !frame.data.safe_details?.cleanup_pending;throw error;}
        const envelope=frame.data,item=envelope.item;
        if(envelope.cursor)state.cursor=envelope.cursor;
        if(envelope.type==='model') {upsertModel(state.models,item.item);if(state.tab==='overview' && id===state.selectedExecution)renderOverview();}
        else {
          const event=item.event,payload=event.payload || {};
          if(item.depth===0 && event.event_type==='ASSISTANT_TEXT_DELTA')state.liveText+=payload.text || '';
          else if(item.depth===0 && event.event_type==='ASSISTANT_THINKING_DELTA')state.liveThinking+=payload.text || '';
          else if(!event.event_type.endsWith('_DELTA')) {
            state.events.set(eventKey(item),item);
            if(state.events.size>100)state.events.delete(state.events.keys().next().value);
          }
          scheduleLiveRender();
        }
      },controller.signal);
      if(!current())return;
      if(ended) {
        const refreshed=await refreshObservation(id,generation);
        if(!current())return;
        if(!refreshed)throw new Error('Canonical observation read was superseded');
        if(state.view==='sessions' && state.session?.active_execution_id && state.session.active_execution_id!==id){await syncObservation();if(!current())return;}
        const status=state.observedInfo?.status;
        if(terminal(status)){connection('Complete');return;}
        if(status==='RECOVERY_REQUIRED'){connection('Recovery required',true);return;}
      }
      throw new Error('Observation connection closed');
    } catch(error) {
      if(!current())return;
      connection('Reconnecting',true);
      // Live deltas can be replayed from the bounded buffer; only the Runtime cursor is durable.
      state.liveText='';state.liveThinking='';renderLive();
      let refreshed=false;
      try {refreshed=await refreshObservation(id,generation);}
      catch {if(current())connection('Reconnecting · Runtime read unavailable',true);}
      if(!current())return;
      if(refreshed && state.view==='sessions' && state.session?.active_execution_id && state.session.active_execution_id!==id){await syncObservation();if(!current())return;}
      if(refreshed && terminal(state.observedInfo?.status)){connection(state.observedInfo.status);return;}
      if(error.reconnect===false){connection(`Paused · ${error.message} · refresh to retry`,true);return;}
      await backoff();delay=Math.min(delay*2,5000);
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
    if(generation!==state.generation){const draft=state.drafts.get(sessionId);if(draft?.prompt===promptValue)draft.prompt='';if(state.selectedSession===sessionId && $('prompt').value===promptValue)$('prompt').value='';return;}
    if($('prompt').value===promptValue)$('prompt').value=''; await openSelection(sessionId,result.execution_id,{preserveInspector:true}); await loadList();
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
    if(action==='fork')navigate(result.session_id,null);else await openSelection(id,state.selectedExecution,{preserveInspector:true});
    await loadList();
  } finally {state.actionsPending.delete(key);setDisabled();}
}

async function executionAction(action, {active=false}={}) {
  const id=active?state.session?.active_execution_id:state.selectedExecution,generation=state.generation,executionGeneration=state.executionGeneration;
  const current=()=>generation===state.generation && (active?id===state.observedExecution:executionGeneration===state.executionGeneration && id===state.selectedExecution);
  if(!id || !(active?state.observedInfo:state.execution))return;
  const key=`${id}:${action}`;
  if(state.actionsPending.has(key) || state.actionsPending.has(`${id}:end-stopped`))return;
  const payload={};let noticeRevision=state.noticeRevision;
  if(['retry','fork'].includes(action)){const value=prompt(`${action==='retry'?'Retry':'Fork'} with this prompt`,$('prompt').value);if(!value?.trim())return;payload.prompt=value;}
  if(action==='recover' && !confirm('Resume this execution only after confirming its previous executor has stopped. This may call models or tools. Unresolved external effects must be resolved first. Proceed?'))return;
  state.actionsPending.add(key);setDisabled();
  try {
    const result=await mutate(`/api/executions/${enc(id)}/${action}`,payload);
    if(!current())return;
    if(action==='cancel'){
      if(noticeRevision===state.noticeRevision)noticeRevision=notice(result.cancelled?'Cancellation confirmed; checking session release.':`Cancellation requested; terminal outcome is not yet confirmed. If the previous executor has stopped, choose ${$('end-stopped').textContent}.`,id);
      await refreshSelected();
      if(active && current())await refreshObservation(id,state.observationGeneration);
    }
    else {
      const info=await api(`/api/executions/${enc(result.execution_id)}`);
      if(current())navigate(info.session_id,result.execution_id);
    }
  } catch(error) {
    if(!current() || noticeRevision!==state.noticeRevision)return;
    if(action==='cancel' && error.code==='STORAGE_CONFLICT' && current()){
      try {
        await refreshSelected();
        if(active && current())await refreshObservation(id,state.observationGeneration);
        if(!current())return;
        const status=(active?state.observedInfo:state.execution)?.status;
        error.message+=`; Runtime status re-read: ${status || 'unknown'}. ${terminal(status)?'Execution is terminal; no cancellation retry is needed.':'Cancellation is not confirmed. If still needed, choose Stop execution again.'}`;
      } catch(readbackError) {
        if(!current())return;
        error.message+=`; cancellation outcome is unresolved (${readbackError.message}). Refresh to check the Runtime before another action.`;
      }
    }
    if(action==='recover' && generation===state.generation && id===state.selectedExecution){
      state.recoveryReadbackId=id;
      try {await refreshSelected();error.message+='; canonical state re-read. Recovery was not resent.';}
      catch(readbackError){error.message+=`; outcome unresolved (${readbackError.message}). Refresh before another recovery action.`;}
    }
    if(current() && noticeRevision===state.noticeRevision)throw error;
  } finally {state.actionsPending.delete(key);setDisabled();}
}

async function endStoppedExecution({active=false}={}) {
  const id=active?state.session?.active_execution_id:state.selectedExecution,generation=state.generation,executionGeneration=state.executionGeneration;
  if(!id || !(active?state.observedInfo:state.execution) || [...state.actionsPending].some(key=>key.startsWith(`${id}:`)))return;
  if(!confirm('Confirm the previous executor has stopped. End this execution and release its session? This will not resume its model or tool work.'))return;
  const key=`${id}:end-stopped`,path=`/api/executions/${enc(id)}`;
  const selected=()=>generation===state.generation && (active?id===state.observedExecution:executionGeneration===state.executionGeneration && id===state.selectedExecution);
  state.endReadbackId=id;state.actionsPending.add(key);setDisabled();let noticeRevision=notice('Ending stopped execution…');
  try {
    let info=await api(path), cancellationAccepted=false;
    if(!selected())return;
    if(!terminal(info.status) && info.status!=='CANCELLING') {
      const result=await mutate(`${path}/cancel`,{},{scope:'end-stopped'});
      cancellationAccepted=result.execution_id===id;
      if(!selected())return;
      info=await api(path);
      if(!selected())return;
    }
    // A successful cancel durably records intent even while unknown effects keep RECOVERY_REQUIRED.
    if(info.status==='CANCELLING' || info.status==='RECOVERY_REQUIRED' && cancellationAccepted) {
      await mutate(`${path}/recover`,{},{scope:'end-stopped'});
      if(!selected())return;
      info=await api(path);
      if(!selected())return;
    }
    if(!terminal(info.status))throw new Error(`Execution is still ${info.status}; ending is not confirmed. Check Recovery for unresolved external effects.`);
    const session=info.session_id ? (await api(sessionURL(info.session_id,'',{include_timeline:'false'}))).session : null;
    if(!selected())return;
    if(session?.active_execution_id===id)throw new Error('Execution is terminal, but its session release is not confirmed.');
    state.endReadbackId=null;if(id===state.selectedExecution){state.execution=info;renderExecution();}if(id===state.observedExecution)state.observedInfo=info;
    if(session && state.selectedSession===info.session_id){state.session=session;renderSessionHeader();}
    const completed=session?.active_execution_id ? 'Execution ended; another execution now owns the session.' : session ? 'Execution ended and session released.' : 'Execution ended.';
    if(noticeRevision===state.noticeRevision)noticeRevision=notice(completed);
    try {await refreshSelected();}
    catch(error){if(selected() && noticeRevision===state.noticeRevision){error.message=`${completed} Display refresh failed: ${error.message}`;showError(error);}}
  } catch(error) {
    if(!selected() || noticeRevision!==state.noticeRevision)return;
    error.message+='; choose End stopped execution again to check current state before continuing. No action was automatically resent.';
    showError(error);
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
async function fromHash() {
  if(!state.config)return;
  const params=new URLSearchParams(location.hash.slice(1));
  const view=params.get('view') || (params.has('session')?'sessions':params.has('execution')?'executions':'sessions');
  setView(['sessions','executions','metrics'].includes(view)?view:'sessions',false);
  if(state.view==='metrics')return;
  if(state.view==='executions') {
    const id=params.get('execution') || state.browserExecution;
    if(id){showInspector(true,true);await selectExecution(id);}
    else {++state.executionGeneration;state.selectedExecution=null;state.execution=null;$('execution-meta').textContent='Choose an execution';clear('inspector-content');renderLive();setDisabled();}
    return;
  }
  await openSelection(params.get('session'),params.get('execution'));
}
window.addEventListener('hashchange',()=>fromHash().catch(showError));window.addEventListener('beforeunload',stopStream);
$('new-session').onclick=openNew;$('welcome-new').onclick=openNew;$('new-form').onsubmit=createSession;
$('composer').onsubmit=sendMessage;$('prompt').onkeydown=event=>{if(event.key==='Enter'&&(event.ctrlKey||event.metaKey)){event.preventDefault();$('composer').requestSubmit();}};
$('menu').onclick=()=>$('sidebar').classList.toggle('open');
$('options-toggle').onclick=()=>{const open=$('composer-options').hidden;$('composer-options').hidden=!open;$('options-toggle').setAttribute('aria-expanded',String(open));};
$('inspector-toggle').onclick=()=>showInspector($('inspector').hidden,true);
$('inspector-close').onclick=()=>showInspector(false,true);
$('show-active').onclick=()=>{const id=state.session?.active_execution_id;if(id){showInspector(true,true);selectExecution(id).catch(showError);}};
$('action-menu').onkeydown=event=>{if(event.key==='Escape'){$('action-menu').open=false;$('actions-toggle').focus();}};
$('open-form').onsubmit=event=>{Promise.resolve(openExact(event)).catch(showError);};
$('filter').oninput=renderList;$('list-action').onchange=setListFilterAvailability;
$('filter-form').onsubmit=event=>{event.preventDefault();(state.view==='executions' && $('list-action').value==='recent'?loadRecentExecutions():loadList()).catch(showError);};
$('list-more').onclick=()=>loadList(true).catch(showError);$('turns-more').onclick=()=>moreTurns().catch(showError);
$('detail-more').onclick=()=>loadDetail(true,state.detailSelectors).catch(showError);
$('refresh').onclick=()=>{if(!state.cancellationNoticeId)notice('');(async()=>{if(state.view==='metrics'){await loadMetrics();return;}const generation=state.generation,executionGeneration=state.executionGeneration;await refreshSelected();if(generation===state.generation && executionGeneration===state.executionGeneration)await syncObservation({refresh:true});})().catch(showError);};
$('settings').onclick=showSettings;$('export').onclick=()=>exportResult().catch(showError);
$('rename-session').onclick=()=>sessionAction('update').catch(showError);$('fork-session').onclick=()=>sessionAction('fork').catch(showError);$('close-session').onclick=()=>sessionAction('close').catch(showError);
$('cancel').onclick=()=>executionAction('cancel',{active:true}).catch(showError);$('stop-run').onclick=()=>executionAction('cancel').catch(showError);$('retry').onclick=()=>executionAction('retry').catch(showError);$('fork-run').onclick=()=>executionAction('fork').catch(showError);$('end-stopped').onclick=()=>endStoppedExecution();$('end-active').onclick=()=>endStoppedExecution({active:true});
$('metrics-form').onsubmit=event=>loadMetrics(event).catch(showError);
document.querySelectorAll('[data-view]').forEach(node=>node.onclick=()=>navigateView(node.dataset.view));
document.querySelectorAll('[data-tab]').forEach(node=>node.onclick=()=>{state.tab=node.dataset.tab;loadDetail().catch(showError);});
document.querySelectorAll('[data-close]').forEach(node=>node.onclick=()=>$(node.dataset.close).close());
setDisabled();
bootstrap().catch(showError);
