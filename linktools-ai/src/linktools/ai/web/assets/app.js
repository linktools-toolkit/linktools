import {terminal, modelKey, eventKey, upsertModel, mergePage, readSSE, metricValue, duration, modelLabel, usageLabel, promptLayers, clientAttemptKey, clientBodyBytes, clientBodyText, updateStreamAttempts, streamAttemptText, streamResponsePreview} from './console.js';
import {renderMarkdown, inputPresentation, capturedRequest} from './message.js';

const $ = id => document.getElementById(id);
const state = {config:null, view:'sessions', drafts:new Map(), conversationExecution:null, browserExecution:null, filters:new Map(), list:[], listCursor:null, listSessionReads:new Map(), session:null, turns:[], turnCursor:null, hasEarlierTurns:false, timelineError:'',focusComposerSession:null,
  execution:null, executionGeneration:0, observedExecution:null, observedInfo:null, observationGeneration:0, observationReadGeneration:0, tab:'models', traceSelection:null, conversationMode:'chat', nativeTimeline:null, readerSelection:null, readerRaw:false, readerHttp:false, readerCache:null, readerExpanded:new Set(), details:[], detailCursor:null, models:new Map(), events:new Map(),
  generation:0, refreshGeneration:0, sessionReadGeneration:0, sessionAppliedGeneration:0, listGeneration:0, detailGeneration:0, metricsGeneration:0, stream:null, cursor:null,
  liveText:'', liveThinking:'', liveTools:new Map(), liveAttempts:new Map(), liveActivityOpen:false, pending:new Map(), actionsPending:new Set(), pendingForks:new Map(), recoveryReadbackId:null, endReadbackId:null, cancellationNoticeId:null, noticeRevision:0, selectedSession:null, selectedExecution:null};
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
  node.append(body,button('Copy',async()=>{
    if(!navigator.clipboard)throw new Error('Clipboard unavailable.');
    await navigator.clipboard.writeText(node.markdownSource);
  },'quiet copy-message'));
  node.markdownBody=body;updateMarkdownMessage(node,source);
  return node;
}
function updateMarkdownMessage(node,source) {
  if(node.markdownSource===source)return;
  node.markdownBody.innerHTML=renderMarkdown(source);
  node.markdownSource=source;
}
function userMessage(value) {
  const input=inputPresentation(value), node=element('div','user-input');
  if(input.text)node.append(markdownMessage(input.text,'user'));
  else node.append(element('p','muted',input.attachments.length?'Attached input':'Structured input'));
  if(input.attachments.length) {
    const files=element('ul','attachments');
    input.attachments.forEach(item=>files.append(element('li','',item)));
    node.append(files);
  }
  return node;
}
function clear(id) { $(id).replaceChildren(); }
function requestFocusWithin(container) {
  return container.contains(document.activeElement) ? document.activeElement?.dataset?.requestFocus : null;
}
function restoreRequestFocus(container,key) {
  if(!key)return;
  [...container.querySelectorAll('[data-request-focus]')].find(node=>node.dataset.requestFocus===key)?.focus({preventScroll:true});
}
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
  const closingActivity=!open && state.view==='sessions' && state.conversationMode==='activity';
  if(closingActivity){state.conversationMode='chat';renderPage();}
  if(open && state.view==='sessions' && state.conversationMode==='original'){state.conversationMode='chat';renderPage();renderConversation();}
  if(state.view==='sessions')state.conversationInspectorOpen=open;
  if(state.view==='executions')open=true;
  if(!open)++state.detailGeneration;
  $('inspector').hidden=!open;$('inspector-toggle').setAttribute('aria-expanded',String(open));
  if(!open)$('run-actions').open=false;
  if(focus){const target=open?(state.view==='executions'?$('execution-meta'):$('inspector-close')):$(closingActivity?'activity-mode':'actions-toggle');target.focus();if(open)target.scrollIntoView({block:'nearest'});}
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
  $('current-status').hidden=label==='Session available' && !['CANCELLED','FAILED'].includes(state.observedInfo?.status);
  $('recovery-shortcut').hidden=!owner || !['RECOVERY_REQUIRED','CANCELLING','START_UNKNOWN'].includes(info?.status);
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
  $('open-recovery').disabled=readonly || !state.selectedExecution;
  ['new-session','welcome-new'].forEach(id => $(id).disabled=readonly);
  ['send','planning','thinking','prompt'].forEach(id => $(id).disabled=readonly || state.session?.status === 'CLOSED' || !state.selectedSession);
  ['rename-session','fork-session','close-session'].forEach(id => $(id).disabled=readonly || !state.session);
  const ending=state.actionsPending.has(`${state.selectedExecution}:end-stopped`);
  const recoverable=['RECOVERY_REQUIRED','PENDING_START','STARTED','CANCELLING'].includes(state.execution?.status);
  ['retry','fork-run','end-stopped'].forEach(id => $(id).disabled=readonly || !state.execution || ending);
  $('end-stopped').disabled ||= !recoverable && state.endReadbackId!==state.selectedExecution;
  $('end-stopped').hidden=readonly || !state.execution || !recoverable && state.endReadbackId!==state.selectedExecution;
  $('end-stopped').textContent=state.execution?.session_id ? 'End previous execution and free session' : 'End stopped execution';
  if($('resume-run'))$('resume-run').disabled=readonly || !recoverable || ending || state.recoveryReadbackId===state.selectedExecution || state.actionsPending.has(`${state.selectedExecution}:recover`);
  $('export').disabled=!state.execution || !terminal(state.execution.status);
  $('run-recovery').disabled=!state.execution;
  $('detail-view').disabled=!state.execution;
  $('native-mode').disabled=!state.selectedExecution;$('activity-mode').disabled=!state.selectedExecution;
  $('detail-turn-picker').hidden=state.view!=='sessions';
  const needsRecovery=['RECOVERY_REQUIRED','START_UNKNOWN','CANCELLING'].includes(state.execution?.status);
  $('detail-attention').hidden=!needsRecovery && state.execution?.status!=='FAILED';
  $('detail-attention-text').textContent=state.execution?.status==='FAILED'?`Failed${state.execution.error_code?' · '+state.execution.error_code:''}. Actions offers retry or fork.`:needsRecovery?`${state.execution.status.replaceAll('_',' ')}${state.execution.error_code?' · '+state.execution.error_code:''}. Review the recorded state before continuing.`:'';
  $('detail-recovery').hidden=!needsRecovery;
  $('recovery-controls').hidden=state.tab!=='recovery';
  $('session-details').hidden=state.tab!=='overview' || state.view!=='sessions' || !state.session || state.execution?.session_id!==state.session.session_id;
  const active=state.session?.active_execution_id===state.observedExecution?state.observedInfo:null;
  $('end-active').hidden=readonly || !active || !['RECOVERY_REQUIRED','PENDING_START','STARTED','CANCELLING'].includes(active.status) && state.endReadbackId!==active.execution_id;
  $('end-active').disabled=Boolean(active && [...state.actionsPending].some(key=>key.startsWith(`${active.execution_id}:`)));
  const stoppable=!readonly && state.execution && !terminal(state.execution.status);
  $('stop-run').hidden=!stoppable;
  $('cancel').hidden=readonly || !active || terminal(active.status);
  $('inspector-active-stop').hidden=readonly || !active || terminal(active.status) || active.execution_id===state.selectedExecution;
  $('inspector-active-stop').disabled=readonly || !active || state.actionsPending.has(`${active.execution_id}:end-stopped`);
  $('cancel').setAttribute('aria-label',`Stop current execution ${active?.execution_id || ''}`);
  $('stop-run').disabled=ending;$('cancel').disabled=Boolean(active && state.actionsPending.has(`${active.execution_id}:end-stopped`));
  $('fork-session').disabled=readonly || !state.session || state.actionsPending.has(`session:${state.selectedSession}:fork`);
  $('turns-more').disabled=state.actionsPending.has(`session:${state.selectedSession}:turns`);
}
function renderPage() {
  const sessions=state.view==='sessions', executions=state.view==='executions', metrics=state.view==='metrics';
  $('metrics-view').hidden=!metrics;
  $('conversation-heading').hidden=!sessions || !state.selectedSession;
  $('page-heading').hidden=sessions && Boolean(state.selectedSession);
  $('conversations-home').hidden=sessions;
  $('conversation-view').hidden=metrics || sessions && !state.selectedSession;
  $('conversation-column').hidden=!sessions || state.conversationMode==='activity';
  $('conversation-nav').hidden=!sessions || !state.selectedSession;
  for(const [id,mode] of [['chat-mode','chat'],['native-mode','original'],['activity-mode','activity']])$(id).classList.toggle('active',state.conversationMode===mode);
  $('welcome').hidden=!sessions || Boolean(state.selectedSession);
  $('execution-list').hidden=!executions;
  $('browse-panel').hidden=metrics;
  (executions?$('execution-list'):$('sidebar-browse')).append($('browse-panel'));
  $('composer').hidden=!sessions || !state.selectedSession || state.conversationMode!=='chat';
  $('conversation-column').classList.toggle('reading',state.conversationMode==='original');
  const cancelHome=$(sessions && state.conversationMode!=='chat'?'reader-actions':'composer-actions');if($('cancel').parentElement!==cancelHome)cancelHome.append($('cancel'));
  $('session-details').hidden=!sessions;
  $('inspector-close').hidden=executions;
  $('inspector').classList.toggle('execution-page',executions || sessions && state.conversationMode==='activity');
  if(executions || sessions && state.conversationMode==='activity')showInspector(true);
  $('execution-live').hidden=!executions;
  if($('live'))clear('live');clear('execution-live');renderLive();
}
function closeSidebar(focus=false) {
  $('sidebar').classList.remove('open');$('menu').setAttribute('aria-expanded','false');
  if(focus)$('menu').focus();
}
function setView(view, refreshSelection=true) {
  $('run-actions').open=false;closeSidebar();
  if(view!==state.view){saveDraft();++state.generation;state.nativeTimeline=null;state.filters.set(state.view,$('filter').value);$('filter').value=state.filters.get(view) || '';state.list=[];clear('list');}
  state.view=view;
  $('workspace-tools').open=false;
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
  const sessionReadGeneration=view==='sessions'?++state.sessionReadGeneration:0;
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
  if(view==='sessions'){
    if(!more)state.listSessionReads.clear();
    payload.items.forEach(item=>state.listSessionReads.set(item.session_id,sessionReadGeneration));
  }
  state.listCursor=payload.next_cursor; $('list-more').hidden=!state.listCursor; $('list-scope').hidden=!payload.recent_only;
  $('list-scope').textContent='Showing recent sessions in read-only mode';
  renderList();
}
function setListFilterAvailability() {
  ['filter-agent','filter-session','filter-parent'].forEach(id=>$(id).disabled=$('list-action').value==='recent');
}
async function loadRecentExecutions() {
  if(state.actionsPending.has('recent-scan'))return;
  state.actionsPending.add('recent-scan');const generation=++state.listGeneration,noticeRevision=state.noticeRevision;
  try {
    const payload=await api('/api/executions?recent=true&limit=20').catch(error=>{
      if(generation===state.listGeneration && state.view==='executions' && noticeRevision===state.noticeRevision)throw error;
      return null;
    });
    if(!payload || generation!==state.listGeneration || state.view!=='executions')return;
    state.list=payload.items;state.listCursor=null;$('list-more').hidden=true;
    $('execution-filter-summary').textContent='Execution filters · Newest 20';
    $('list-scope').textContent='Newest 20 by creation time · scanned visible execution metadata';$('list-scope').hidden=false;
    renderList();
  } finally {state.actionsPending.delete('recent-scan');}
}
function renderList() {
  const focus=$('list').contains(document.activeElement)?document.activeElement?.dataset?.listKey:null;
  clear('list'); const filter=$('filter').value.trim().toLowerCase();
  const items=state.view==='sessions' ? state.list.map(item=>{
    if(item.session_id!==state.session?.session_id)return item;
    // Session revisions cover metadata, while admission/release can keep the same revision.
    const newer=state.session.revision>item.revision || state.session.revision===item.revision && state.sessionAppliedGeneration>=(state.listSessionReads.get(item.session_id) || 0);
    return newer?state.session:item;
  }) : state.list;
  const rows=items.filter(item => JSON.stringify(item).toLowerCase().includes(filter));
  if (!rows.length) $('list').append(element('p','empty',filter ? 'No matches in loaded rows.':'Nothing here yet.'));
  rows.forEach(item => {
    const session=state.view==='sessions', id=session ? item.session_id:item.execution_id;
    const active=session ? id===state.selectedSession:id===state.selectedExecution;
    const node=button('',async () => navigate(session ? id:null,session ? null:id),`list-item${active?' active':''}`);node.dataset.listKey=`${state.view}:${id}`;
    node.append(element('span','list-title',session ? item.metadata?.title || short(id) : short(id)));
    const sub=element('span','list-subtitle');
    sub.append(element('i',`status-dot ${item.status==='FAILED'?'failed':item.active_execution_id || !terminal(item.status) && !session ? 'running':''}`));
    sub.append(element('span','',session ? `${item.active_execution_id?'Running · ':item.status==='CLOSED'?'Closed · ':''}${item.agent_id || 'Agent'}` : `${item.status} · ${item.agent_id || 'task'} · ${short(id)}`));
    node.title=`${item.status} · ${item.agent_id || 'task'} · ${id}`;
    node.append(sub); $('list').append(node);
  });
  if(focus)[...$('list').children].find(node=>node.dataset.listKey===focus)?.focus({preventScroll:true});
}
function stopStream() { state.stream?.abort(); state.stream=null;cancelLiveRender(); }
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
async function readSession(id, metadataOnly=false) {
  const readGeneration=++state.sessionReadGeneration;
  try {return {...await api(sessionURL(id,'',metadataOnly?{include_timeline:'false'}:{limit:'50'})),readGeneration};}
  catch(error) {
    if(metadataOnly || error.code!=='SESSION_HISTORY_UNAVAILABLE')throw error;
    const payload=await api(sessionURL(id,'',{include_timeline:'false'}));
    return {...payload,timeline_error:error.message,readGeneration};
  }
}
function acceptSession(payload) {
  if(payload.session.session_id!==state.selectedSession || payload.readGeneration<state.sessionAppliedGeneration)return false;
  state.session={...payload.session};state.sessionAppliedGeneration=payload.readGeneration;return true;
}
function renderSessionHeader() {
  if(!state.session)return;
  $('conversation-title').textContent=state.session.metadata?.title || 'Conversation';
  $('conversation-meta').textContent=`${state.session.agent_id}${state.session.status==='CLOSED'?' · CLOSED':''}${state.session.history_quality!=='complete'?' · history '+state.session.history_quality:''}`;
  $('session-details').replaceChildren(rawDetail(state.session,'Session metadata'));
}
const thinkingLevels=['low','medium','high','xhigh','max'];
const thinkingLabels=['Low','Medium','High','Extra high','Maximum'];
function readThinking() {
  return thinkingLevels[Number($('thinking').value)] || 'low';
}
function renderThinking() {
  const label=thinkingLabels[thinkingLevels.indexOf(readThinking())];
  $('thinking-value').textContent=label;$('thinking').setAttribute('aria-valuetext',label);
}
function saveDraft() {
  if(!state.selectedSession)return;
  state.drafts.set(state.selectedSession,{prompt:$('prompt').value,planning:$('planning').checked,thinking:$('thinking').value});
}
function restoreDraft(sessionId) {
  const draft=state.drafts.get(sessionId) || {prompt:'',planning:false,thinking:'0'};
  $('prompt').value=draft.prompt;
  $('planning').checked=draft.planning;$('thinking').value=draft.thinking ?? '0';renderThinking();
}
async function openSelection(sessionId, executionId, {preserveInspector=false}={}) {
  const preserveObservation=Boolean(sessionId && sessionId===state.selectedSession);saveDraft();
  const generation=++state.generation,focusComposer=state.focusComposerSession===sessionId;state.focusComposerSession=null;
  if(!preserveObservation){stopStream(); ++state.observationGeneration; state.observedExecution=null;state.observedInfo=null; state.cursor=null; state.liveText=''; state.liveThinking='';state.liveAttempts.clear();state.liveTools.clear(); state.liveActivityOpen=false;state.models.clear(); state.events.clear();}
  state.selectedSession=sessionId || null; state.selectedExecution=executionId || null;
  restoreDraft(state.selectedSession);
  state.nativeTimeline=null;
  state.session=null; state.execution=null; state.turns=[]; state.turnCursor=null; state.hasEarlierTurns=false;state.timelineError='';
  const explicitExecution=Boolean(executionId) || state.view==='executions';$('action-menu').open=false;$('run-actions').open=false;
  closeSidebar();renderPage();
  if(!preserveInspector)showInspector(explicitExecution,explicitExecution && Boolean(sessionId || executionId));
  $('conversation-title').textContent='Loading…';
  $('conversation-meta').textContent=''; clear('session-details'); if(!preserveObservation)clear('conversation'); clear('inspector-content');
  setDisabled(); renderList(); notice(state.config?.read_only ? 'Read-only. Configure a model at startup to send messages.':'');
  if (!sessionId && !executionId) return;
  try {
    if (sessionId) {
      const payload=await readSession(sessionId);
      if (generation!==state.generation) return;
      if(acceptSession(payload)){state.turns=payload.timeline?.items || [];state.turnCursor=payload.timeline?.next_cursor || null;state.timelineError=payload.timeline_error || '';}
      else if(!state.session)return;
      renderSessionHeader();
      executionId=executionId || state.session.active_execution_id || state.turns.at(-1)?.execution_id;
      renderConversation();
    } else {
      $('conversation-title').textContent='Execution'; $('conversation-meta').textContent=executionId;renderConversation();
    }
    if (executionId) await selectExecution(executionId,generation);
    setDisabled(); renderList();
    if(focusComposer && generation===state.generation && state.selectedSession===sessionId)$('prompt').focus();
  } catch(error) { if (generation===state.generation) showError(error); }
}
async function moreTurns() {
  const generation=state.generation,id=state.selectedSession,cursor=state.turnCursor,noticeRevision=state.noticeRevision,key=`session:${id}:turns`;
  if(!id || !cursor || state.actionsPending.has(key))return;
  state.actionsPending.add(key);$('turns-more').disabled=true;
  try {
    const payload=await api(sessionURL(id,"",{limit:"50",cursor}));
    if(generation!==state.generation || id!==state.selectedSession)return;
    // Earlier pages supply history only; their session owner may predate live readback.
    state.turns=mergePage(payload.timeline.items,state.turns,item=>item.execution_id);state.turnCursor=payload.timeline.next_cursor;state.hasEarlierTurns=true;
    renderConversation({prepend:true});
  } catch(error) {
    if(generation===state.generation && id===state.selectedSession && noticeRevision===state.noticeRevision)throw error;
  } finally {state.actionsPending.delete(key);$('turns-more').disabled=state.actionsPending.has(`session:${state.selectedSession}:turns`);}
}
function renderTurnPickers() {
  const options=state.turns.map((turn,index)=>({id:turn.execution_id,label:`Turn ${index+1} · ${date(turn.created_at)} · ${turn.status} · ${short(turn.execution_id)}`}));
  if(state.selectedExecution && !options.some(option=>option.id===state.selectedExecution))options.push({id:state.selectedExecution,label:`Execution ${short(state.selectedExecution)}`});
  const signature=json(options);
  for(const id of ['detail-turn','request-turn']) {
    const select=$(id);
    if(!select)continue;
    if(select.dataset.options!==signature){
      select.replaceChildren();options.forEach(({id:executionId,label})=>{const option=element('option','',id==='request-turn'?(label.startsWith('Turn ')?label.split(' · ')[0]:'Execution'):label);option.value=executionId;option.title=label;option.setAttribute('aria-label',label);select.append(option);});select.dataset.options=signature;
    }
    select.value=state.selectedExecution || '';select.title=options.find(option=>option.id===state.selectedExecution)?.label || '';select.disabled=!options.length;
  }
}
function renderConversation({prepend=false}={}) {
  const liveFocus=document.activeElement?.dataset?.liveFocus,requestFocus=requestFocusWithin($('conversation'));
  const container=$('conversation'), previousTop=container.scrollTop, previousHeight=container.scrollHeight;
  const follows=previousTop+container.clientHeight>=previousHeight-40, original=state.conversationMode==='original';
  clear('conversation'); $('turns-more').hidden=!state.turnCursor;
  renderTurnPickers();
  for(const [id,mode] of [['chat-mode','chat'],['native-mode','original'],['activity-mode','activity']])$(id).classList.toggle('active',state.conversationMode===mode);
  if(state.timelineError)container.append(element('p','empty',`${state.timelineError} · Inspect retained executions; loaded turns may be incomplete.`));
  if(original) {
    const section=element('section','model-requests');renderNativeTimeline(section,{conversation:true});container.append(section);renderTurnPickers();
  } else state.turns.forEach(turn => {
    const section=element('article','turn');
    section.append(userMessage(turn.user_input));
    const replies=turn.items.filter(item=>item.item_kind!=='user');
    replies.forEach(item => {
      if (item.item_kind==='assistant') {
        section.append(typeof item.content==='string'?markdownMessage(item.content):rawDetail(item.content,'Structured output'));
      } else if (['tool_call','tool_result','thinking'].includes(item.item_kind)) {
        const detail=element('details','tool-message'); detail.append(element('summary','',`${item.item_kind.replaceAll('_',' ')} · ${item.tool_name || ''}`),element('pre','',text(item.content))); section.append(detail);
      }
    });
    if (!turn.conversation_committed) section.append(element('p','muted','Incomplete turn; history is not yet committed.'));
    if (turn.error_code) section.append(element('p','muted',turn.error_code==='EXECUTION_CANCELLED' ? 'Execution cancelled.' : `${turn.error_code}${Object.keys(turn.safe_error_details || {}).length?' · '+text(turn.safe_error_details):''}`));
    const heading=element('div','turn-heading');heading.append(element('span','',date(turn.created_at)));
    if(turn.status!=='SUCCEEDED')heading.append(element('span','badge',turn.status));
    section.append(heading);container.append(section);
  });
  if (!state.turns.length && !state.timelineError && !original)container.append(element('p','empty',state.selectedSession ? 'Send a message to get started.':'Choose an execution to inspect.'));
  const live=element('section','turn');live.id='live';
  if(original){const preview=element('details','live-preview');preview.append(element('summary','','Live preview · not original response'),live);container.append(preview);}else container.append(live);
  renderLive(liveFocus);restoreRequestFocus(container,requestFocus);
  container.scrollTop=original?previousTop:prepend?previousTop+container.scrollHeight-previousHeight:follows?container.scrollHeight:previousTop;
}

let liveRenderTimer=null,liveRenderFrame=null,liveRenderCost=0,previousLiveRenderCost=0;
const liveFrameBudget=1000/60;
function cancelLiveRender() {
  if(liveRenderTimer!==null){clearTimeout(liveRenderTimer);liveRenderTimer=null;}
  if(liveRenderFrame!==null){cancelAnimationFrame(liveRenderFrame);liveRenderFrame=null;}
}
function scheduleLiveRender() {
  if(liveRenderTimer!==null || liveRenderFrame!==null)return;
  const frame=()=>{liveRenderTimer=null;liveRenderFrame=requestAnimationFrame(()=>{liveRenderFrame=null;renderLive();});};
  // Sustained expensive parsing retains batching without penalizing one slow frame.
  if(Math.min(liveRenderCost,previousLiveRenderCost)>liveFrameBudget)liveRenderTimer=setTimeout(frame,60-liveFrameBudget);
  else frame();
}
function acceptLiveResponse(item) {
  const request=state.liveAttempts.get(modelKey(item));
  if(!request || item.status!=='SUCCEEDED' || item.content_included!==true || !Array.isArray(item.response?.parts))return false;
  request.canonicalResponse=item.response;request.canonicalError=null;request.finished=true;request.status='SUCCEEDED';
  for(const attempt of request.attempts.values())if(['RUNNING','SUCCEEDED'].includes(attempt.status)){
    attempt.text='';attempt.thinking='';attempt.parts.clear();
  }
  scheduleLiveRender();return true;
}
async function readLiveResponse(request,force=false) {
  if(request.canonicalResponse || request.canonicalPending || request.canonicalError && !force)return;
  request.canonicalPending=true;request.canonicalError=null;scheduleLiveRender();
  const generation=state.observationGeneration;
  const current=()=>generation===state.observationGeneration && state.liveAttempts.get(request.key)===request;
  try {
    const params=new URLSearchParams({include_content:'true',limit:'1',agent_run_seq:request.agent_run_seq,model_request_seq:request.model_request_seq});
    const page=await api(`/api/executions/${enc(request.execution_id)}/models?${params}`);
    if(!current())return;
    const item=page.items.find(item=>modelKey(item)===request.key);
    if(!item || !acceptLiveResponse(item))request.canonicalError='Committed response content unavailable';
  } catch(error) {if(current())request.canonicalError='Committed response read unavailable';}
  finally {if(current()){request.canonicalPending=false;scheduleLiveRender();}}
}
function renderLiveAttempts(live,wanted) {
  live.attemptNodes ||= new Map();const retained=new Set();
  for(const request of state.liveAttempts.values()) {
    if(request.execution_id!==state.observedExecution)continue;
    const attempts=[...request.attempts.values()].sort((a,b)=>a.seq-b.seq);
    for(const attempt of attempts.filter(item=>['FAILED','CANCELLED'].includes(item.status))) {
      const key=request.key+':'+attempt.seq;retained.add(key);
      let node=live.attemptNodes.get(key);
      if(!node) {
        node=element('details','tool-message failed-attempt');node.open=attempt.expanded || false;
        node.ontoggle=()=>{if(node.isConnected)attempt.expanded=node.open;};
        node.label=element('summary');node.body=element('pre');node.thinking=element('pre');node.append(node.label,node.thinking,node.body);
        live.attemptNodes.set(key,node);
      }
      node.label.textContent=`${attempt.status} · run ${request.agent_run_seq} · request ${request.model_request_seq} · attempt ${attempt.seq} · partial content is not an answer`;
      const content=streamAttemptText(attempt),thinking=streamAttemptText(attempt,'thinking');
      if(node.body.textContent!==content)node.body.textContent=content;
      if(node.thinking.textContent!==thinking)node.thinking.textContent=thinking;
      node.body.hidden=!content;node.thinking.hidden=!thinking;wanted.push(node);
    }
    const contributing=attempts.filter(item=>item.status==='SUCCEEDED' || item.seq===request.current && item.status==='RUNNING');
    if(!contributing.length)continue;
    const key=request.key+':response';retained.add(key);
    let node=live.attemptNodes.get(key);
    if(!node) {
      node=element('section','stream-attempt');node.label=element('div','message-label');
      node.thinking=element('details','tool-message');node.thinkingBody=element('pre');node.thinking.append(element('summary','','Thinking · SDK display'),node.thinkingBody);
      node.body=markdownMessage('');node.append(node.label,node.thinking,node.body);live.attemptNodes.set(key,node);
    }
    const preview=streamResponsePreview(request),canonical=request.canonicalResponse;
    const responseText=kind=>canonical.parts.filter(part=>part.part_kind===kind && typeof part.content==='string').map(part=>part.content).join('');
    const content=canonical?responseText('text'):request.status==='SUCCEEDED'?'':preview.text;
    const thinking=canonical?responseText('thinking'):request.status==='SUCCEEDED'?'':preview.thinking;
    const status=canonical?'COMPLETED MODEL RESPONSE · committed SDK text':request.status==='SUCCEEDED'?`MODEL REQUEST SUCCEEDED · ${request.canonicalError || 'awaiting committed response'}`:['FAILED','CANCELLED'].includes(request.status)?`INCOMPLETE RESPONSE · ${request.status}`:'LIVE RESPONSE · incomplete SDK display';
    const label=`${status} · run ${request.agent_run_seq} · request ${request.model_request_seq}${!canonical && preview.truncated?' · preview GAP · noncontiguous fragment':''}`;
    if(node.label.textContent!==label)node.label.textContent=label;
    node.thinking.hidden=!thinking;if(node.thinkingBody.textContent!==thinking)node.thinkingBody.textContent=thinking;
    if(!node.reload){node.reload=button('Load committed response',()=>readLiveResponse(request,true));node.append(node.reload);}
    node.reload.hidden=request.status!=='SUCCEEDED' || Boolean(canonical);node.reload.disabled=Boolean(request.canonicalPending);
    updateMarkdownMessage(node.body,content);node.body.hidden=!content;wanted.push(node);
  }
  for(const key of live.attemptNodes.keys())if(!retained.has(key))live.attemptNodes.delete(key);
}
function updateLiveTools(item) {
  const {event}=item,payload=event.payload || {};
  if(['EXECUTION_SUCCEEDED','EXECUTION_FAILED','EXECUTION_CANCELLED','EXECUTION_RECOVERY_REQUIRED','EXECUTION_START_UNKNOWN'].includes(event.event_type)) {
    for(const call of state.liveTools.values())if(call.execution_id===item.execution_id && !call.finished)call.unconfirmed=true;
  }
  if(!['TOOL_CALL_STARTED','TOOL_CALL_FINISHED'].includes(event.event_type) || !payload.call_id)return;
  const key=JSON.stringify([item.execution_id,payload.agent_run_seq,payload.call_id]);
  const previous=state.liveTools.get(key),finished=event.event_type==='TOOL_CALL_FINISHED';
  // Replayed starts must not undo an observed result for the same call.
  if((previous?.finished || previous?.unconfirmed) && !finished)return;
  state.liveTools.set(key,{...item,finished});
}
function renderLiveTools(live,wanted) {
  if(!state.liveTools.size)return;
  if(!live.toolsNode){
    live.toolsNode=element('section','live-tools');live.toolsNode.setAttribute('aria-label','Tool calls');
    live.toolNodes=new Map();
  }
  const calls=[...state.liveTools],recent=new Set(calls.filter(([,item])=>item.finished).slice(-12).map(([key])=>key)),rows=[];
  for(const [key,item] of calls) {
    if(item.finished && !recent.has(key))continue;
    let row=live.toolNodes.get(key);
    if(!row){
      row=element('div','live-tool');row.name=element('span','tool-name');row.status=element('span','tool-status');
      row.status.setAttribute('role','status');row.status.setAttribute('aria-live','polite');row.append(row.name,row.status);live.toolNodes.set(key,row);
    }
    const payload=item.event.payload || {},name=`${item.depth?'↳ ':''}${payload.tool_name || 'Tool'}${item.depth?' · '+(item.agent_id || 'subagent'):''}`;
    const status=item.finished ? ({SUCCEEDED:'Completed',FAILED:'Failed',CANCELLED:'Cancelled'}[payload.status] || 'Finished · outcome unconfirmed') : item.unconfirmed?'Outcome unconfirmed':'Started · awaiting result';
    const detail=status+(payload.error_code?' · '+payload.error_code:'');
    if(row.name.textContent!==name)row.name.textContent=name;
    if(row.status.textContent!==detail)row.status.textContent=detail;
    row.classList.toggle('error',item.finished && ['FAILED','CANCELLED'].includes(payload.status));rows.push(row);
  }
  for(const [key,row] of live.toolNodes)if(!rows.includes(row))live.toolNodes.delete(key);
  if(rows.length!==live.toolsNode.children.length || rows.some((row,index)=>row!==live.toolsNode.children[index]))live.toolsNode.replaceChildren(...rows);
  wanted.push(live.toolsNode);
}
function renderLive(liveFocus=document.activeElement?.dataset?.liveFocus) {
  cancelLiveRender();const started=performance.now();
  try {
  let live=$('live');
  if(state.view==='executions'){live=$('execution-live');if(state.selectedExecution!==state.observedExecution){live.replaceChildren();return;}}
  if (!live) return;
  const container=$('conversation'), follows=container.scrollTop+container.clientHeight>=container.scrollHeight-40,wanted=[];
  if (state.liveThinking) {
    if(!live.thinkingNode){live.thinkingNode=element('details','tool-message');live.thinkingBody=element('pre');live.thinkingNode.append(element('summary','','Live thinking · incomplete SDK response'),live.thinkingBody);}
    if(live.thinkingSource!==state.liveThinking){live.thinkingBody.textContent=state.liveThinking;live.thinkingSource=state.liveThinking;}
    wanted.push(live.thinkingNode);
  }
  if (state.liveText) {
    if(!live.markdownMessage)live.markdownMessage=markdownMessage(state.liveText);
    else updateMarkdownMessage(live.markdownMessage,state.liveText);
    live.liveLabel ||= element('div','message-label');
    const label=`LIVE RESPONSE · incomplete SDK display · ${state.observedInfo?.agent_id || 'AGENT'} · ${short(state.observedExecution)}`;
    if(live.liveLabel.textContent!==label)live.liveLabel.textContent=label;
    wanted.push(live.liveLabel,live.markdownMessage);
  }
  renderLiveTools(live,wanted);
  renderLiveAttempts(live,wanted);
  const events=[...state.events.values()].slice(-12);
  if(!live.activityEvents || events.length!==live.activityEvents.length || events.some((item,index)=>item!==live.activityEvents[index])) {
  live.activityEvents=events;live.activityFocusNodes=new Map();live.activityNodes=[];
  const activity=element('details','tool-message');activity.id='live-activity';activity.open=state.liveActivityOpen;
  activity.ontoggle=()=>{if($('live-activity')===activity)state.liveActivityOpen=activity.open;};
  const attention=item=>['EXECUTION_START_UNKNOWN','EXECUTION_RECOVERY_REQUIRED','EXECUTION_FAILED','CANCEL_REQUESTED','EXECUTION_CANCELLED','APPROVAL_REQUESTED','EXTERNAL_REQUESTED'].includes(item.event.event_type)
    || ['MODEL_ATTEMPT_FINISHED','MODEL_REQUEST_FINISHED','TOOL_CALL_FINISHED'].includes(item.event.event_type) && ['FAILED','CANCELLED'].includes(item.event.payload?.status);
  const ordinary=events.filter(item=>!attention(item)).length;
  const summary=element('summary','',`Activity · ${ordinary} recent events`);summary.dataset.liveFocus='summary';activity.append(summary);
  if(ordinary)live.activityFocusNodes.set('summary',summary);
  events.forEach(item => {
    const payload=item.event.payload || {};
    const row=element('div',`live-event${item.depth?' child':''}`,`${item.depth?'↳ ':''}${item.agent_id || 'task'} · ${item.event.event_type.replaceAll('_',' ').toLowerCase()}${payload.status?' · '+payload.status:''}${payload.error_code?' · '+payload.error_code:''}${payload.tool_name?' · '+payload.tool_name:''}${payload.call_id?' #'+short(payload.call_id):''}`);
    if(item.depth){const inspect=button('Inspect subagent',async()=>{showInspector(true,true);await selectExecution(item.execution_id);});inspect.dataset.liveFocus=eventKey(item);row.append(inspect);live.activityFocusNodes.set(inspect.dataset.liveFocus,inspect);}
    if(attention(item))live.activityNodes.push(row);else activity.append(row);
  });
  if(ordinary)live.activityNodes.push(activity);
  }
  wanted.push(...live.activityNodes);
  if(wanted.length!==live.children.length || wanted.some((node,index)=>node!==live.children[index])) {
    const focus=document.activeElement,preserve=focus && wanted.some(node=>node.contains(focus));
    live.replaceChildren(...wanted);if(preserve)focus.focus({preventScroll:true});
  }
  live.activityFocusNodes.get(liveFocus)?.focus({preventScroll:true});
  if(follows)container.scrollTop=container.scrollHeight;
  } finally {previousLiveRenderCost=liveRenderCost;liveRenderCost=performance.now()-started;}
}
function observedInfo(info, readGeneration=state.observationReadGeneration) {
  if(info.execution_id!==state.observedExecution)return info;
  const previous=state.observedInfo;
  if(previous && readGeneration!==state.observationReadGeneration)return previous;
  state.observedInfo=info;
  if(terminal(info.status) || ['RECOVERY_REQUIRED','START_UNKNOWN'].includes(info.status)) {
    for(const call of state.liveTools.values())if(!call.finished)call.unconfirmed=true;
  }
  if(['FAILED','CANCELLED'].includes(info.status))updateStreamAttempts(state.liveAttempts,{execution_id:info.execution_id,event:{event_type:'EXECUTION_'+info.status}});
  if(terminal(info.status))for(const request of state.liveAttempts.values())if(request.execution_id===info.execution_id && request.status==='SUCCEEDED')readLiveResponse(request);
  return info;
}
async function selectExecution(id, generation=state.generation) {
  const executionGeneration=++state.executionGeneration,selectionChanged=id!==state.selectedExecution;
  const readGeneration=id===state.observedExecution?++state.observationReadGeneration:state.observationReadGeneration;
  if(state.nativeTimeline?.id!==id){state.nativeTimeline=null;state.readerSelection=null;state.readerRaw=false;state.readerCache=null;state.readerExpanded.clear();}
  if(selectionChanged){$('run-actions').open=false;state.traceSelection=null;}
  state.selectedExecution=id;if(state.view==='sessions')state.conversationExecution=id;else if(state.view==='executions')state.browserExecution=id; state.execution=null;renderTurnPickers();setDisabled();clear('inspector-content');renderLive();
  $('execution-meta').textContent=`Loading ${id}…`;
  if(selectionChanged && state.view==='sessions' && state.conversationMode==='original')renderConversation();
  try {
  let info=await api(`/api/executions/${enc(id)}`);
  if (generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution) return;
  info=observedInfo(info,readGeneration);
  state.execution=info; renderExecution();
  if(state.view==='sessions' && state.conversationMode==='original')await loadNativeTimeline();
  if(generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
  if(state.tab==='models' && state.view==='sessions' && state.conversationMode==='original')renderNativeViews();
  else await loadDetail(false);
  setDisabled();
  if (generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution) await syncObservation();
  } catch(error){
    if(generation===state.generation && executionGeneration===state.executionGeneration && id===state.selectedExecution && (id!==state.observedExecution || readGeneration===state.observationReadGeneration)){
      if(selectionChanged && state.view==='sessions' && state.conversationMode==='original'){state.nativeTimeline={id,items:[],cursor:null,loading:false,error:error.message,pending:new Map()};renderNativeViews();}
      throw error;
    }
  }
}
function renderExecution() {
  const info=state.execution; if (!info) return;
  renderCurrentState();renderTurnPickers();
  $('detail-title').textContent=info.session_id?'Turn details':'Execution details';
  const elapsed=info.started_at && info.terminal_at?duration(Math.max(0,new Date(info.terminal_at)-new Date(info.started_at))*1000000):'Duration unavailable';
  $('execution-meta').replaceChildren(element('strong','',`${info.agent_id || info.task_id || 'Execution'} · ${info.status}`),element('span','run-duration',elapsed));
  const identity=element('div','mono run-identity',short(info.execution_id));identity.title=info.execution_id;$('execution-meta').append(identity);
  if(info.usage)$('execution-meta').append(element('p','run-usage',`${info.usage.logical_requests} requests · ${usageLabel(info.usage)}${info.usage.unknown_usage_requests?' · '+info.usage.unknown_usage_requests+' usage unknown':''}`));
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
function downloadClientBody(value,name) {
  const bytes=clientBodyBytes(value);if(bytes===null)return;
  const url=URL.createObjectURL(new Blob([bytes],{type:'application/octet-stream'})),anchor=document.createElement('a');
  anchor.href=url;anchor.download=name;anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function nativeBody(value,name,{request=false}={}) {
  const container=element('div');
  if(value.body==null){container.append(element('p','muted',`Original body unavailable · ${value.capture_error || 'not observed'}`));return container;}
  const download=button('Download original bytes',async()=>downloadClientBody(value,name));download.dataset.requestFocus=name+':download';container.append(download);
  let source;
  try {source=clientBodyText(value);}
  catch(error){container.append(element('p','muted',`Text display unavailable · ${error.message}. Original bytes remain downloadable.`));return container;}
  const full=element('details');full.open=true;
  const summary=element('summary','',request?'Original outbound request body · native input and envelope':value.content_encoding?'HTTP-decoded response display':'Original response text');summary.dataset.requestFocus=name+':body';
  full.append(summary,element('pre','json',source));
  container.append(full);return container;
}
function literalSource(raw, label='') {
  const node=element('div','native-field');if(label)node.append(element('span','native-field-label',label));node.append(element('pre','native-literal',raw));return node;
}
function fieldCounts(node) {
  const counts=new Map();for(const field of node.fields())counts.set(field.name,(counts.get(field.name)||0)+1);return counts;
}
function nativeContent(container,node) {
  if(node.kind==='string'){container.append(element('pre',node.text?'native-text':'native-empty',node.text || 'content: empty string'));return;}
  if(node.kind!=='array'){container.append(literalSource(node.raw,'content'));return;}
  if(!node.items().length){container.append(element('p','native-empty','content: empty array'));return;}
  node.items().forEach((block,index)=>{
    const row=element('div','native-block'),types=block.fields().filter(field=>field.name==='type');
    const type=types.length===1 && types[0].value.kind==='string'?types[0].value.text:null;
    row.append(element('span','native-field-label',`content[${index}]${type?' · '+type:''}`));
    if(['text','input_text','output_text','summary_text','thinking'].includes(type)){
      const counts=fieldCounts(block);
      for(const field of block.fields()){
        if(field.name==='type')continue;
        if(['text','thinking'].includes(field.name) && counts.get(field.name)===1 && field.value.kind==='string')row.append(element('pre','native-text',field.value.text));
        else row.append(literalSource(field.raw));
      }
    } else {if(type==='redacted_thinking')row.append(element('p','muted','Opaque provider thinking data, not readable reasoning.'));row.append(element('pre','native-literal',block.raw));}
    container.append(row);
  });
}
function nativeToolCalls(container,node,key) {
  if(node.kind!=='array'){container.append(literalSource(node.raw,'tool_calls'));return;}
  if(!node.items().length){container.append(element('p','native-empty','tool_calls: empty array'));return;}
  node.items().forEach((call,index)=>{
    const row=element('div','native-tool'),fields=call.fields(),counts=fieldCounts(call);
    const unique=name=>counts.get(name)===1?fields.find(field=>field.name===name)?.value:null;
    const fn=unique('function'),id=unique('id'),type=unique('type'),members=fn?.fields() || [];
    const member=name=>members.filter(field=>field.name===name);
    const names=member('name'),argumentsFields=member('arguments');
    const compact=fn?.kind==='object' && names.length===1 && names[0].value.kind==='string' && argumentsFields.length===1 && argumentsFields[0].value.kind==='string' && (!counts.has('id') || id?.kind==='string') && (!counts.has('type') || type?.kind==='string' && type.text==='function');
    const heading=element('div','native-tool-heading');row.append(heading);
    if(compact){
      heading.append(element('strong','',names[0].value.text || 'name: empty string'));
      if(id)heading.append(element('span','native-call-id',id.text || 'id: empty string'));
      row.append(element('pre',argumentsFields[0].value.text?'native-literal':'native-empty',argumentsFields[0].value.text || 'arguments: empty string'));
      const extra=readerDetail(`${key}:call:${index}`,'Fields','native-fields');
      extra.append(element('span','native-field-label',`tool_calls[${index}]`));
      if(type)extra.append(literalSource(type.raw,'type'));
      for(const field of fields){
        if(field.name==='function'){for(const value of members.filter(value=>!['name','arguments'].includes(value.name)))extra.append(literalSource(value.raw,'function'));}
        else if(!['id','type'].includes(field.name))extra.append(literalSource(field.raw));
      }
      heading.append(extra);
    } else row.append(element('pre','native-literal',call.raw));
    container.append(row);
  });
}

function nativeMessages(container,projection,key,api) {
  if(api==='openai-responses'){
    const inputs=projection.fields.filter(field=>['instructions','input'].includes(field.name));
    if(!inputs.length)container.append(element('p','muted','No instructions or input items were captured. Provider-side references may supply additional context; retained SDK history does not reconstruct it.'));
    for(const field of inputs){
      container.append(element('h3','',field.name));
      if(field.value.kind!=='array'){nativeContent(container,field.value);continue;}
      field.value.items().forEach((item,index)=>{
        const row=element('article','native-message'),fields=item.fields(),counts=fieldCounts(item);
        const kind=fields.find(value=>value.name==='type' && counts.get('type')===1)?.value;
        row.append(element('strong','',`input[${index}]${kind?.kind==='string'?' · '+kind.text:''}`));
        if(item.kind!=='object')row.append(literalSource(item.raw));
        else for(const value of fields){
          if(value.name==='content' && counts.get('content')===1)nativeContent(row,value.value);
          else if(value.name==='summary' && counts.get('summary')===1){row.append(element('p','muted','Provider-returned reasoning summary'));nativeContent(row,value.value);}
          else {if(value.name==='encrypted_content')row.append(element('p','muted','Opaque provider reasoning data, not readable reasoning.'));row.append(literalSource(value.raw));}
        }
        container.append(row);
      });
    }
    return;
  }
  if(api==='anthropic')for(const field of projection.fields.filter(field=>field.name==='system')){
    container.append(element('h3','','System content'));nativeContent(container,field.value);
  }
  if(projection.root.kind!=='object' || !projection.messages.length){container.append(element('p','muted','This captured body has no messages field. Its original source is shown without inventing messages.'),element('pre','native-literal',projection.source));return;}
  if(projection.messages.length>1)container.append(element('p','source-warning','Multiple messages fields were captured. Each field is shown separately in source order; no effective field is guessed.'));
  projection.messages.forEach((field,group)=>{
    if(projection.messages.length>1)container.append(element('h3','',`messages field ${group+1}`));
    if(field.value.kind!=='array'){container.append(literalSource(field.raw,'messages is not an array'));return;}
    if(!field.value.items().length)container.append(element('p','native-empty','messages: empty array'));
    field.value.items().forEach((message,index)=>{
      const row=element('article','native-message'),heading=element('div','native-role'),body=element('div','native-message-body'),fields=message.fields(),counts=fieldCounts(message),roles=fields.filter(item=>item.name==='role');
      const role=roles.length===1 && roles[0].value.kind==='string'?roles[0].value.text:roles.length>1?'ambiguous role':roles.length?'non-string role':'role absent';
      heading.dataset.role=role;heading.append(element('span','',index),element('strong','',role));row.append(heading,body);
      if(message.kind!=='object')body.append(element('pre','native-literal',message.raw));
      else {
        if([...counts.values()].some(count=>count>1))body.append(element('p','source-warning','Repeated field names are preserved below; no last value is selected.'));
        const extra=readerDetail(`${key}:message:${group}:${index}`,'Fields','native-fields');
        if(!counts.has('content'))(counts.has('tool_calls')?extra:body).append(element('p','native-empty','content field absent'));
        for(const item of fields){
          if(item.name==='role' && roles.length===1 && item.value.kind==='string')continue;
          if(counts.get(item.name)>1){body.append(literalSource(item.raw));continue;}
          if(item.name==='content')nativeContent(counts.has('tool_calls') && item.value.kind==='string' && !item.value.text?extra:body,item.value);
          else if(item.name==='tool_calls')nativeToolCalls(body,item.value,`${key}:message:${group}:${index}`);
          else if(item.name==='tool_call_id' && item.value.kind==='string'){body.classList.add('native-tool-result');body.append(element('span','native-call-id',item.value.text || 'tool_call_id: empty string'));}
          else extra.append(literalSource(item.raw));
        }
        if(extra.children.length>1)body.append(extra);
      }
      container.append(row);
    });
  });
}
function sdkMessages(container,item,key) {
  container.append(element('p','source-note','Model input captured at SDK preparation, reconstructed from retained context. Message and part order are preserved; instructions and tools remain separate SDK fields, not invented provider messages.'));
  for(const attempt of item.stream_attempts || []) {
    const failed=attempt.status!=='SUCCEEDED';
    const detail=readerDetail(key+':stream:'+attempt.stream_attempt_seq,`Model stream attempt ${attempt.stream_attempt_seq} · ${attempt.status}${failed?' · partial content is not an answer':' · successful segment; logical response below'}`,'stream-attempt-history');
    const metadata=Object.fromEntries(Object.entries(attempt).filter(([name])=>name!=='response'));
    detail.append(properties(metadata));
    if(failed)detail.append(attempt.response!=null?rawDetail(attempt.response,'Failed SDK partial response'):element('p','muted','Partial response not retained or not included.'));
    container.append(detail);
  }
  const request=item.request;
  if(item.content_included===false || !Array.isArray(request?.messages)){
    container.append(element('p','muted',item.status==='RUNNING'?'Model input is not committed yet.':'No retained SDK model input is available.'));return;
  }
  request.messages.forEach((message,index)=>{
    const row=element('article','native-message'),heading=element('div','native-role'),body=element('div','native-message-body');
    heading.append(element('span','',index),element('strong','',message.kind || 'message'));row.append(heading,body);
    for(const part of message.parts || []){
      const section=element('section','sdk-part');section.append(element('strong','',part.part_kind || 'part'));
      if(Object.hasOwn(part,'content'))section.append(element('pre','native-literal',text(part.content)));
      const fields=Object.fromEntries(Object.entries(part).filter(([name])=>!['part_kind','content'].includes(name)));
      if(Object.keys(fields).length)section.append(element('pre','native-literal',json(fields)));
      body.append(section);
    }
    const fields=Object.fromEntries(Object.entries(message).filter(([name])=>!['kind','parts'].includes(name)));
    if(Object.keys(fields).length)body.append(rawDetail(fields,'Message fields and instructions'));
    container.append(row);
  });
  const envelope=Object.fromEntries(Object.entries(request).filter(([name])=>name!=='messages'));
  if(Object.keys(envelope).length){const detail=readerDetail(key+':sdk-envelope','Instructions, tools & model settings','native-envelope');detail.append(element('pre','native-literal',json(envelope)));container.append(detail);}
  if(item.response!=null)container.append(rawDetail(item.response,['FAILED','CANCELLED'].includes(item.status)
    ?`Incomplete SDK response · ${item.status} · diagnostic aggregate, not a completed answer`:'SDK response'));
}
function readerDetail(key,label,className) {
  const detail=element('details',className),summary=element('summary','',label);
  detail.open=state.readerExpanded.has(key);summary.dataset.requestFocus=key;
  detail.ontoggle=()=>{if(!detail.isConnected)return;if(detail.open)state.readerExpanded.add(key);else state.readerExpanded.delete(key);};
  detail.append(summary);return detail;
}
function renderNativeTimeline(container,{conversation=false}={}) {
  const view=state.nativeTimeline;
  container.classList.add('native-reader');
  const toolbar=element('div','native-toolbar');container.append(toolbar);
  if(conversation){
    toolbar.id='request-context';
    const turn=element('select','native-turn');turn.id='request-turn';turn.setAttribute('aria-label','Choose turn for model requests');turn.dataset.requestFocus='reader:turn';
    turn.onchange=()=>selectExecution(turn.value).catch(showError);toolbar.append(turn);
  }
  if(!view || view.id!==state.selectedExecution){container.append(element('p','muted','Loading model records…'));return;}
  const choices=[];
  for(const item of view.items){
    if(!state.readerHttp){choices.push({key:modelKey(item),item,attempt:null,index:0});continue;}
    const attempts=item.client_requests || [];
    if(!attempts.length)choices.push({key:modelKey(item)+':unavailable',item,attempt:null,index:0});
    else attempts.forEach((attempt,index)=>choices.push({key:clientAttemptKey(item,index),item,attempt,index}));
  }
  let selected=choices.find(choice=>choice.key===state.readerSelection);
  if(!selected && state.readerSelection?.endsWith(':unavailable'))selected=choices.find(choice=>modelKey(choice.item)+':unavailable'===state.readerSelection);
  if(selected)state.readerSelection=selected.key;
  if(!selected){selected=choices[0];state.readerSelection=selected?.key || null;}
  const select=element('select','native-request');select.setAttribute('aria-label','Select model request and attempt');select.dataset.requestFocus='reader:selection';
  choices.forEach(choice=>{const option=element('option','',`Request ${choice.item.model_request_seq} · ${!state.readerHttp?'SDK input':choice.attempt?'Attempt '+(choice.index+1):'not captured'}${choice.item.agent_run_seq>1?' · run '+choice.item.agent_run_seq:''}${choice.item.purpose && choice.item.purpose!=='agent'?' · '+choice.item.purpose:''}${choice.item.execution_id!==view.id?' · '+short(choice.item.execution_id):''}`);option.value=choice.key;select.append(option);});
  select.value=state.readerSelection || '';select.disabled=!selected;select.onchange=()=>{state.readerExpanded.delete('reader:tools');state.readerSelection=select.value;state.readerCache=null;renderNativeViews();};toolbar.append(select);
  const status=selected?.attempt?.response?.status_code;
  if(status>=400 || selected?.item.status && selected.item.status!=='SUCCEEDED')toolbar.append(element('span','native-status',status>=400?`HTTP ${status}`:selected.item.status));
  const tools=readerDetail('reader:tools','···','native-tools');tools.children[0].setAttribute('aria-label','Request tools');
  const actions=element('div','native-tool-actions');tools.append(actions);toolbar.append(tools);
  const closeTools=()=>{tools.open=false;state.readerExpanded.delete('reader:tools');tools.children[0].focus({preventScroll:true});};
  tools.onkeydown=event=>{if(event.key==='Escape')closeTools();};
  const source=button(state.readerHttp?'Read model input':'Original HTTP diagnostics',async()=>{closeTools();state.readerHttp=!state.readerHttp;state.readerSelection=null;state.readerCache=null;await loadNativeTimeline(false,true);});source.dataset.requestFocus='reader:source';actions.append(source);
  const raw=button(state.readerRaw?'Read messages':'View raw',async()=>{closeTools();state.readerRaw=!state.readerRaw;renderNativeViews();});raw.dataset.requestFocus='reader:raw';raw.disabled=!selected?.attempt;if(state.readerHttp)actions.append(raw);
  if(state.readerHttp && selected?.attempt?.body!=null){const download=button('Download bytes',async()=>{downloadClientBody(selected.attempt,`${selected.key}-request.bin`);closeTools();});download.dataset.requestFocus='reader:download';actions.append(download);}
  if(selected){const info=button('Inspect request',async()=>{const key=modelKey(selected.item)+':metadata';state.readerExpanded.has(key)?state.readerExpanded.delete(key):state.readerExpanded.add(key);closeTools();renderNativeViews();});info.dataset.requestFocus='reader:info';actions.append(info);}
  if(view.error)container.append(element('p','source-warning',view.error));
  if(!selected)container.append(element('p','empty',view.loading?'Loading model records…':'No records for this selection.'));
  else {
    const {item,attempt}=selected;
    const metadata=readerDetail(modelKey(item)+':metadata','Request details','native-envelope native-info');metadata.hidden=!metadata.open;metadata.append(element('p','native-identity',`${modelLabel(item.model)} · ${item.purpose}`),element('p','source-note',state.readerHttp?'SDK-client input, in original array order. View raw and Download bytes retain the source; readable strings are decoded. Provider-server receipt is not observable here. Supported Chat, Responses and Anthropic dispatches are captured; SDK history never fills missing captures.':'SDK preparation evidence, not provider wire bytes.'),element('pre','json',json({execution_id:item.execution_id,agent_run_seq:item.agent_run_seq,model_request_seq:item.model_request_seq,purpose:item.purpose,step_index:item.step_index,output_retry_index:item.output_retry_index,started_at:item.started_at,finished_at:item.finished_at,duration_ns:item.duration_ns,status:item.status,error_code:item.error_code,usage:item.usage,api:attempt?.api,record_http_bodies:attempt?.record_http_bodies,diagnostics:attempt?.diagnostics})));if(item.usage)metadata.append(element('p','muted',usageLabel(item.usage)));container.append(metadata);
    const toggleInfo=metadata.ontoggle;metadata.ontoggle=()=>{toggleInfo();if(!metadata.isConnected)return;metadata.hidden=!metadata.open;if(!metadata.open && metadata.contains(document.activeElement))tools.children[0].focus({preventScroll:true});};
    if(!state.readerHttp)sdkMessages(container,item,selected.key);
    else if(!attempt)container.append(element('p','muted',!item.client_requests_included?'Original client content was not requested.':item.status==='RUNNING'?'Request running; capture is saved on completion.':item.client_requests===null?'Original client messages were not recorded.':'No supported client dispatch was observed.'));
    else {
      try {
        const source=clientBodyText(attempt);
        if(source===null && attempt.record_http_bodies===false)container.append(element('p','muted','HTTP body recording is disabled. Attempt diagnostics remain available in Inspect request.'));
        else if(source===null)throw new Error(attempt.capture_error || 'Original body unavailable');
        else if(state.readerRaw)container.append(element('pre','native-literal native-raw',source));
        else {
          if(state.readerCache?.source!==source){state.readerCache={source,projection:capturedRequest(source)};}
          nativeMessages(container,state.readerCache.projection,selected.key,attempt.api);
          const rendered=attempt.api==='openai-responses'?['input','instructions']:attempt.api==='anthropic'?['messages','system']:['messages'];
          const envelope=state.readerCache.projection.fields.filter(field=>!rendered.includes(field.name));
          if(envelope.length){const detail=readerDetail(selected.key+':envelope','Request envelope fields','native-envelope');envelope.forEach(field=>detail.append(literalSource(field.raw)));container.append(detail);}
        }
      } catch(error){container.append(element('p','source-warning',`Readable view unavailable: ${error.message}. The original request remains authoritative.`));try {const source=clientBodyText(attempt);if(source!==null)container.append(element('pre','native-literal',source));}catch{}}
      const observed=attempt.diagnostics || {};
      if(observed.read_outcome)container.append(element('p','source-note',`HTTP read: ${observed.read_outcome}${observed.read_error_type?' ('+observed.read_error_type+')':''}.`));
      if(observed.close_started_ns!=null)container.append(element('p','source-note',`HTTP close: ${observed.close_error_type?'error ('+observed.close_error_type+')':observed.close_finished_ns!=null?'finished':'started'}.`));
      const sdkEvents=['entered','exhausted','cancel_requested','close_started','close_finished','entry_error','iteration_error','exit_error','close_error'].filter(event=>observed[`sdk_${event}_ns`]!=null).sort((a,b)=>observed[`sdk_${a}_ns`]-observed[`sdk_${b}_ns`]);
      if(sdkEvents.length)container.append(element('p','source-note','SDK stream observations: '+sdkEvents.map(event=>event.replaceAll('_',' ')+(observed[`sdk_${event}_type`]?' ('+observed[`sdk_${event}_type`]+')':'')).join(' · ')+'. These are lifecycle observations, not provider success.'));
      let completion='body EOF unknown';
      if(attempt.response?.format==='sse')completion=Object.hasOwn(observed,'protocol_terminal')?observed.protocol_terminal===null?'SSE terminal not seen':'SSE terminal: '+observed.protocol_terminal:attempt.response.complete?'SSE terminal observed':'SSE terminal unknown';
      else if(attempt.response?.complete || observed.read_outcome==='eof')completion='body EOF observed';
      else if(attempt.response?.body!=null && !attempt.response.capture_error)completion='body EOF not observed';
      const response=readerDetail(selected.key+':response',attempt.response?`Observed response · HTTP ${attempt.response.status_code} · ${completion}`:'No response observed','native-response');
      if(attempt.response)response.append(element('p','muted',`${attempt.response.format} · ${attempt.response.content_encoding || 'identity'} · Separate from this request’s input messages. ${attempt.response.body==null?'Response body not retained; terminal absence is not evidence of a broken stream.':'Only bytes pulled by the SDK were captured; protocol terminal and HTTP EOF are separate observations.'}`),nativeBody(attempt.response,`${selected.key}-response.bin`));container.append(response);
    }
    if(item.execution_id!==state.selectedExecution)container.append(button('Open subagent',async()=>selectExecution(item.execution_id)));
  }
  const refresh=button('Refresh requests',async()=>loadNativeTimeline(false,true));refresh.dataset.requestFocus='refresh';refresh.setAttribute('aria-busy',String(view.loading));container.append(refresh);
  if(view.cursor){const more=button('Load more requests',async()=>loadNativeTimeline(true));more.dataset.requestFocus='more';more.setAttribute('aria-busy',String(view.loading));container.append(more);}
}
function renderNativeViews() {
  if(state.view==='sessions' && state.conversationMode==='original')renderConversation();
  if(state.tab==='models' && !$('inspector').hidden && state.nativeTimeline?.id===state.selectedExecution){
    state.details=state.nativeTimeline.items;state.detailCursor=state.nativeTimeline.cursor;renderDetails();$('detail-more').hidden=!state.detailCursor;
  }
}
function mergeNativeItems(previous,incoming) {
  const executions=new Map(),merged=new Map(previous.map(item=>[modelKey(item),item]));
  incoming.forEach(item=>upsertModel(merged,item));
  for(const item of merged.values()) {
    if(!executions.has(item.execution_id))executions.set(item.execution_id,[]);
    executions.get(item.execution_id).push(item);
  }
  return [...executions.values()].flatMap(items=>items.sort((a,b)=>a.agent_run_seq-b.agent_run_seq || a.model_request_seq-b.model_request_seq));
}
async function loadNativeTimeline(more=false,refresh=false) {
  const id=state.selectedExecution,generation=state.generation;
  if(!id || state.view==='metrics')return;
  let view=state.nativeTimeline;
  if(view?.id===id && view.http===state.readerHttp){
    if(view.loading)return;
    if(!more && !refresh){if(view.pending.size)scheduleNativeRefresh();return;}
  } else {
    if(nativeRefresh){clearTimeout(nativeRefresh.timer);nativeRefresh=null;}
    view={id,http:state.readerHttp,items:[],cursor:null,loading:false,error:null,pending:new Map()};
  }
  if(more && !view.cursor)return;
  if(refresh)view.pending.forEach(request=>{request.error=null;});
  state.nativeTimeline=view;view.loading=true;view.error=null;renderNativeViews();
  const current=()=>state.nativeTimeline===view && generation===state.generation && id===state.selectedExecution;
  const incoming=[],target=refresh?Math.max(10,view.items.length):10;
  let cursor=more?view.cursor:null;
  try {
    do {
      const params=new URLSearchParams({limit:'10',...(view.http?{include_client_requests:'true'}:{include_content:'true'})});
      if(cursor)params.set('cursor',cursor);
      const page=await api(`/api/executions/${enc(id)}/models?${params}`);
      if(!current())return;
      incoming.push(...page.items);cursor=page.next_cursor || null;
    } while(refresh && cursor && incoming.length<target);
    view.items=mergeNativeItems(view.items,incoming);view.cursor=cursor;incoming.forEach(acceptLiveResponse);
    incoming.filter(item=>item.status!=='RUNNING').forEach(item=>view.pending.delete(modelKey(item)));
  } catch(error) {if(current())view.error=error.message;}
  finally {if(current()){view.loading=false;view.error ||= [...view.pending.values()].map(request=>request.error).filter(Boolean).join('\n');renderNativeViews();if(view.pending.size)scheduleNativeRefresh();}}
}
let nativeRefresh=null;
function scheduleNativeRefresh(item) {
  const view=state.nativeTimeline;
  if(!view || view.id!==state.selectedExecution)return;
  if(item){
    const key=modelKey(item);
    if(view.items.some(value=>modelKey(value)===key && value.status!=='RUNNING' && (view.http?value.client_requests_included:value.content_included)))return;
    view.pending.set(key,{item,error:null});
  }
  if(![...view.pending.values()].some(request=>!request.error) || view.loading || !(state.view==='sessions' && state.conversationMode==='original' || state.tab==='models' && !$('inspector').hidden))return;
  const id=state.selectedExecution,generation=state.generation;
  if(nativeRefresh?.id===id && nativeRefresh.generation===generation)return;
  if(nativeRefresh)clearTimeout(nativeRefresh.timer);
  const refresh={id,generation,timer:null};nativeRefresh=refresh;
  refresh.timer=setTimeout(()=>{
    if(nativeRefresh!==refresh)return;nativeRefresh=null;
    if(state.nativeTimeline===view && id===state.selectedExecution && generation===state.generation && (state.view==='sessions' && state.conversationMode==='original' || state.tab==='models' && !$('inspector').hidden))refreshNativeRequests(view,generation).catch(showError);
  },60);
}
async function refreshNativeRequests(view,generation) {
  if(view.loading)return;
  const current=()=>state.nativeTimeline===view && generation===state.generation && view.id===state.selectedExecution;
  view.loading=true;renderNativeViews();
  try {
    for(const [key,request] of [...view.pending]) {
      if(request.error)continue;
      if(!(state.view==='sessions' && state.conversationMode==='original' || state.tab==='models' && !$('inspector').hidden))break;
      const item=request.item;
      try {
        const params=new URLSearchParams({limit:'1',...(view.http?{include_client_requests:'true'}:{include_content:'true'}),agent_run_seq:String(item.agent_run_seq),model_request_seq:String(item.model_request_seq)});
        const page=await api(`/api/executions/${enc(item.execution_id)}/models?${params}`);
        if(!current())return;
        if(!page.items.some(value=>modelKey(value)===key))throw new Error('Model request record is no longer available');
        view.items=mergeNativeItems(view.items,page.items);page.items.forEach(acceptLiveResponse);
        if(view.pending.get(key)===request)view.pending.delete(key);
      } catch(error){
        if(!current())return;
        request.error=`${item.execution_id} · request ${item.model_request_seq}: ${error.message}`;
      }
    }
  } finally {
    if(current()){
      view.loading=false;view.error=[...view.pending.values()].map(request=>request.error).filter(Boolean).join('\n') || view.error;
      renderNativeViews();scheduleNativeRefresh();
    }
  }
}
async function loadDetail(more=false, selectors={}, reuseNative=false) {
  if (!state.selectedExecution || $('inspector').hidden || state.view==='metrics') return;
  const generation=++state.detailGeneration, selection=state.generation, id=state.selectedExecution, tab=state.tab;
  if (!more) { state.details=[]; state.detailCursor=null; $('inspector-content').classList.remove('native-reader');clear('inspector-content'); }
  $('detail-more').hidden=true;
  $('detail-view').value=tab;setDisabled();
  if (tab==='overview') { renderOverview(); return; }
  if (tab==='models') {
    await loadNativeTimeline(more,!more && !reuseNative && !(state.view==='sessions' && state.conversationMode==='original'));
    if(generation!==state.detailGeneration || selection!==state.generation || id!==state.selectedExecution || tab!==state.tab)return;
    renderNativeViews();
    if(state.nativeTimeline?.error)showError(new Error(state.nativeTimeline.error));
    return;
  }
  if (tab==='recovery' && state.config.read_only) { $('inspector-content').append(element('p','empty','Recovery controls require an execution Runtime.')); return; }
  const params=new URLSearchParams({limit:'50',...selectors});
  if (['history','transcript','sdkmodels'].includes(tab)) params.set('include_content','true');
  if (more && state.detailCursor) params.set('cursor',state.detailCursor);
  try {
    const payload=await api(`/api/executions/${enc(id)}/${tab==='sdkmodels'?'models':tab}?${params}`);
    if (generation!==state.detailGeneration || selection!==state.generation || id!==state.selectedExecution || tab!==state.tab) return;
    const items=Array.isArray(payload)?payload:payload.items;
    state.details.push(...items); state.detailCursor=payload.next_cursor || null; state.detailSelectors=selectors;
    renderDetails(); $('detail-more').hidden=!state.detailCursor;
  } catch(error) { if(generation===state.detailGeneration && selection===state.generation && id===state.selectedExecution && tab===state.tab) showError(error); }
}
function renderOverview() {
  $('inspector-content').classList.remove('native-reader');clear('inspector-content'); const info=state.execution; if (!info) return;
  $('inspector-content').append(properties({Agent:info.agent_id,Kind:info.binding_kind,Session:info.session_id,Lineage:info.lineage_kind,Created:date(info.created_at),Started:date(info.started_at),Finished:date(info.terminal_at),Error:info.error_code}));
  if(info.usage) { const card=element('div','detail-card'); card.append(element('h3','','Usage'),properties({Requests:info.usage.logical_requests,'Input tokens':info.usage.input_tokens,'Output tokens':info.usage.output_tokens,'Unknown usage':info.usage.unknown_usage_requests}),rawDetail(info.usage)); $('inspector-content').append(card); }
  $('inspector-content').append(rawDetail(info,'Execution metadata'));
  if(state.models.size && state.selectedExecution===state.observedExecution) {
    $('inspector-content').append(element('h3','','Live model requests'));
    state.models.forEach(item=>{const card=element('div','detail-card');card.append(element('h3','',`Request ${item.model_request_seq} · ${item.status}`),element('p','',`${item.execution_id} · run ${item.agent_run_seq} · ${item.purpose}`),rawDetail(item));$('inspector-content').append(card);});
  }
}
function traceItem(item) {
  const row=item.payload || {},key=`${item.execution_id}:${item.step_event_seq}`,selected=state.traceSelection===key;
  const node=element('section','trace-item'),toggle=button('',async()=>{state.traceSelection=selected?null:key;renderDetails();});
  toggle.className='trace-row';toggle.dataset.requestFocus=`trace:${key}`;toggle.setAttribute('aria-expanded',String(selected));
  toggle.append(element('span','trace-seq',`#${item.step_event_seq}`),element('span','trace-label',`${(row.kind || 'Step').replaceAll('_',' ').toLowerCase()}${row.tool_name?' · '+row.tool_name:''}`));
  if(row.status)toggle.append(element('span','trace-status',row.status));
  toggle.append(element('span','trace-duration',duration(row.duration_ns)));node.append(toggle);
  if(!selected)return node;
  const detail=element('div','trace-detail');
  detail.append(element('p','mono',`${item.execution_id} · ${row.scope || '—'} · run ${row.agent_run_seq ?? '—'} · step ${row.step_index ?? '—'} · request #${row.model_request_seq ?? '—'}`),element('p','muted',`${row.purpose || '—'} · ${duration(row.duration_ns)}${row.token_usage?' · '+usageLabel(row.token_usage):''}`));
  const selectors={};['agent_run_seq','model_request_seq','step_index','tool_call_id'].forEach(key=>{if(row[key]!=null)selectors[key]=row[key];});
  if(row.call_id)selectors.tool_call_id=row.call_id;
  const content=button('Read SDK content',async()=>{
    if(item.execution_id!==state.selectedExecution){
      const generation=state.generation,selection=selectExecution(item.execution_id),executionGeneration=state.executionGeneration;
      await selection;
      if(generation!==state.generation || executionGeneration!==state.executionGeneration || item.execution_id!==state.selectedExecution || state.tab!=='trace')return;
    }
    state.tab='history';await loadDetail(false,selectors);
  });content.dataset.requestFocus=`trace:${key}:content`;detail.append(content);
  if(row.child_execution_id){const child=button('Open subagent',async()=>state.view==='sessions'?selectExecution(row.child_execution_id):navigate(null,row.child_execution_id));child.dataset.requestFocus=`trace:${key}:child`;detail.append(child);}
  const raw=rawDetail(item,'Original event record');raw.children[0].dataset.requestFocus=`trace:${key}:raw`;detail.append(raw);node.append(detail);return node;
}
function renderDetails() {
  $('inspector-content').classList.remove('native-reader');
  const requestFocus=requestFocusWithin($('inspector-content'));
  clear('inspector-content');
  if(['history','transcript','sdkmodels'].includes(state.tab)) {
    const note=element('div','source-note');
    const explanation=state.tab==='history' ? 'SDK structured records, not outbound messages. Instructions is a separate SDK request field, not an earlier system or developer message. Display order is not client message order.' : state.tab==='transcript' ? 'SDK text excerpts only: user text and assistant text. System, instructions, tools and other structured parts are omitted; this is not a complete model request.' : 'SDK preparation snapshot, not the final client request. Categories describe framework input, not outbound message order.';
    note.append(element('p','',state.tab==='transcript'?'SDK text excerpts, not a complete request.':'SDK records, not outbound messages.'));
    const about=element('details');about.append(element('summary','','About this SDK view'),element('p','',explanation));note.append(about);
    note.append(button('View model messages',async()=>{state.tab='models';await loadDetail();}));$('inspector-content').append(note);
  }
  if(state.tab==='recovery' && !state.config.read_only) {
    const resume=button('Resume stopped execution',()=>executionAction('recover'));resume.id='resume-run';
    $('inspector-content').append(element('p','empty','Resume continues the original work and may call models or tools. To end it instead, use End stopped execution.'),resume);setDisabled();
  }
  if(state.tab==='models'){renderNativeTimeline($('inspector-content'));restoreRequestFocus($('inspector-content'),requestFocus);return;}
  if(!state.details.length) $('inspector-content').append(element('p','empty',terminal(state.execution?.status)?'No records for this selection.':'No committed detail yet; live progress may arrive first.'));
  state.details.forEach(item=>{
    const card=element('div','detail-card');
    if(state.tab==='sdkmodels') {
      card.append(element('h3','',`SDK request snapshot ${item.model_request_seq} · ${item.status}`),element('p','',`${short(item.execution_id)} · ${modelLabel(item.model)} · ${duration(item.duration_ns)}`));
      if(item.usage)card.append(element('p','',usageLabel(item.usage)));
      sdkMessages(card,item,modelKey(item));
      if(Array.isArray(item.request?.messages)){const layers=element('details');layers.append(element('summary','','SDK prompt architecture'),properties(Object.fromEntries(promptLayers(item.request))));card.append(layers);}
      card.append(rawDetail(item,'SDK prompt, response projection & metadata'));
    } else if(state.tab==='trace') {
      $('inspector-content').append(traceItem(item));return;
    } else if(state.tab==='recovery') {
      card.append(element('h3','',item.tool_name),element('p','',`Effect outcome unknown · ${item.tool_call_id}`),rawDetail(item));
      ['not_applied','applied','failed'].forEach(resolution=>card.append(button(resolution.replaceAll('_',' '),async()=>{
        const id=state.selectedExecution,generation=state.generation,executionGeneration=state.executionGeneration,detailGeneration=state.detailGeneration,noticeRevision=state.noticeRevision;
        const current=()=>generation===state.generation && executionGeneration===state.executionGeneration && detailGeneration===state.detailGeneration && id===state.selectedExecution && state.tab==='recovery';
        if(!confirm(`Confirm the external effect was ${resolution.replaceAll('_',' ')}? This controls recovery.`))return;
        let result=null; if(resolution==='applied'){const raw=prompt('Recorded tool result (JSON)','null');if(raw===null)return;result=JSON.parse(raw);}
        try {await mutate(`/api/executions/${enc(id)}/resolve`,{operation_id:item.operation_id,expected_fence:item.fence,resolution,result});if(current())await loadDetail();}
        catch(error){if(current() && noticeRevision===state.noticeRevision)throw error;}
      })));
    } else {
      const instructions=state.tab==='history' && item.item_kind==='instructions';
      card.classList.toggle('sdk-field',instructions);
      const title=instructions?'SDK instructions field':state.tab==='transcript'?'SDK text excerpt':`SDK ${item.item_kind || 'unknown'} part${item.tool_name?' · '+item.tool_name:''}`;
      const locator=instructions?`Request record ${item.message_seq} · field metadata`:`Record ${item.message_seq}${item.part_index!=null?' · part '+item.part_index:''}`;
      card.append(element('h3','',title),element('p','',`${locator} · run ${item.agent_run_seq ?? '—'}${item.execution_id?' · '+short(item.execution_id):''}`),element('pre','json',text(item.content ?? item.text)),rawDetail(item));
    }
    if(state.tab==='history' && item.execution_id!==state.selectedExecution)card.append(button('Open subagent',async()=>selectExecution(item.execution_id)));
    $('inspector-content').append(card);
  });
  restoreRequestFocus($('inspector-content'),requestFocus);
}
async function refreshSelected() {
  const refreshGeneration=++state.refreshGeneration, generation=state.generation, executionGeneration=state.executionGeneration, sessionId=state.view==='sessions'?state.selectedSession:null, id=state.selectedExecution, noticeRevision=state.noticeRevision, noticeId=state.cancellationNoticeId;
  try {
  if(sessionId) {
    const payload=await readSession(sessionId);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    if(acceptSession(payload)) {
      state.timelineError=payload.timeline_error || '';renderSessionHeader();
      if(payload.timeline){state.turns=mergePage(state.turns,payload.timeline.items,item=>item.execution_id);if(!state.hasEarlierTurns)state.turnCursor=payload.timeline.next_cursor;}
      else{state.turnCursor=null;state.hasEarlierTurns=false;}
      const selected=state.turns.find(item=>item.execution_id===state.observedExecution);
      if(payload.timeline && selected?.conversation_committed){state.liveText='';state.liveThinking='';state.liveAttempts.clear();state.liveTools.clear();state.events.clear();}
      renderConversation();setDisabled();
    }
  }
  if(id) {
    const readGeneration=id===state.observedExecution?++state.observationReadGeneration:state.observationReadGeneration;
    let info=await api(`/api/executions/${enc(id)}`);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    info=observedInfo(info,readGeneration);state.execution=info;if(state.recoveryReadbackId===id)state.recoveryReadbackId=null;renderExecution();
    const nativeVisible=state.view==='sessions' && state.conversationMode==='original' || state.tab==='models' && !$('inspector').hidden;
    if(nativeVisible)await loadNativeTimeline(false,true);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    if(state.tab==='models' && nativeVisible)renderNativeViews();else await loadDetail();
    setDisabled();
  }
  if(noticeId) {
    const info=noticeId===id?state.execution:await api(`/api/executions/${enc(noticeId)}`);
    if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
    if(terminal(info.status)) {
      const payload=info.session_id ? await readSession(info.session_id,true) : null,session=payload?.session;
      if(refreshGeneration!==state.refreshGeneration || generation!==state.generation || executionGeneration!==state.executionGeneration || id!==state.selectedExecution)return;
      if(payload && sessionId===info.session_id && acceptSession(payload))renderSessionHeader();
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
    state.cursor=null;state.liveText='';state.liveThinking='';state.liveAttempts.clear();state.liveTools.clear();state.liveActivityOpen=false;state.models.clear();state.events.clear();renderLive();
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
  if(payload && sessionId===state.selectedSession && acceptSession(payload)) {
    state.timelineError=payload.timeline_error || '';renderSessionHeader();
    if(payload.timeline){state.turns=mergePage(state.turns,payload.timeline.items,item=>item.execution_id);if(!state.hasEarlierTurns)state.turnCursor=payload.timeline.next_cursor;}
    const turn=state.turns.find(item=>item.execution_id===id);
    if(payload.timeline && turn?.conversation_committed){state.liveText='';state.liveThinking='';state.liveAttempts.clear();state.liveTools.clear();state.events.clear();}
    renderConversation();renderList();
  }
  if(state.cancellationNoticeId===id && terminal(info.status)) {
    const session=info.session_id ? (payload && info.session_id===sessionId ? payload.session : (await api(sessionURL(info.session_id,'',{include_timeline:'false'}))).session) : null;
    if(!current())return;
    reconcileCancellation(info,session,noticeRevision);
  }
  if(id===state.selectedExecution && terminal(info.status) && (state.view==='sessions' && state.conversationMode==='original' || state.tab==='models' && !$('inspector').hidden))await loadNativeTimeline(false,true);
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
          renderLive();
          if(id===state.selectedExecution){state.execution=info;renderExecution();}
          setDisabled();return;
        }
        if(frame.event==='observation_error') {const error=new Error(`${frame.data.error_code}: observation interrupted; checking Runtime state`);error.reconnect=frame.data.origin==='stream' && !frame.data.safe_details?.cleanup_pending;throw error;}
        const envelope=frame.data,item=envelope.item;
        if(envelope.cursor)state.cursor=envelope.cursor;
        if(envelope.type==='model') {
          const previous=state.models.get(modelKey(item.item));upsertModel(state.models,item.item);
          const request=state.liveAttempts.get(modelKey(item.item));
          if(!acceptLiveResponse(item.item) && request && item.visibility==='durable_history' && item.item.status==='SUCCEEDED'){
            request.status='SUCCEEDED';request.finished=true;readLiveResponse(request);
          }
          if(item.item.status!=='RUNNING' && (!previous || previous.status==='RUNNING') && (id===state.selectedExecution || item.item.execution_id===state.selectedExecution))scheduleNativeRefresh(item.item);
          if(state.tab==='overview' && id===state.selectedExecution)renderOverview();
        }
        else {
          const event=item.event,payload=event.payload || {};
          updateLiveTools(item);
          const attemptHandled=item.depth===0 && updateStreamAttempts(state.liveAttempts,item);
          if(item.depth===0 && event.event_type==='STREAM_RESET'){state.liveText='';state.liveThinking='';}
          else if(!attemptHandled && item.depth===0 && event.event_type==='ASSISTANT_TEXT_DELTA')state.liveText+=payload.text || '';
          else if(!attemptHandled && item.depth===0 && event.event_type==='ASSISTANT_THINKING_DELTA')state.liveThinking+=payload.text || '';
          if(!event.event_type.endsWith('_DELTA')) {
            state.events.set(eventKey(item),item);
            if(state.events.size>100)state.events.delete(state.events.keys().next().value);
          }
          scheduleLiveRender();
        }
      },controller.signal);
      if(!current())return;
      if(ended) {
        // A competing readback is not a broken stream; retain incomplete live content.
        let refreshed=false;
        while(current() && !refreshed)refreshed=await refreshObservation(id,generation);
        if(!current())return;
        if(state.view==='sessions' && state.session?.active_execution_id && state.session.active_execution_id!==id){await syncObservation();if(!current())return;}
        const status=state.observedInfo?.status;
        if(terminal(status)){connection('Complete');return;}
        if(status==='RECOVERY_REQUIRED'){connection('Recovery required',true);return;}
      }
      throw new Error('Observation connection closed');
    } catch(error) {
      if(!current())return;
      connection('Reconnecting',true);
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
  } finally {if(state.stream===controller){state.stream=null;if(current())renderLive();}}
}
async function sendMessage(event) {
  event.preventDefault(); const sessionId=state.selectedSession,generation=state.generation,promptValue=$('prompt').value;
  if(!sessionId || !promptValue.trim())return;
  $('send').disabled=true;notice('');
  try {
    // Rejected admission is terminal for its key; another explicit send is a new attempt.
    const result=await mutate(sessionURL(sessionId,"messages"),{prompt:promptValue,planning:$('planning').checked,thinking:readThinking()}, {newAttemptOn:['SESSION_BUSY','SESSION_CONFLICT']});
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
  $('new-dialog').dataset.sessionId=crypto.randomUUID();$('create-session').disabled=false;$('new-dialog').showModal();$('new-agent').focus();
}
async function createSession(event) {
  event.preventDefault();const button=$('create-session');button.disabled=true;
  const sessionId=$('new-dialog').dataset.sessionId,generation=state.generation,noticeRevision=state.noticeRevision;
  try {
    const session=await mutate('/api/sessions',{session_id:sessionId,title:$('new-title').value,agent_id:$('new-agent').value,cwd:$('new-cwd').value || null});
    if($('new-dialog').open && $('new-dialog').dataset.sessionId===sessionId && generation===state.generation){$('new-dialog').close();state.conversationMode='chat';state.focusComposerSession=session.session_id;setView('sessions',false);navigate(session.session_id,null);}
    await loadList();
  } catch(error){if($('new-dialog').open && $('new-dialog').dataset.sessionId===sessionId && generation===state.generation && noticeRevision===state.noticeRevision)showError(error);}
  finally {if($('new-dialog').dataset.sessionId===sessionId)button.disabled=false;}
}
async function sessionAction(action) {
  const id=state.selectedSession,generation=state.generation,noticeRevision=state.noticeRevision;if(!id || !state.session)return;
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
    const result=await mutate(sessionURL(id,action),payload).catch(error=>{
      if(generation===state.generation && id===state.selectedSession && noticeRevision===state.noticeRevision)throw error;
      return null;
    });
    if(!result)return;
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
    else if(action==='recover') {
      await refreshSelected();
      if(current())await syncObservation({refresh:true});
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
    const payload=info.session_id ? await readSession(info.session_id,true) : null,session=payload?.session;
    if(!selected())return;
    if(session?.active_execution_id===id)throw new Error('Execution is terminal, but its session release is not confirmed.');
    state.endReadbackId=null;if(id===state.selectedExecution){state.execution=info;renderExecution();}if(id===state.observedExecution)state.observedInfo=info;
    if(payload && state.selectedSession===info.session_id && acceptSession(payload))renderSessionHeader();
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
  event?.preventDefault();const generation=++state.metricsGeneration,selection=state.generation,noticeRevision=state.noticeRevision;
  const current=()=>generation===state.metricsGeneration && selection===state.generation && state.view==='metrics';
  const params=new URLSearchParams();
  [['metric-name','metric'],['metric-groups','group_by'],['metric-bucket','bucket_seconds'],['metric-filters','filters'],['metric-correlations','correlation_filters'],['metric-aggregation','aggregation'],['metric-percentile','percentile']].forEach(([id,key])=>{if($(id).value)params.set(key,$(id).value);});
  if($('metric-start').value)params.set('start',new Date($('metric-start').value).toISOString());
  if($('metric-end').value)params.set('end',new Date($('metric-end').value).toISOString());
  const payload=await api('/api/metrics?'+params).catch(error=>{
    if(current() && noticeRevision===state.noticeRevision)throw error;
    return null;
  });
  if(!payload || !current())return;
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
  $('settings-content').append(properties({
    Workspace:config.workspace,'Asset root':config.asset_root,Namespace:config.runtime?.namespace,
    Tenant:config.runtime?.tenant_id,Model:config.model || 'Not configured',
    'API key':config.api_key_configured?'Configured':'Not configured',
    'Base URL':config.base_url_configured?'Configured':'Default',
    Mode:config.read_only?'Read-only':'Local execution',
  }));
  const inputs=element('div','detail-card');inputs.append(element('h3','','Captured input MIME types'),element('p','','Category names are expanded when the model is captured. Only matching attachment MIME types are allowed. Plain text is unaffected; provider support has not been verified.'),element('pre','json',config.input_mime_types?json(config.input_mime_types):'No model configured'));$('settings-content').append(inputs);
  if(config.runtime) {
    const status=element('div','detail-card');
    status.append(element('h3','','Runtime status'),element('p','','Current storage composition and metric recorder state, not a connectivity or storage health check.'),rawDetail(config.runtime,'Inspect Runtime status'));
    $('settings-content').append(status);
  }
  const card=element('div','detail-card');card.append(element('h3','','Captured capabilities'));config.capabilities.forEach(item=>card.append(element('p','',`${item.kind} · ${item.id} · revision ${item.revision}`)));if(!config.capabilities.length)card.append(element('p','','Capability composition is loaded when an execution Runtime starts.'));$('settings-content').append(card);$('settings-dialog').showModal();
}
async function bootstrap() {
  state.config=await api('/api/config');$('workspace-label').textContent=state.config.workspace || 'Local Runtime';
  state.config.capabilities.filter(item=>item.kind==='agent').forEach(item=>{const option=element('option','',item.id);option.value=item.id;$('new-agent').append(option);});
  state.config.metric_names.forEach(name=>{const option=element('option','',name.replace('linktools.',''));option.value=name;$('metric-names').append(option);});
  if(state.config.read_only){connection('Read-only');notice('Read-only. Configure a model at startup to send messages.');}
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
$('menu').onclick=()=>{const open=$('menu').getAttribute('aria-expanded')!=='true';$('sidebar').classList.toggle('open',open);$('menu').setAttribute('aria-expanded',String(open));};
$('sidebar-close').onclick=()=>closeSidebar(true);
window.addEventListener('keydown',event=>{if(event.key==='Escape' && $('menu').getAttribute('aria-expanded')==='true' && !$('new-dialog').open && !$('settings-dialog').open)closeSidebar(true);});
$('new-title').oninvalid=()=>$('new-options').open=true;
$('detail-turn').onchange=()=>selectExecution($('detail-turn').value).catch(showError);
async function openRecovery(active=false) {
  const id=active?state.session?.active_execution_id:state.selectedExecution;if(!id)return;
  showInspector(true,true);state.tab='recovery';await selectExecution(id);
}
$('open-recovery').onclick=()=>{ $('action-menu').open=false;return openRecovery(Boolean(state.session?.active_execution_id)).catch(showError); };
$('recovery-shortcut').onclick=()=>openRecovery(true).catch(showError);
$('thinking').oninput=renderThinking;
$('inspector-toggle').onclick=()=>{$('action-menu').open=false;const open=$('inspector').hidden,reuseNative=state.conversationMode==='original';showInspector(open,true);if(open)return loadDetail(false,{},reuseNative).catch(showError);};
$('inspector-close').onclick=()=>showInspector(false,true);
$('show-active').onclick=()=>{const id=state.session?.active_execution_id;if(id){showInspector(true,true);selectExecution(id).catch(showError);}};
$('action-menu').onkeydown=event=>{if(event.key==='Escape'){$('action-menu').open=false;$('actions-toggle').focus();}};
function closeRunActions(){if($('run-actions').open){$('run-actions').open=false;$('run-actions-toggle').focus();}}
$('run-actions').onkeydown=event=>{if(event.key==='Escape')closeRunActions();};
$('run-recovery').onclick=()=>{closeRunActions();return openRecovery().catch(showError);};
$('detail-recovery').onclick=()=>openRecovery().catch(showError);
$('open-form').onsubmit=event=>{Promise.resolve(openExact(event)).catch(showError);};
$('filter').oninput=renderList;$('list-action').onchange=setListFilterAvailability;
$('filter-form').onsubmit=event=>{event.preventDefault();(state.view==='executions' && $('list-action').value==='recent'?loadRecentExecutions():loadList()).catch(showError);};
$('native-mode').onclick=()=>{state.readerExpanded.delete('reader:tools');closeRunActions();$('action-menu').open=false;state.conversationMode='original';showInspector(false);renderPage();renderConversation();$('conversation').scrollTop=0;$('native-mode').focus();return loadNativeTimeline().catch(showError);};
$('chat-mode').onclick=()=>{state.readerExpanded.delete('reader:tools');$('action-menu').open=false;state.conversationMode='chat';showInspector(false);renderPage();renderConversation();($('prompt').disabled?$('chat-mode'):$('prompt')).focus();};
$('activity-mode').onclick=()=>{state.readerExpanded.delete('reader:tools');closeRunActions();$('action-menu').open=false;showInspector(true);state.conversationMode='activity';state.tab='trace';renderPage();return loadDetail().catch(showError);};
$('list-more').onclick=()=>loadList(true).catch(showError);$('turns-more').onclick=()=>moreTurns().catch(showError);
$('detail-more').onclick=()=>loadDetail(true,state.detailSelectors).catch(showError);
$('refresh').onclick=()=>{if(!state.cancellationNoticeId)notice('');(async()=>{if(state.view==='metrics'){await loadMetrics();return;}const generation=state.generation,executionGeneration=state.executionGeneration;await refreshSelected();if(generation===state.generation && executionGeneration===state.executionGeneration)await syncObservation({refresh:true});})().catch(showError);};
let settingsRequest=null;
$('settings').onclick=()=>{
  if(!settingsRequest)settingsRequest=api('/api/config').then(config=>{state.config=config;showSettings();}).catch(showError).finally(()=>{settingsRequest=null;});
  return settingsRequest;
};
$('export').onclick=()=>{closeRunActions();return exportResult().catch(showError);};
$('rename-session').onclick=()=>sessionAction('update').catch(showError);$('fork-session').onclick=()=>sessionAction('fork').catch(showError);$('close-session').onclick=()=>sessionAction('close').catch(showError);
$('inspector-active-stop').onclick=()=>executionAction('cancel',{active:true}).catch(showError);
$('cancel').onclick=()=>executionAction('cancel',{active:true}).catch(showError);$('stop-run').onclick=()=>executionAction('cancel').catch(showError);$('retry').onclick=()=>{closeRunActions();return executionAction('retry').catch(showError);};$('fork-run').onclick=()=>{closeRunActions();return executionAction('fork').catch(showError);};$('end-stopped').onclick=()=>endStoppedExecution();$('end-active').onclick=()=>endStoppedExecution({active:true});
$('metrics-form').onsubmit=event=>loadMetrics(event).catch(showError);
document.querySelectorAll('[data-view]').forEach(node=>node.onclick=()=>navigateView(node.dataset.view));
$('detail-view').onchange=()=>{state.tab=$('detail-view').value;return loadDetail().catch(showError);};
document.querySelectorAll('[data-close]').forEach(node=>node.onclick=()=>$(node.dataset.close).close());
setDisabled();
bootstrap().catch(showError);
