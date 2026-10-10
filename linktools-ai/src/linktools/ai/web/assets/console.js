export const terminal = status => ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(status);
export const modelKey = item => `${item.execution_id}:${item.agent_run_seq}:${item.model_request_seq}`;
export const historyKey = item => `${item.execution_id}:${item.agent_run_seq}:${item.message_seq}:${item.part_index}:${item.item_kind}:${item.tool_call_id || ''}`;
export function eventKey(item) {
  const event=item.event, payload=event.payload || {};
  const key=[item.execution_id,event.event_type];
  if (['MODEL_REQUEST_STARTED','MODEL_REQUEST_FINISHED'].includes(event.event_type)) key.push(payload.agent_run_seq,payload.model_request_seq);
  else if (['TOOL_CALL_STARTED','TOOL_CALL_FINISHED'].includes(event.event_type)) key.push(payload.agent_run_seq,payload.call_id);
  else if (event.event_type==='ASSISTANT_PART_COMPLETED') key.push(payload.agent_run_seq,payload.message_seq,payload.part_index);
  else key.push(event.durable_seq);
  return JSON.stringify(key);
}
export function upsertModel(items, item) {
  const key = modelKey(item), previous = items.get(key);
  if (previous && previous.status !== 'RUNNING' && item.status === 'RUNNING') return;
  items.set(key, item);
}
export function mergePage(previous, incoming, key) {
  const items = new Map(previous.map(item => [key(item), item]));
  incoming.forEach(item => items.set(key(item), item));
  return [...items.values()];
}
export function parseSSE(frame) {
  let event = 'message', id = null, data = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trimStart();
    else if (line.startsWith('id:')) id = line.slice(3).trimStart();
    else if (line.startsWith('data:')) data.push(line.slice(5).trimStart());
  }
  return data.length ? {event, id, data: JSON.parse(data.join('\n'))} : null;
}
export async function readSSE(response, onEvent, signal) {
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let pending = '';
  try {
    while (!signal.aborted) {
      const {done, value} = await reader.read();
      if (done) break;
      pending = (pending + decoder.decode(value, {stream:true})).replace(/\r\n/g, '\n');
      let boundary;
      while ((boundary = pending.indexOf('\n\n')) >= 0) {
        const item = parseSSE(pending.slice(0, boundary));
        pending = pending.slice(boundary + 2);
        if (item) await onEvent(item);
      }
    }
  } finally { await reader.cancel(); reader.releaseLock(); }
}
export function metricValue(result, point) {
  const value = point.value;
  if (value == null) return '—';
  if (result.metric.endsWith('_ratio')) return `${(value * 100).toFixed(1)}%`;
  if (result.unit === 'ns') return duration(value);
  return new Intl.NumberFormat(undefined, {maximumFractionDigits:2}).format(value);
}

export function duration(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  for (const [scale, unit] of [[1e9,'s'],[1e6,'ms'],[1e3,'us']]) {
    if (Math.abs(value) >= scale) return `${(value/scale).toFixed(3)} ${unit}`;
  }
  return `${value.toFixed(0)} ns`;
}
export function modelLabel(model={}) {
  return ['model_name','route_id','name','model','model_identity','provider'].map(key=>model[key]).find(value=>typeof value==='string' && value) || '—';
}
export function usageLabel(usage) {
  if (!usage) return '—';
  const number=value=>new Intl.NumberFormat().format(value || 0);
  let label=`${number(usage.input_tokens)} in / ${number(usage.output_tokens)} out`;
  if (usage.cache_read_tokens || usage.cache_write_tokens) label+=` · ${number(usage.cache_read_tokens)} cache read / ${number(usage.cache_write_tokens)} cache write`;
  return label;
}

const sequence=value=>value==null?[]:Array.isArray(value)?value:[value];
const records=value=>sequence(value).filter(item=>item && typeof item==='object' && !Array.isArray(item));
const parts=message=>Array.isArray(message.parts)?records(message.parts):[];
const partKind=part=>part.part_kind || part.kind || part.type || 'other';
function contentChars(value) {
  if (typeof value==='string') return [...value].length;
  if (Array.isArray(value)) return value.reduce((sum,item)=>sum+contentChars(item),0);
  if (value && typeof value==='object') return contentChars(Object.values(value));
  return 0;
}
function instructionSummary(values) {
  const names=[...new Set(values.map(item=>item.name).filter(name=>typeof name==='string' && name))];
  const sources=names.length?` · sources: ${names.slice(0,6).join(', ')}${names.length>6?', …':''}`:'';
  return `${values.length} part(s) · ~${contentChars(values.map(item=>item.content)).toLocaleString()} chars${sources}`;
}
function collectAttachments(value, mediaTypes) {
  if (!value || typeof value!=='object') return;
  if (!Array.isArray(value) && typeof value.media_type==='string' && value.media_type && Number.isInteger(value.size) && value.size>=0 && /^[a-f0-9]{64}$/.test(value.digest || '')) {
    mediaTypes.set(value.media_type,(mediaTypes.get(value.media_type) || 0)+1);return;
  }
  Object.values(value).forEach(item=>collectAttachments(item,mediaTypes));
}
export function promptLayers(request={}) {
  const parameters=request.parameters || {}, messages=records(request.messages);
  const instructions=records(parameters.instruction_parts), allParts=messages.flatMap(parts);
  const system=allParts.filter(part=>partKind(part)==='system-prompt');
  const conversation=messages.map(message=>parts(message).filter(part=>partKind(part)!=='system-prompt')).filter(items=>items.length);
  const kinds=new Map(), attachments=new Map();
  conversation.flat().forEach(part=>kinds.set(partKind(part),(kinds.get(partKind(part)) || 0)+1));
  allParts.filter(part=>partKind(part)==='user-prompt').forEach(part=>collectAttachments(part.content,attachments));
  const count=[...attachments.values()].reduce((sum,value)=>sum+value,0);
  const attachmentTypes=[...attachments].sort(([a],[b])=>a.localeCompare(b)).map(([type,amount])=>amount===1?type:`${type} ×${amount}`).join(', ');
  const kindCounts=[...kinds].sort(([a],[b])=>a.localeCompare(b)).map(([kind,amount])=>`${kind}=${amount}`).join(', ');
  const yesNo=value=>value===true?'yes':value===false?'no':'—';
  return [
    ['System Prompt',`${system.length} part(s) · ~${contentChars(system.map(part=>part.content)).toLocaleString()} chars`],
    ['Fixed Instructions (F0/F1)',instructionSummary(instructions.filter(item=>!item.dynamic))],
    ['Dynamic Instructions (O)',instructionSummary(instructions.filter(item=>item.dynamic))],
    ['Conversation Context',`${conversation.length} message(s) · ${conversation.flat().length} part(s)${kindCounts?' · '+kindCounts:''}`],
    ['Input Attachments',`${count} ${count===1?'attachment':'attachments'}${attachmentTypes?' · '+attachmentTypes:''}`],
    ['Tool Contract',`${sequence(parameters.function_tools).length} function · ${sequence(parameters.native_tools).length} native · ${sequence(parameters.revealed_tool_names).length} revealed · ${sequence(parameters.deferred_capability_ids).length} deferred capabilities`],
    ['Model Output Contract',`mode=${parameters.output_mode || '—'} · text output=${yesNo(parameters.allow_text_output)} · image output=${yesNo(parameters.allow_image_output)}`],
  ];
}
