export const terminal = status => ['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(status);
export const modelKey = item => `${item.execution_id}:${item.agent_run_seq}:${item.model_request_seq}`;
export const historyKey = item => `${item.execution_id}:${item.agent_run_seq}:${item.message_seq}:${item.part_index}:${item.item_kind}:${item.tool_call_id || ''}`;
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
  if (result.unit === 'ns') return value >= 1e9 ? `${(value/1e9).toFixed(2)} s` : `${(value/1e6).toFixed(1)} ms`;
  return new Intl.NumberFormat(undefined, {maximumFractionDigits:2}).format(value);
}
