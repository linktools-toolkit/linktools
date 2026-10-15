import MarkdownIt from './markdown-it.js';

const markdown = new MarkdownIt({html:false});
const validateLink = markdown.validateLink;
markdown.validateLink = url => validateLink(url) && !/^data:/i.test(url);
markdown.renderer.rules.link_open = (tokens, index, options, env, renderer) => {
  tokens[index].attrSet('target', '_blank');
  tokens[index].attrSet('rel', 'noopener noreferrer');
  return renderer.renderToken(tokens, index, options);
};
for(const kind of ['th_open','td_open'])markdown.renderer.rules[kind] = (tokens, index, options, env, renderer) => {
  const token=tokens[index], alignment={'text-align:left':'left','text-align:center':'center','text-align:right':'right'}[token.attrGet('style')];
  if(alignment){token.attrs=token.attrs.filter(([name])=>name!=='style');token.attrSet('class',`align-${alignment}`);}
  return renderer.renderToken(tokens, index, options);
};
// Messages must not fetch remote images merely because a user opens a conversation.
markdown.renderer.rules.image = (tokens, index) => {
  const token=tokens[index], escape=markdown.utils.escapeHtml;
  return `<a href="${escape(token.attrGet('src'))}" target="_blank" rel="noopener noreferrer">${escape(token.content || 'Image')}</a>`;
};

export function renderMarkdown(source) {
  return markdown.render(source);
}

export function inputPresentation(value) {
  if(typeof value==='string')return {text:value,attachments:[],structured:false};
  const parts=[], attachments=[];
  const attachment=item=>{
    const name=item.path || item.identifier || item.url || item.file_id || 'Attachment';
    const description=[name,item.media_type,Number.isFinite(item.size)?`${item.size} bytes`:null].filter(Boolean).join(' · ');
    attachments.push(description);
  };
  const read=item=>{
    if(!item || typeof item!=='object')return;
    if(item.kind==='text' && typeof item.text==='string')parts.push(item.text);
    else if(item.kind==='text-content' && typeof item.content==='string')parts.push(item.content);
    else if(item.kind==='items' && Array.isArray(item.items))item.items.forEach(read);
    else if(item.kind==='native' && item.codec==='user-content-v1' && Array.isArray(item.value?.items))item.value.items.forEach(read);
    else if(['workspace-file','binary','image-url','audio-url','document-url','video-url','uploaded-file'].includes(item.kind))attachment(item);
  };
  if(value?.version===1) {
    read(value.prompt);
    if(Array.isArray(value.files))value.files.forEach(item=>{if(item && typeof item==='object')attachment(item);});
    if(!attachments.length && Array.isArray(value.attachments))value.attachments.forEach(item=>{if(item && typeof item==='object')attachment(item);});
  }
  return {text:parts.join('\n\n'),attachments,structured:true};
}

// Offsets index decoded source text, never the captured bytes used for downloads.
// JSON.parse validates syntax; its numeric/object values are not display sources.
export function capturedRequest(source) {
  JSON.parse(source);
  const tokens=[...source.matchAll(/"(?:\\[\s\S]|[^"\\])*"|[{}\[\],:]|[^\s{}\[\],:]+/g)];
  const ends=new Uint32Array(tokens.length),stack=[];
  for(let i=0;i<tokens.length;i++) {
    const token=tokens[i][0];ends[i]=i+1;
    if(token==='{' || token==='[')stack.push(i);
    else if(token==='}' || token===']'){const start=stack.pop();ends[start]=i+1;}
  }
  const cache=new Map();
  const range=start=>{
    if(cache.has(start))return cache.get(start);
    const first=tokens[start],last=tokens[ends[start]-1],raw=source.slice(first.index,last.index+last[0].length);
    const kind=first[0]==='{'?'object':first[0]==='['?'array':first[0][0]==='"'?'string':first[0]==='null'?'null':'literal';
    let members,items;
    const node={kind,raw,text:kind==='string'?JSON.parse(first[0]):null,
      fields(){
        if(kind!=='object')return [];
        if(members)return members;
        members=[];
        for(let i=start+1;i<ends[start]-1;){
          const key=tokens[i],value=range(i+2),end=ends[i+2];
          members.push({name:JSON.parse(key[0]),raw:source.slice(key.index,tokens[end-1].index+tokens[end-1][0].length),value});
          i=end+(tokens[end]?.[0]===','?1:0);
        }
        return members;
      },
      items(){
        if(kind!=='array')return [];
        if(items)return items;
        items=[];
        for(let i=start+1;i<ends[start]-1;){items.push(range(i));i=ends[i]+(tokens[ends[i]]?.[0]===','?1:0);}
        return items;
      }
    };
    cache.set(start,node);return node;
  };
  const root=range(0),fields=root.fields(),messages=fields.filter(field=>field.name==='messages');
  return {source,root,fields,messages};
}
