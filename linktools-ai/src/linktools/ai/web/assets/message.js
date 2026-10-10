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
