import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const root=process.argv[2], url=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const vendor=url(readFileSync(resolve(root,'markdown-it.js'),'utf8'));
const source=readFileSync(resolve(root,'message.js'),'utf8').replace("'./markdown-it.js'",JSON.stringify(vendor));
const {renderMarkdown,inputPresentation}=await import(url(source));

const rendered=renderMarkdown('# 你好🌱\n\n- **first**\n- *second*\n\n| key | value |\n| --- | --- |\n| a | b |\n\n[guide](https://example.com/docs)\n\n```js\nconst answer = "<safe>";\n```');
for(const tag of ['h1','ul','li','strong','em','table','th','td','a','pre','code'])assert.match(rendered,new RegExp('<'+tag+'(?: |>)'));
assert.match(rendered,/rel="noopener noreferrer"/);
assert.match(rendered,/&lt;safe&gt;/);
const aligned=renderMarkdown('| Left | Right |\n| :--- | ---: |\n| a | b |');
assert.match(aligned,/class="align-right"/);assert.doesNotMatch(aligned,/style=/);
const incomplete='```python\nprint("<script>")';
assert.match(renderMarkdown(incomplete),/<pre><code class="language-python">print\(&quot;&lt;script&gt;&quot;\)/);
assert.match(renderMarkdown(incomplete+'\n```\n\nDone'),/<\/pre>\n<p>Done<\/p>/);
for(const value of ['<script>alert(1)</script>','<img src=x onerror=alert(1)>','<svg onload=alert(1)>','<a id="location" name="__proto__">x</a>']) {
  const result=renderMarkdown(value);assert.ok(!/<(?:script|img|svg|a)\b/.test(result));assert.match(result,/&lt;/);
}
for(const value of ['javascript:alert(1)','JaVaScRiPt:alert(1)','javascript&#58;alert(1)','vbscript:alert(1)','file:///tmp/secret','data:text/html,test','data:image/png;base64,AA==']) {
  assert.doesNotMatch(renderMarkdown(`[link](${value})`),/<a\b/);
  assert.doesNotMatch(renderMarkdown(`![image](${value})`),/<(?:img|a)\b/);
}
assert.match(renderMarkdown('![a < b](https://example.com/image.png)'),/a &lt; b<\/a>/);
assert.doesNotMatch(renderMarkdown('![a](https://example.com/image.png)'),/<img\b/);

assert.deepEqual(inputPresentation('**source**'),{text:'**source**',attachments:[],structured:false});
const canonical={version:1,prompt:{kind:'text',text:'Question **bold**'},files:[{path:'src/example.py',media_type:'text/plain',size:12}],attachments:[]};
assert.equal(inputPresentation(canonical).text,'Question **bold**');
assert.deepEqual(inputPresentation(canonical).attachments,['src/example.py · text/plain · 12 bytes']);
const nested={version:1,prompt:{kind:'items',items:[{kind:'text',text:'First'},{kind:'native',codec:'user-content-v1',value:{items:[{kind:'text-content',content:'Second',metadata:{text:'not a prompt'}},{kind:'image-url',url:'https://example.com/picture',media_type:'image/png'}]}},{kind:'binary',media_type:'application/pdf',size:40} ]}};
assert.equal(inputPresentation(nested).text,'First\n\nSecond');
assert.deepEqual(inputPresentation(nested).attachments,['https://example.com/picture · image/png','Attachment · application/pdf · 40 bytes']);
assert.deepEqual(inputPresentation({unknown:'data',prompt:{text:'not the canonical format'}}),{text:'',attachments:[],structured:true});
assert.deepEqual(inputPresentation({version:1,attachments:[{media_type:'image/png',size:5}]}).attachments,['Attachment · image/png · 5 bytes']);
console.log('Markdown and structured input contracts passed');
