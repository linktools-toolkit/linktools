import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve} from 'node:path';

const root=process.argv[2], url=source=>'data:text/javascript;base64,'+Buffer.from(source).toString('base64');
const vendor=url(readFileSync(resolve(root,'markdown-it.js'),'utf8'));
const source=readFileSync(resolve(root,'message.js'),'utf8').replace("'./markdown-it.js'",JSON.stringify(vendor));
const {renderMarkdown,inputPresentation,capturedRequest}=await import(url(source));

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

// Native readable views never serialize parsed numbers or merge duplicate keys.
const original='{ "duplicate":1,"duplicate":2,"large":9007199254740993,"messages":[{"role":"developer","content":"  line\\n\\n\\tquote \\\" slash \\\\ 🌱 <b>literal</b>  ","unknown":-0},{"role":"assistant","content":"","tool_calls":[{"id":"call","function":{"name":"read","arguments":"{  \\\"x\\\": 1e400 }"}}]},{"role":"tool","tool_call_id":"call","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"https://invalid.example/image","detail":"low"}},{"type":"unknown","value":9007199254740993},{"type":"text","text":"after"}]}]}';
const native=capturedRequest(original),messages=native.messages[0].value.items();
assert.equal(native.source,original);assert.equal(native.messages.length,1);assert.equal(messages.length,3);
assert.deepEqual(native.fields.map(field=>field.name),['duplicate','duplicate','large','messages']);
assert.equal(native.fields[2].value.raw,'9007199254740993');assert.equal(messages[0].fields().at(-1).value.raw,'-0');
assert.equal(messages[0].fields()[1].value.text,'  line\n\n\tquote " slash \\ 🌱 <b>literal</b>  ');
const calls=messages[1].fields().find(field=>field.name==='tool_calls').value.items();
assert.equal(calls[0].fields().find(field=>field.name==='function').value.fields()[1].value.text,'{  "x": 1e400 }');
const blocks=messages[2].fields().find(field=>field.name==='content').value.items();assert.equal(blocks.length,4);
assert.equal(blocks[2].raw,'{"type":"unknown","value":9007199254740993}');
for(const source of ['{"messages":[null,0,"x",[],{}]}','{"messages":null}','{"messages":[]}','{}','null','[1]'])assert.equal(capturedRequest(source).source,source);
const duplicate=capturedRequest('{"messages":[],"messages":[{"role":"user","role":"tool","content":"first","\\u0063ontent":"second","unknown":1,"unknown":2}]}');
assert.equal(duplicate.messages.length,2);assert.deepEqual(duplicate.messages[1].value.items()[0].fields().map(field=>field.name),['role','role','content','content','unknown','unknown']);
assert.equal(duplicate.messages[1].value.items()[0].fields()[3].raw,'"\\u0063ontent":"second"');
for(const source of ['', '{"messages":', '{"messages":[{"content":"bad\nline"}]}'])assert.throws(()=>capturedRequest(source));
const large=capturedRequest('{"messages":['+Array.from({length:1000},(_,i)=>'{"role":"user","content":"'+i+'","unknown":{"nested":[0,1,2]}}').join(',')+']}');assert.equal(large.messages[0].value.items().length,1000);
assert.equal(large.messages[0].value.items()[999].fields()[1].value.text,'999');
console.log('Captured source contracts passed: ordered spans, decoded strings, exact unknown numbers, escaped duplicate keys, mixed content and invalid input');

assert.deepEqual(capturedRequest('{\"messages\":[{\"content\":\"x\",\"large\":9007199254740993,\"overflow\":1e400,\"negative_zero\":-0}]}').messages[0].value.items()[0].fields().slice(1).map(field=>field.value.raw),['9007199254740993','1e400','-0']);
