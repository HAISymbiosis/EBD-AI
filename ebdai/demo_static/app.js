'use strict';
const $ = id => document.getElementById(id);
const token = location.hash.slice(1) || sessionStorage.getItem('bdi-token') || '';
if(token) sessionStorage.setItem('bdi-token', token);
history.replaceState(null, '', location.pathname);
let documentState=null, demos=[], overrides={}, edits={}, tab='explore', runState={status:'idle'}, lastRunSignature='', startedAt=0, switching=false;
const busy=()=>['starting','running'].includes(runState.status);
const dirty=()=>Object.keys(edits).length>0;
async function api(path, body){
 const response=await fetch(path,{method:body?'POST':'GET',headers:{'X-Demo-Token':token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});
 const result=await response.json(); if(!response.ok) throw Error(result.error||response.statusText); return result;
}
function notice(message, action){
 const box=$('notice');box.replaceChildren();box.hidden=!message;
 if(message) box.append(document.createTextNode(message));
 if(action){const b=document.createElement('button');b.textContent='Reload from disk';b.onclick=()=>load(documentState.name,true).catch(showError);box.append(b);}
}
function showError(error){notice(error.message);}
function el(tag, text, cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
const parameterLabels={nRules:'Maximum rules',nAnts:'Antecedents per rule',n_gen:'Generations',pop_size:'Population size',patience:'Early stopping patience',random_state:'Random seed',n_linguistic_variables:'Fuzzy sets per feature',ds_mode:'Rule weighting mode',lam:'Fairness penalty',test_size:'Test split size',train_size:'Training split size',n_splits:'Cross-validation folds',max_rules:'Maximum rules',max_depth:'Maximum depth',n_output_lvs:'Output fuzzy sets',large_dataset_samples:'Large dataset samples',large_dataset_features:'Large dataset features'};
function menu(){
 $('menu').replaceChildren();
 for(const demo of demos.filter(d=>d.title.toLowerCase().includes($('search').value.toLowerCase()))){
  const button=el('button',demo.title);button.classList.toggle('selected',documentState?.name===demo.name);
  button.onclick=()=>load(demo.name).catch(showError);$('menu').append(button);
 }
}
async function load(name, discard=false){
 if(dirty()&&!discard&&!confirm('Discard unsaved browser edits and load this notebook?'))return;
 switching=true;
 try{documentState=await api('/api/notebook?name='+encodeURIComponent(name));overrides={};edits={};notice('');
 $('title').textContent=documentState.title;$('filename').textContent='Demos / '+name;
 menu();controls();render();sessionStorage.setItem('bdi-demo',name);
 }finally{switching=false;}
}
function controls(){
 $('parameters').replaceChildren();let group,previous;
 for(const p of documentState.parameters){
  const key=p.cell+':'+p.call;
  if(key!==previous){group=el('div',undefined,'parameter-group');group.append(el('h3',`Cell ${p.cell+1} · ${p.call}`));$('parameters').append(group);previous=key;}
  const label=el('label',undefined,'parameter');label.append(el('span',parameterLabels[p.name]||p.name));
  const input=el('input');input.dataset.parameter=p.name;input.setAttribute('aria-label',`${p.name}, cell ${p.cell+1}, ${p.call}`);
  input.value=JSON.stringify(overrides[p.id]===undefined?p.value:overrides[p.id]);
  input.title='JSON value: numbers, null, true/false, or quoted text';
  if(typeof p.value==='number'){input.type='number';input.step=Number.isInteger(p.value)?'1':'any';}
  input.onchange=()=>{try{const value=JSON.parse(input.value);if(value!==null&&!['number','boolean','string'].includes(typeof value)&&!Array.isArray(value))throw Error();if(JSON.stringify(value)===JSON.stringify(p.value))delete overrides[p.id];else overrides[p.id]=value;input.classList.toggle('changed',p.id in overrides);input.setCustomValidity('');}catch{input.setCustomValidity('Enter a number, null, true/false, quoted text, or a JSON list.');input.reportValidity();}};
  label.append(input);group.append(label);
 }
 if(!documentState.parameters.length)$('parameters').append(el('p','No literal hyperparameters found. Use Notebook & code to edit this example.'));
}
function inline(text){
 const fragment=document.createDocumentFragment();
 const pattern=/(\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
 let start=0;
 for(const match of text.matchAll(pattern)){
  fragment.append(document.createTextNode(text.slice(start,match.index)));
  if(match[2])fragment.append(el('strong',match[2]));
  else if(match[3])fragment.append(el('code',match[3]));
  else if(/^https?:\/\//.test(match[5])){const link=el('a',match[4]);link.href=match[5];link.target='_blank';link.rel='noopener noreferrer';fragment.append(link);}
  else fragment.append(document.createTextNode(match[4]));
  start=match.index+match[0].length;
 }
 fragment.append(document.createTextNode(text.slice(start)));return fragment;
}
function markdown(text){
 const box=el('div',undefined,'markdown');
 // Deliberately render notebook prose as text, never as unsandboxed HTML.
 for(const line of text.split('\n')){const match=line.match(/^(#{1,6}) (.*)/);const node=el(match?'h'+Math.min(match[1].length+1,6):'p');node.append(inline(match?match[2]:line));box.append(node);}
 return box;
}
function output(item){
 const box=el('div',undefined,'output');const data=item.data||{};const text=x=>Array.isArray(x)?x.join(''):x;
 if(item.output_type==='error'){box.classList.add('error');box.append(el('pre',text(item.traceback?.join('\n')||item.evalue).replace(/\x1b\[[0-9;]*m/g,'')));}
 else if(data['image/png']){const img=el('img');img.alt='Notebook plot';img.src='data:image/png;base64,'+text(data['image/png']);box.append(img);}
 else if(data['text/html']){const frame=el('iframe');frame.title='Notebook table';frame.setAttribute('sandbox','');frame.srcdoc='<meta charset="utf-8"><style>body{font:13px system-ui;color:#192d34}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #dce5e3;text-align:right}tr:nth-child(even){background:#f3f6f2}</style>'+text(data['text/html']);box.append(frame);}
 else {box.append(el('pre',text(item.text||data['text/plain']||'This rich output can be viewed in Jupyter.')));}
 return box;
}
function render(){
 if(!documentState)return;
 const content=$('content');content.replaceChildren();
 const sameRun=runState.name===documentState.name&&runState.notebook;
 const notebook=tab==='explore'&&sameRun?runState.notebook:documentState.notebook;
 if(tab==='code'){
  const tools=el('div',undefined,'code-tools');const save=el('button','Save notebook','primary');save.id='save';save.onclick=saveEdits;
  const reload=el('button','Reload from disk');reload.onclick=()=>load(documentState.name).catch(showError);
  tools.append(save,reload,el('span','Saves source to the original file. Runs never overwrite notebook outputs.'));content.append(tools);
 }else{
  content.append(el('div',sameRun?`Your run · ${runState.status}${runState.revision!==documentState.revision?' · notebook has changed since this run':''}`:'Saved notebook preview · choose settings and run to generate your own results.','badge'));
 }
 notebook.cells.forEach((cell,index)=>{
  if(cell.cell_type==='markdown'&&tab==='explore'){content.append(markdown(cell.source));return;}
  const section=el('article',undefined,'cell');const head=el('div',undefined,'cell-head');head.append(el('span',`CELL ${index+1} · ${cell.cell_type.toUpperCase()}`));
  if(cell.cell_type==='code'){const run=el('button','Run through here');run.disabled=busy();run.onclick=()=>startRun(index);head.append(run);}
  section.append(head);
  if(tab==='code'){
   const editor=el('textarea',undefined,'code');editor.value=edits[index]??cell.source;editor.spellcheck=false;editor.setAttribute('aria-label',`Cell ${index+1} source`);
   editor.style.height=Math.min(700,Math.max(150,editor.value.split('\n').length*22+36))+'px';
   editor.oninput=()=>{if(editor.value===cell.source)delete edits[index];else edits[index]=editor.value;$('save').textContent=dirty()?'Save notebook •':'Save notebook';};section.append(editor);
  }else{
   if(!(cell.outputs||[]).length)section.append(el('div','Run this section to see its output.','empty'));
   for(const item of cell.outputs||[])section.append(output(item));
   const details=el('details');details.style.padding='0 16px 12px';details.append(el('summary','Inspect executed code'),el('pre',cell.source));section.append(details);
  }
  content.append(section);
 });
 updateStatus();
}
async function saveEdits(){
 try{if(!dirty())return;documentState=await api('/api/save',{name:documentState.name,revision:documentState.revision,cells:edits});edits={};overrides={};controls();render();notice('Notebook saved. Controls now reflect your code.');}catch(error){notice(error.message,true);}
}
async function startRun(through=null){
 try{
  if(dirty())throw Error('Save your browser code edits before running.');
  if([...$('parameters').querySelectorAll('input')].some(input=>!input.reportValidity()))return;
  runState=await api('/api/run',{name:documentState.name,revision:documentState.revision,overrides,through});startedAt=Date.now();lastRunSignature='';notice('');tab='explore';syncTabs();render();
 }catch(error){showError(error);}
}
function updateStatus(){
 $('run').disabled=busy();$('stop').disabled=!busy();$('download').disabled=!runState.id||runState.name!==documentState?.name||busy();
 const matching=runState.name===documentState?.name;
 $('status').textContent=busy()?`${matching?'Running':runState.name} · cell ${(runState.cell??0)+1} of ${runState.total||'…'}`:matching?runState.status==='complete'?'Run complete':runState.status:'Ready';
 $('progress').max=runState.total||1;$('progress').value=runState.completed||0;
 $('elapsed').textContent=busy()&&startedAt?Math.floor((Date.now()-startedAt)/1000)+'s':'';
}
function syncTabs(){for(const button of document.querySelectorAll('.tab')){button.classList.toggle('active',button.dataset.tab===tab);button.setAttribute('aria-selected',String(button.dataset.tab===tab));}}
for(const button of document.querySelectorAll('.tab'))button.onclick=()=>{tab=button.dataset.tab;syncTabs();render();};
$('search').oninput=menu;$('run').onclick=()=>startRun();$('reset').onclick=()=>{overrides={};controls();};
$('stop').onclick=async()=>{try{runState=await api('/api/stop',{});render();}catch(error){showError(error);}};
$('download').onclick=async()=>{try{const response=await fetch('/api/download',{headers:{'X-Demo-Token':token}});if(!response.ok)throw Error('No run available');const url=URL.createObjectURL(await response.blob());const link=el('a');link.href=url;link.download=runState.name.replace(/\.(ipynb|py)$/,'')+'-run.ipynb';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){showError(error);}};
window.addEventListener('beforeunload',event=>{if(dirty()){event.preventDefault();event.returnValue='';}});
async function poll(){
 try{
  const next=await api('/api/state');runState=next;
  const signature=JSON.stringify([next.id,next.status,next.cell,next.completed]);
  if(signature!==lastRunSignature){lastRunSignature=signature;if(tab==='explore')render();if(next.status==='error')notice(next.error||'The notebook reported an error. Inspect the cell output.');}
  updateStatus();
  if(documentState&&!switching){const name=documentState.name;const current=await api('/api/revision?name='+encodeURIComponent(name));if(name===documentState.name&&current.revision!==documentState.revision){if(dirty())notice('The notebook changed on disk. Your browser edits have been kept; reload to use the disk version.',true);else await load(name);}}
 }catch(error){showError(error);}finally{setTimeout(poll,1800);}
}
(async()=>{try{demos=await api('/api/demos');const remembered=sessionStorage.getItem('bdi-demo');await load(demos.find(d=>d.name===remembered)?.name||demos[0].name);poll();}catch(error){showError(error);}})();
