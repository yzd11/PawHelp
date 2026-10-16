"use strict";
(()=>{
const byId = id => document.getElementById(id);
const node = (tag, text, cls) => { const n=document.createElement(tag); if(text!==undefined)n.textContent=String(text); if(cls)n.className=cls; return n; };
async function api(path, method="GET", body) {
  const response=await fetch(path,{method,cache:"no-store",headers:{"X-PawHelp-Local-Admin":"1","Content-Type":"application/json"},body:body===undefined?undefined:JSON.stringify(body)});
  const data=await response.json(); if(!response.ok)throw new Error(typeof data.detail==="string"?data.detail:`请求失败（${response.status}）`);return data;
}

async function busy(button, work) { button.disabled=true;try{await work();}catch(error){byId("status").textContent=error.message; if(byId("detailStatus"))byId("detailStatus").textContent=error.message;}finally{button.disabled=false;} }
function table(headers, rows){const t=node("table"),h=node("tr");headers.forEach(x=>h.append(node("th",x)));const head=node("thead");head.append(h);t.append(head);const body=node("tbody");rows.forEach(row=>{const r=node("tr");row.forEach(x=>r.append(node("td",x??"未提供")));body.append(r);});t.append(body);return t;}
const number=value=>typeof value==="number"&&Number.isFinite(value)?value.toFixed(4):"未评估";
window.PawHelpOperations={byId,node,api,busy,table,number};
})();
