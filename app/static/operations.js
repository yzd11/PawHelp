"use strict";
const {byId,node,api,busy,table,number}=window.PawHelpOperations;
let offset=0, selected=null, reviewStatus="待审", listVersion=0, detailVersion=0, submitting=false;
const sourceNames={retrieval_low_conf:"检索置信度低",self_check:"模型自评证据不足",user_feedback:"用户反馈未解决"};
function localTime(value){const date=value?new Date(value):null;return date&&!Number.isNaN(date.getTime())?date.toLocaleString("zh-CN"):"时间未保存";}
function processingStatus(result){
 const summary=`处理 ${result.processed} 条：新建 ${result.created??0} 个缺口，归并 ${result.merged??0} 条；失败 ${result.failed} 条，因队列变化延后 ${result.deferred} 条。`;
 if(!result.failed)return summary;
 const details=(result.errors||[]).map(error=>`问题池 #${error.pool_id}：${error.message}`);
 return [summary,"失败原话仍在池中。请先确认原因再手动重试，重试会再次调用模型并产生费用。",...details].join("\n");
}
function originalEvidence(row){
 const container=node("div",undefined,"originals");
 if(!row.originals.length)container.append(node("p","尚无关联原话。"));
 row.originals.forEach(original=>{
  const card=node("div",undefined,"original-card"), meta=node("div",undefined,"original-meta");
  meta.append(node("span",sourceNames[original.source]||original.source,"source-tag"),node("time",localTime(original.created_at)),node("span",`问题池 #${original.id}`));
  card.append(meta,node("p",original.raw_question,"raw-question"));
  if(original.reason)card.append(node("p",original.reason,"muted"));
  const chunks=original.retrieved_chunks;
  if(chunks==null)card.append(node("p","未保存可恢复的召回快照：可能该轮未检索，或历史数据未记录。不能据此判断知识库缺失。","snapshot-empty"));
  else if(!chunks.length)card.append(node("p","已检索，零命中：请核查知识覆盖及检索配置，零命中不等于知识库必然缺失。","snapshot-empty"));
  else chunks.forEach(chunk=>{
   const evidence=node("div",undefined,"evidence-chunk");
   evidence.append(node("h4",chunk.title||chunk.question||chunk.section_path||`知识块 ${chunk.chunk_id??"未知"}`),node("pre",chunk.answer??"未保存原文"));
   const score=node("div",undefined,"score-row"),track=node("span",undefined,"score-track"),fill=node("span",undefined,"score-fill");
   const numeric=typeof chunk.score==="number"&&Number.isFinite(chunk.score);
   fill.style.width=`${numeric?Math.min(1,Math.max(0,chunk.score))*100:0}%`;
   track.setAttribute("aria-hidden","true");track.append(fill);
   score.append(node("span",`重排分 ${number(chunk.score)}`),track,node("span",chunk.section_path||"来源路径未保存","muted"));
   evidence.append(score);card.append(evidence);
  });
  container.append(card);
 });
 return container;
}
function reviewCard(row){
 const card=node("article",undefined,"review-card");card.dataset.reviewId=row.id;
 const head=node("div",undefined,"review-row"),badge=node("span",`×${row.occurrence_count}`,"frequency-badge"),body=node("div",undefined,"review-copy");
 body.append(node("h3",row.normalized_question),node("p",row.ai_suggested_answer||"无 AI 建议，请人工补充。","answer-preview"));
 const actions=node("div",undefined,"review-actions"),status=node("span",row.review_status,"status-tag");
 actions.append(status);
 if(row.review_status==="待审"){
  const approve=node("button","通过","approve");approve.dataset.action="approve";approve.onclick=()=>busy(approve,()=>detail(row.id));
  const reject=node("button","驳回","danger");reject.dataset.action="reject";reject.onclick=()=>busy(reject,async()=>{
   await api(`/api/review/${row.id}/reject`,"POST");await loadReviews();
   byId("status").textContent="已驳回，未写入知识库；同类问题再次出现只累加次数。";
  });actions.append(approve,reject);
 }else if(row.review_status==="通过"){
  actions.append(node("span",row.publication_status,"publication-tag"));
  const view=node("button",row.publication_status==="已同步"?"查看核准答案":"查看 / 重试同步");
  view.onclick=()=>busy(view,()=>detail(row.id));actions.append(view);
 }
 head.append(badge,body,actions);card.append(head);
 const expanded=node("details",undefined,"inline-evidence"),summary=node("summary",`展开归并原话与召回证据 · ${row.occurrence_count} 次`);
 expanded.append(summary);let loaded=false,loading=false;
 expanded.ontoggle=async()=>{
  if(!expanded.open||loaded||loading)return;loading=true;
  expanded.querySelector(".evidence-slot")?.remove();
  const slot=node("div","加载原话与快照…","evidence-slot");expanded.append(slot);
  try{const fresh=await api(`/api/review/${row.id}`);if(!expanded.isConnected)return;slot.replaceChildren(originalEvidence(fresh));loaded=true;}
  catch(error){slot.textContent=`加载失败：${error.message}。收起后重新展开可重试。`;}
  finally{loading=false;}
 };
 card.append(expanded);return card;
}
async function loadReviews(){
 const version=++listVersion, currentOffset=offset;
 const result=await api(`/api/review?status=${encodeURIComponent(reviewStatus)}&offset=${currentOffset}&sort=${byId("sort").value}`);
 if(version!==listVersion)return;
 byId("list").replaceChildren(...result.items.map(reviewCard));
 if(!result.items.length)byId("list").append(node("p","当前筛选下没有缺口；未标准化的原话需先手动处理。","empty-list"));
 byId("count").textContent=`尚未标准化的问题池记录：${result.unprocessed} 条；本页 ${result.items.length} 个缺口 · ${reviewStatus}`;
 byId("page").textContent=`第 ${currentOffset/50+1} 页`;byId("prev").disabled=currentOffset===0;byId("next").disabled=result.items.length<50;
 byId("status").textContent="列表已更新。";
}
async function detail(id){
 const version=++detailVersion,row=await api(`/api/review/${id}`);
 if(version!==detailVersion||submitting)return;
 selected=id;byId("question").textContent=row.normalized_question;
 byId("detailMeta").textContent=`出现 ${row.occurrence_count} 次 · ${row.review_status} · ${row.publication_status}${row.published_chunk_id? ` · 知识块 #${row.published_chunk_id}`:""}`;
 byId("approved").value=row.approved_answer??row.ai_suggested_answer??"";
 byId("approved").readOnly=row.review_status!=="待审";
 byId("originals").replaceChildren(originalEvidence(row));
 byId("detailStatus").textContent=row.publication_error||"";
 byId("approve").hidden=row.review_status!=="待审";
 byId("publish").hidden=row.review_status!=="通过"||row.publication_status==="已同步";
 if(!byId("detail").open)byId("detail").showModal();
}
async function submitReview(action){
 if(submitting||selected==null)return;
 if(action==="approve"&&!byId("approved").value.trim()){byId("detailStatus").textContent="请填写核准答案";return;}
 const id=selected,answer=byId("approved").value;submitting=true;
 ["approve","publish","close"].forEach(key=>byId(key).disabled=true);
 try{
  const result=await api(`/api/review/${id}/${action}`,"POST",action==="approve"?{approved_answer:answer}:undefined);
  byId("detail").close();++detailVersion;
  // The write succeeded. Retire the stale card even if the follow-up read fails.
  byId("list").querySelector(`[data-review-id="${id}"]`)?.remove();
  const outcome=result.publication_status==="已同步"?"知识已核准并同步。请开新对话重问同一问题，核对召回和回答。":result.publication_error||"已核准，知识待同步；可在通过页重试同步，无需再次核准。";
  byId("status").textContent=outcome;
  try{await loadReviews();byId("status").textContent=outcome;}
  catch(error){byId("status").textContent=`${outcome}\n列表刷新失败：${error.message}。请刷新列表，不要重复核准。`;}
 }catch(error){byId("detailStatus").textContent=error.message;}
 finally{submitting=false;["approve","publish","close"].forEach(key=>byId(key).disabled=false);}
}
byId("load").onclick=()=>busy(byId("load"),loadReviews);
if(byId("list")){
 document.querySelectorAll("[data-status]").forEach(button=>button.onclick=()=>{
  reviewStatus=button.dataset.status;offset=0;
  document.querySelectorAll("[data-status]").forEach(tab=>tab.setAttribute("aria-pressed",String(tab===button)));
  busy(byId("load"),loadReviews);
 });
 byId("sort").onchange=()=>{offset=0;busy(byId("load"),loadReviews);};
 byId("prev").onclick=()=>{offset=Math.max(0,offset-50);busy(byId("load"),loadReviews);};
 byId("next").onclick=()=>{offset+=50;busy(byId("load"),loadReviews);};
 byId("process").onclick=()=>{
  if(confirm("将调用项目配置的模型并产生费用，只处理配置的有限批量。继续？"))
   busy(byId("process"),async()=>{const result=await api("/api/review/process","POST");await loadReviews();byId("status").textContent=processingStatus(result);});
 };
 byId("close").onclick=()=>{if(!submitting){++detailVersion;byId("detail").close();}};
 byId("detail").addEventListener("cancel",event=>{if(submitting)event.preventDefault();else ++detailVersion;});
 byId("approve").onclick=()=>submitReview("approve");
 byId("publish").onclick=()=>submitReview("publish");
}

// Page load only reads; paid processing still requires confirmation.
busy(byId("load"),loadReviews);
