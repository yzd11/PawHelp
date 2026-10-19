"use strict";
(()=>{
const {byId,node,api,busy,table,number}=window.PawHelpOperations;
const finite=value=>typeof value==="number"&&Number.isFinite(value);
const count=value=>finite(value)?value.toLocaleString("zh-CN",{maximumFractionDigits:0}):"未提供";
const percent=value=>finite(value)?`${(value*100).toFixed(1)}%`:"未评估";
const time=value=>value?new Date(value).toLocaleString("zh-CN"):"未记录";
const labels={recall_at_k:"Recall@K",mrr:"MRR",faithfulness:"Faithfulness",refusal_accuracy:"库外正确拒答率"};
let groups=[],loadVersion=0,calibrationVersion=0;
// Theme accents (theme.css); lines get an ink outline so pale colours stay legible.
const COLORS={ink:"#1f2937",rule:"#e5e5e5",sky:"#38bdf8",lime:"#a3e635",pink:"#f472b6",amber:"#fbbf24"};
function legendItem(label,color,dashed=false){
 const item=node("span"),swatch=node("i",undefined,dashed?"dashed":"");
 if(dashed)swatch.style.borderTopColor=color;else swatch.style.background=color;
 item.append(swatch,label);return item;
}

function cards(id,items){
 byId(id).replaceChildren(...items.map(([label,value,note,warning])=>{
  const card=node("div",undefined,`summary-card${warning?" warning":""}`);
  card.append(node("span",label,"card-label"),node("strong",value),node("p",note));return card;
 }));
}
function empty(id,text){byId(id).replaceChildren(node("p",text,"empty-report"));}

function plot(id,rows,series,{x,caption,ticks,markers=[]}){
 const ns="http://www.w3.org/2000/svg",svg=document.createElementNS(ns,"svg");
 svg.setAttribute("viewBox","0 0 900 350");svg.setAttribute("class","chart");
 svg.setAttribute("role","img");svg.setAttribute("aria-label",caption);
 const make=(tag,attrs,text)=>{
  const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>el.setAttribute(k,v));
  if(text!==undefined)el.textContent=text;svg.append(el);return el;
 };
 make("title",{},caption);
 for(let i=0;i<=4;i++){
  const y=25+(1-i/4)*245;make("line",{x1:55,y1:y,x2:850,y2:y,stroke:COLORS.rule,"stroke-width":2});
  make("text",{x:8,y:y+5},(i/4).toFixed(2));
 }
 const legend=node("div",undefined,"legend");
 series.forEach(([key,color,label])=>{
  let points=[];
  const flush=()=>{
   if(points.length){
    make("polyline",{points:points.join(" "),fill:"none",stroke:COLORS.ink,"stroke-width":6,"stroke-linejoin":"round"});
    make("polyline",{points:points.join(" "),fill:"none",stroke:color,"stroke-width":3,"stroke-linejoin":"round"});
   }points=[];
  };
  rows.forEach((row,i)=>{
   if(!finite(row[key])){flush();return;}
   const px=55+x(row,i)*795,py=25+(1-row[key])*245;points.push(`${px},${py}`);
   if(rows.length<25)make("circle",{cx:px,cy:py,r:4,fill:color,stroke:COLORS.ink,"stroke-width":2});
  });flush();legend.append(legendItem(label,color));
 });
 ticks.forEach(([position,label])=>make("text",{x:55+position*795,y:300,"text-anchor":"middle"},label));
 markers.forEach(([position,color,label],index)=>{
  if(!finite(position))return;
  make("line",{x1:55+position*795,y1:25,x2:55+position*795,y2:270,
   stroke:color,"stroke-width":2,"stroke-dasharray":"5 5"});
  legend.append(legendItem(label,color,true));
 });
 byId(id).replaceChildren(svg,legend);
}

function renderCosts(costs){
 const s=costs.summary, partial=s.unknown_usage_calls>0||(s.unobserved_requests||0)>0;
 const tokens=(row,key)=>row[key]>0||row.usage_complete?count(row[key]):"未采集";
 const average=row=>row.total_tokens>0||row.usage_complete
  ?`${count(row.average_tokens)}${row.usage_complete?"":"（部分）"}`:"未知";
 cards("costSummary",[
  ["已知 token 最多的意图",s.highest_intent||"暂无已知消耗",partial?"用量不完整，不能断言真实最贵意图":"按累计 token 排序"],
  [partial?"已知 token（不完整）":"总 token",s.total_tokens===0&&partial?"未采集":count(s.total_tokens),
   `${count(s.requests)} 次图执行 · 近 ${costs.days} 天`,partial],
  ["意图数",count(s.intent_count),`记录到 ${count(s.model_calls)} 次模型调用 · ${count(s.failed_calls)} 次失败`],
  ["报表刷新时间",time(costs.generated_at),
   `${count(s.unknown_usage_calls)} 次调用用量未知 · ${count(s.unobserved_requests||0)} 次请求无记录`,partial],
 ]);
 byId("costBars").replaceChildren(...costs.items.map(row=>{
  const bar=node("div",undefined,"cost-bar"),caption=node("div",undefined,"bar-caption"),
   track=node("div",undefined,"bar-track"),fill=node("div",undefined,"bar-fill");
  caption.append(node("strong",`${row.intent} · ${row.requests} 次请求`),node("span",
   `${finite(row.token_share)?percent(row.token_share):"占比未知"} · ${tokens(row,"total_tokens")} 已知 token`
   +` · 单均 ${average(row)}${row.unobserved_requests?` · ${row.unobserved_requests} 次无用量记录`:""}`));
  fill.style.width=`${finite(row.token_share)?row.token_share*100:0}%`;
  track.setAttribute("aria-hidden","true");track.append(fill);bar.append(caption,track);return bar;
 }));
 byId("costs").replaceChildren(node("p",costs.scope,"muted"),table(
  ["意图","请求","记录模型调用","输入 token","输出 token","已知总 token","单均 / 占比",
   "失败调用 / 未知用量调用","无用量记录请求","金额估算"],
  costs.items.map(r=>[r.intent,r.requests,r.model_calls,
   tokens(r,"input_tokens"),tokens(r,"output_tokens"),tokens(r,"total_tokens"),
   `${average(r)} / ${finite(r.token_share)?percent(r.token_share):"未知"}`,
   `${r.failed_calls} / ${r.unknown_usage_calls}`,
   r.unobserved_requests||0,r.estimated_cost==null?"不可估算":r.estimated_cost.toFixed(4)])));
 const highest=costs.items[0];
 byId("costInterpretation").textContent=[
  costs.truncated?"窗口内超过 5000 次请求：当前为最新 5000 次的截断样本，非完整窗口统计。":"",
  s.excluded_unlinked_empty_requests
   ?`已隔离 ${s.excluded_unlinked_empty_requests} 条无会话/消息关联且无用量的历史记录（包括测试残留），原记录未删除。`
    +"有用量的历史记录仍计入成本。":"",
  partial?"用量存在缺失，金额不能可靠估算，占比仅基于已知 token。":"",
  s.total_tokens>0
   ?`优先检查「${highest.intent}」链路：占已知 token ${percent(highest.token_share)}，`
    +`平均每请求 ${count(highest.average_tokens)} token。`
    +"高消耗可能来自长提示词、召回内容或多次工具 / 模型调用；到 Langfuse 核对，不能只据此归因。":
    s.requests?"该窗口暂无已知 token 消耗；如有用量未知，请先核对上报。":"该窗口暂无请求，不生成虚构占比。",
 ].filter(Boolean).join("\n");
}

function renderCalibration(report){
 byId("calibrationStatus").textContent=report.message;
 if(!report.available){
  cards("calibrationSummary",[
   ["当前有效阈值",number(report.active_threshold),report.active?"运行配置有效":"未确认有效校准",!report.active]]);
  ["calibrationDistribution","calibrationChart"].forEach(id=>empty(id,"尚无可重现的标注快照，无法绘制。"));
  byId("formula").textContent="暂无可校验的评分公式";
  byId("calibrationReading").textContent=report.active_reason||report.message;
  byId("calibrationNote").textContent="缺少校准数据，请先完成离线校准。";return;
 }
 const stats=report.validation_at_active||report.validation_at_recommended;
 cards("calibrationSummary",[
  ["离线推荐阈值",number(report.recommended_threshold),
   `训练 ${report.training_size} 题 · ${report.validation_passed?"独立验证通过":"独立验证失败，不可发布"}`
   +` · 区间 ${report.optimal_range.map(number).join("–")}`,!report.validation_passed],
  ["当前有效阈值",report.active?number(report.active_threshold):"未生效",
   report.active?"仅展示，不自动修改":"历史建议不代表当前配置有效",!report.active],
  ["验证集 · 相关放行率",percent(stats.relevant_pass_rate),`${report.validation_size} 题独立验证 · ${report.active?"当前":"推荐"}阈值`],
  ["验证集 · 不相关误放率",percent(stats.irrelevant_pass_rate),
   `不相关样本中被放行的比例 · 误放 ${stats.unsafe_accepted} 条`,stats.unsafe_accepted>0],
 ]);
 byId("calibrationDistribution").replaceChildren(table(["类别","题数","最低","p25","中位","p75","最高"],
  report.distributions.map(r=>[r.group,r.count,...["min","p25","median","p75","max"].map(k=>number(r[k]))])));
 plot("calibrationChart",report.scan,[
  ["relevant_pass_rate",COLORS.lime,"相关样例放行率"],["irrelevant_pass_rate",COLORS.pink,"不相关样例误放率"]],{
  x:r=>r.threshold,caption:"训练集证据相关性阈值扫描，纵轴为样例比例，横轴为阈值；独立验证不参与选择",
  ticks:[0,.2,.4,.6,.8,1].map(t=>[t,`阈值 ${t.toFixed(1)}`]),
  markers:[[report.recommended_threshold,COLORS.ink,`推荐 ${number(report.recommended_threshold)}`],
   [report.active_threshold,COLORS.sky,`当前 ${number(report.active_threshold)}`]],
 });
 const train=report.training_at_active||report.training_at_recommended;
 byId("calibrationReading").textContent=
  `${report.active?"当前":"历史推荐"}阈值下：训练集相关放行 ${percent(train.relevant_pass_rate)}，`
  +`不相关误放 ${percent(train.irrelevant_pass_rate)}；放行中不相关占比 ${percent(train.unsafe_share)}。`
  +`独立验证相关放行 ${percent(stats.relevant_pass_rate)}，不相关误放 ${percent(stats.irrelevant_pass_rate)}。`
  +"这些是有限快照结果，不是生产安全保证。";
 byId("formula").textContent=report.formula;
 byId("calibrationNote").textContent=[report.active?"当前校准与文档 / 模型版本匹配。":`当前校准未生效：${report.active_reason}`,
  `标注来源：${report.reviewer||"未记录"}`,report.scope_limit].filter(Boolean).join("\n");
}

function metricCell(value,delta){
 const td=node("td",undefined,"metric");td.append(node("span",number(value)));
 if(finite(delta))td.append(node("span",
  `${Math.abs(delta)<1e-9?" →":delta<0?" ↓":" ↑"} ${delta>0?"+":""}${delta.toFixed(4)}`,
  delta< -1e-9?"delta-down":delta>1e-9?"delta-up":""));return td;
}
function renderTrend(){
 const group=groups.find(g=>g.key===byId("group").value);
 if(!group){
  cards("trendSummary",[["已跑轮次","0","暂无保存的评估轮次"]]);empty("trend","至少两轮同配置完整评估后显示趋势。");
  empty("runs","尚无评估轮次。");byId("trendInterpretation").textContent="暂无评估记录。";return;
 }
 const latest=group.latest;
 cards("trendSummary",[
  ["本组轮次",String(group.rows.length),`${group.complete_count} 轮完整 · 最近最多 100 轮记录`],
  ["最新完整 · Faithfulness",number(latest?.values.faithfulness),
   latest?`轮次 #${latest.id} · ${time(latest.created_at)}`:"尚无完整评估"],
  ["最新完整 · MRR",number(latest?.values.mrr),`Recall@K ${number(latest?.values.recall_at_k)}`],
  ["较前轮下滑",latest?.previous_id?`${group.declining.length} 项`:"不可比较",
   latest?.previous_id?`对比同组完整轮次 #${latest.previous_id}`:"至少需要两轮完整评估",group.declining.length>0],
 ]);
 const valid=group.rows.filter(r=>r.comparable).reverse();
 if(valid.length<2)empty("trend","本组尚不足两轮完整评估，缺失指标不补零。" );
 else plot("trend",valid.map(r=>r.values),[["recall_at_k",COLORS.sky,"Recall@K"],["mrr",COLORS.lime,"MRR"],
  ["faithfulness",COLORS.pink,"Faithfulness"],["refusal_accuracy",COLORS.amber,"库外正确拒答率"]],{
  x:(_,i)=>i/(valid.length-1),caption:"同配置完整评估轮次趋势；横轴按评估顺序，缺失分数留空",
  ticks:valid.map((r,i)=>[i/(valid.length-1),`#${r.id}`])
   .filter((_,i)=>i===0||i===valid.length-1||i%Math.ceil(valid.length/8)===0),
 });
 const t=table(["轮次 / 时间","触发","题数",...Object.values(labels),"状态 / 对比轮次"],[]),body=t.querySelector("tbody");
 group.rows.forEach(r=>{
  const tr=node("tr");
  tr.append(node("td",`#${r.id} · ${time(r.created_at)}`),node("td",r.triggered_by),node("td",r.dataset_size));
  Object.keys(labels).forEach(k=>tr.append(metricCell(r.values[k],r.deltas[k])));
  tr.append(node("td",r.comparable?`完整${r.previous_id?` · 对比 #${r.previous_id}`:" · 首轮基线"}`:"部分 / 无比较标识（不连线）"));
  body.append(tr);
 });byId("runs").replaceChildren(t);
 byId("trendInterpretation").textContent=!latest?.previous_id?"当前无可比较的前一轮；请勿把不同配置或缺失分数当成退化。":group.declining.length?
  `轮次 #${latest.id} 对比 #${latest.previous_id}：${group.declining.map(k=>labels[k]).join("、")} 下滑，`
  +"请检查这两轮报告、失败样例和知识变更；单次变化不代表因果结论。":
  `轮次 #${latest.id} 对比 #${latest.previous_id}：已评估指标未出现下滑；未评估指标不参与判断。`;
}

async function loadObservations(){
 const version=++loadVersion,calVersion=++calibrationVersion;
 const result=await api(`/api/observability?days=${byId("days").value||7}`);
 if(version!==loadVersion)return;
 const c=result.configuration;
 byId("configuration").textContent=`Langfuse ${result.costs.langfuse_enabled?"已启用（导出情况在控制台核对）":"未启用"}`
  +` · 标准化周期 ${c.flywheel_interval_seconds||"关闭"} · 评估周期 ${c.evaluation_interval_seconds||"关闭"}`
  +` · 每桶 ${c.sample_per_bucket} 题 · 单轮模型调用上限 ${c.max_model_calls}`;
 try{const url=new URL(c.langfuse_url);if(["http:","https:"].includes(url.protocol))byId("langfuse").href=url.href;}
 catch{/* Keep the safe default link. */}
 renderCosts(result.costs);if(calVersion===calibrationVersion)renderCalibration(result.calibration);
 const oldKey=byId("group").value;groups=result.trend.groups;
 byId("group").replaceChildren(...groups.map((g,i)=>{
  const option=node("option",`组 ${i+1} · ${g.rows[0].dataset_size} 题 · ${g.key.slice(0,10)} · ${g.complete_count} 轮完整`);
  option.value=g.key;return option;
 }));
 if(groups.some(g=>g.key===oldKey))byId("group").value=oldKey;
 renderTrend();
 byId("status").textContent=result.last_evaluation_error
  ||`报表已刷新 · ${time(result.costs.generated_at)} · 共 ${result.trend.total_runs} 轮评估`;
}
byId("load").onclick=()=>busy(byId("load"),loadObservations);
byId("days").onchange=()=>busy(byId("load"),loadObservations);
byId("group").onchange=renderTrend;
byId("analyzeCalibration").onclick=()=>busy(byId("analyzeCalibration"),async()=>{
 const version=++calibrationVersion;byId("calibrationStatus").textContent="正在后台离线扫描已有标注快照…";
 try{
  const report=await api("/api/observability/calibration/analyze","POST");
  if(version===calibrationVersion)renderCalibration(report);
 }catch(error){
  if(version===calibrationVersion)byId("calibrationStatus").textContent=`重算失败：${error.message}`;throw error;
 }
});
byId("evaluate").onclick=()=>{
 if(confirm("小批量评估会发送题目和召回知识给模型并产生费用；保存到独立报告目录，不覆盖原报告。继续？"))
  busy(byId("evaluate"),async()=>{await api("/api/observability/evaluate","POST");await loadObservations();});
};
busy(byId("load"),loadObservations);
})();
