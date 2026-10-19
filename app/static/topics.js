"use strict";
(() => {
  const {byId, node, api, table} = window.PawHelpOperations;
  const content = byId("content"), path = location.pathname;
  const params = new URLSearchParams(location.search);
  const subpages = [["/acceptance", "运行概览"], ["/acceptance/eval", "质量报告"], ["/acceptance/errors", "问题复核"]];
  const sources = {low_confidence: "检索置信度低", user_feedback: "用户反馈未解决",
    negative_feedback: "用户反馈未解决", retrieval_low_confidence: "检索置信度低",
    retrieval_low_conf: "检索置信度低", self_check: "回答自检未通过",
    synthetic: "模拟样本", real_pool: "真实问题池"};
  const reviews = {pending: "待审核", approved: "已通过", rejected: "已驳回", published: "已发布",
    "待审": "待审核", "通过": "已通过", "驳回": "已驳回"};
  const score = value => typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
  const percent = value => typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
  const date = value => value && Number.isFinite(Date.parse(value))
    ? new Date(value).toLocaleString("zh-CN", {hour12: false}) : "—";
  const body = (report, key) => report[key]?.present ? report[key].data : null;
  const currentEvaluation = report => report.evaluation?.present && !report.evaluation.stale
    ? report.evaluation.data : null;
  function heading(title, subtitle) {
    byId("title").textContent = title; document.title = `${title} · PawHelp`;
    byId("subtitle").textContent = subtitle;
  }
  function link(href, text, cls) { const a = node("a", text, cls); a.href = href; return a; }
  function section(title, note) {
    const box = node("section"); box.append(node("h2", title));
    if (note) box.append(node("p", note, "muted")); content.append(box); return box;
  }
  function notice(box, text, tone = "") { box.append(node("p", text, `topic-notice ${tone}`)); }
  function empty(box, text) { box.append(node("p", text, "empty-list")); }
  function badge(text, tone = "") { return node("span", text, `topic-tag ${tone}`); }
  function cards(values, target = content) {
    const box = node("div", undefined, "topic-cards");
    for (const [label, value, note] of values) {
      const card = node("article"); card.append(node("div", label, "muted"), node("p", value ?? "—", "topic-value"));
      if (note) card.append(node("small", note, "muted")); box.append(card);
    }
    target.append(box); return box;
  }
  function tags(labels, classify = () => "") {
    const box = node("span", undefined, "topic-tags");
    for (const label of labels || []) box.append(badge(label, classify(label))); return box;
  }
  function tabular(box, headers, rows) {
    const wrap = node("div", undefined, "table-wrap"); wrap.append(table(headers, rows)); box.append(wrap);
  }
  function details(box, title, text) {
    const d = node("details"); d.append(node("summary", title), node("p", text, "muted")); box.append(d); return d;
  }
  function pageLink(label, page, overrides = {}) {
    const query = new URLSearchParams({label, page});
    if (path === "/topics/questions") for (const key of ["q", "source", "review_status"]) {
      const value = overrides[key] ?? params.get(key); if (value && value !== "all") query.set(key, value);
    }
    return `/topics/questions?${query}`;
  }
  function foldSection(box, title) {
    const fold = node("details", undefined, "topic-technical");
    fold.append(node("summary", title || box.querySelector("h2")?.textContent || "详细信息"));
    box.before(fold); fold.append(box); box.querySelector("h2")?.remove();
    box.classList.add("topic-fold-body"); return fold;
  }
  function selection(label, name, options, value) {
    const field = node("label", label, "topic-field"), select = node("select"); select.name = name;
    for (const [key, text] of options) { const o = node("option", text); o.value = key; select.append(o); }
    select.value = value || "all"; field.append(select); return field;
  }
  function pager(target, page, pages, total, change) {
    const nav = node("nav", undefined, "topic-pagination"); nav.setAttribute("aria-label", "分页");
    for (const [label, next, disabled] of [["← 上一页", page - 1, page <= 1], ["下一页 →", page + 1, page >= pages]]) {
      const button = node("button", label); button.disabled = disabled;
      button.addEventListener("click", () => change(next)); nav.append(button);
    }
    nav.append(node("span", `第 ${page} / ${Math.max(1, pages)} 页 · 共 ${total} 条 · 每页 10 条`, "muted"));
    target.append(nav);
  }
  async function topics() {
    heading("主题分布", "了解用户关注的问题，安排知识维护。");
    const report = await api("/api/topics/distribution");
    cards([["已归类记录", report.total], ["独立问题", report.unique_gaps_total, "按归并条目去重"],
      ["有记录的主题", `${report.hit_topics} / ${report.topics.length}`], ["待归类", report.pending_merged, "已完成问题归并"]]);
    const toolbar = node("div", undefined, "topic-page-actions");
    toolbar.append(node("p", report.latest ? `最近归类 ${date(report.latest)}` : "暂无归类记录", "muted"),
      link("/review", "前往知识审核 →", "topic-action primary"));
    content.append(toolbar);
    const box = section("主题概览");
    const filters = node("div", undefined, "topic-filterbar"), searchLabel = node("label", "搜索主题", "topic-field");
    const search = node("input"); search.type = "search"; search.placeholder = "输入主题名称"; searchLabel.append(search);
    const sortField = selection("排序", "sort", [["count", "问题记录最多"], ["unique_gaps", "独立问题最多"]], "count");
    const toggleLabel = node("label", undefined, "topic-checkbox"), toggle = node("input");
    toggle.type = "checkbox"; toggleLabel.append(toggle, node("span", "显示无记录主题"));
    filters.append(searchLabel, sortField, toggleLabel); box.append(filters);
    const results = node("div"); box.append(results);
    function render() {
      results.replaceChildren();
      const key = sortField.querySelector("select").value;
      const visible = [...report.topics]
        .filter(item => (toggle.checked || item.count > 0) && item.label.includes(search.value.trim()))
        .sort((a, b) => b[key] - a[key] || b.count - a.count);
      if (!visible.length) empty(results, report.total
        ? "没有符合条件的主题，请调整筛选。" : "暂无归类记录。完成问题归并与分类后，结果将显示在这里。");
      const max = Math.max(1, ...report.topics.map(item => item.count));
      for (const item of visible) {
        const d = node("details", undefined, `topic-distribution-row ${item.count ? "" : "is-zero"}`),
          row = node("summary", undefined, "topic-overview-row");
        const name = node("span", undefined, "topic-name");
        name.append(node("strong", item.label), node("small", `${item.unique_gaps} 个独立问题`, "muted"));
        const track = node("span", undefined, "topic-track"), fill = node("span", undefined, "topic-fill");
        fill.style.width = `${item.count / max * 100}%`; track.append(fill); track.setAttribute("aria-hidden", "true");
        row.append(name, track, node("strong", `${item.count} 条`, "topic-record-count"),
          node("span", "查看示例", "topic-expand-label"));
        d.append(row);
        const samples = node("div", undefined, "topic-samples"), list = node("ul");
        for (const text of item.samples) list.append(node("li", text));
        if (!item.samples.length) samples.append(node("p", "该主题暂无问题。", "muted"));
        samples.append(list, link(pageLink(item.label, 1), "查看问题明细 →", "topic-action"));
        d.append(samples); results.append(d);
      }
      results.append(node("p", `显示 ${visible.length} / ${report.topics.length} 个主题 · 一条问题可能属于多个主题`, "muted"));
    }
    search.addEventListener("input", render); sortField.querySelector("select").addEventListener("change", render);
    toggle.addEventListener("change", render); render();
  }
  async function questions() {
    const label = params.get("label") || "商品信息", requestedPage = Number(params.get("page"));
    const page = Number.isFinite(requestedPage) ? Math.max(1, Math.floor(requestedPage)) : 1;
    const request = new URLSearchParams({label, page});
    for (const key of ["q", "source", "review_status"]) if (params.get(key)) request.set(key, params.get(key));
    const [report, distribution] = await Promise.all([api(`/api/topics/questions?${request}`),
      api("/api/topics/distribution")]);
    heading(`${label} · 问题明细`, "查看用户问法、来源和知识审核进度。");
    const categories = node("nav", undefined, "topic-category-tabs");
    categories.setAttribute("aria-label", "切换主题类目");
    for (const item of [...distribution.topics].sort((a, b) => b.count - a.count)) {
      const a = link(pageLink(item.label, 1), `${item.label} ${item.count}`, item.count ? "" : "is-zero");
      if (item.label === label) a.setAttribute("aria-current", "page"); categories.append(a);
    }
    content.append(categories);
    const box = section("问题列表");
    const top = node("div", undefined, "topic-page-actions");
    top.append(link("/topics", "← 主题分布", "topic-action"), link("/review", "知识审核 →", "topic-action"));
    box.append(top);
    const form = node("form", undefined, "topic-filterbar"); form.method = "get"; form.action = "/topics/questions";
    const hidden = node("input"); hidden.type = "hidden"; hidden.name = "label"; hidden.value = label;
    const textLabel = node("label", "搜索问题", "topic-field"), search = node("input");
    search.type = "search"; search.name = "q"; search.maxLength = 200; search.placeholder = "输入问题关键词";
    search.value = params.get("q") || ""; textLabel.append(search);
    const sourceField = selection("问题来源", "source", [["all", "全部来源"], ["retrieval_low_conf", "检索置信度低"],
      ["self_check", "回答自检未通过"], ["user_feedback", "用户反馈未解决"]], params.get("source"));
    const reviewField = selection("知识审核", "review_status", [["all", "全部状态"], ["待审", "待审核"], ["通过", "已通过"],
      ["驳回", "已驳回"], ["unmerged", "未归并"]], params.get("review_status"));
    const submit = node("button", "筛选"); submit.type = "submit";
    form.append(hidden, textLabel, sourceField, reviewField, submit,
      link(pageLink(label, 1, {q: "", source: "all", review_status: "all"}), "重置", "topic-reset"));
    box.append(form, node("p", `筛选结果 ${report.total} 条`, "muted"));
    if (!report.items.length) empty(box, report.total ? "本页没有记录，请返回上一页或切换类目。" : "该类暂无已归类问题。");
    for (const item of report.items) {
      const card = node("article", undefined, "topic-question"); card.append(node("h3", item.text), tags(item.labels));
      const meta = node("div", undefined, "topic-meta");
      for (const text of [`来源：${sources[item.source] || item.source || "未记录"}`,
        `同义归并：${item.occurrence_count == null ? "未归并" : `${item.occurrence_count} 条原话`}`,
        `审核：${reviews[item.review_status] || item.review_status || "未归并"}`, `归类于 ${date(item.classified_at)}`]) {
        meta.append(node("span", text));
      }
      card.append(meta); details(card, "查看用户原话", item.raw_text);
      if (!item.normalized) card.append(node("p", "未归并，当前显示用户原话。", "muted")); box.append(card);
    }
    pager(box, report.page, report.pages, report.total, next => { location.href = pageLink(label, next); });
    details(box, "字段说明", "归并频次为同一独立问题累计的原话数量；审核状态属于知识审核，不代表分类标签已人工确认。");
  }
  function policyFor(report, label) {
    const policy = body(report, "acceptance_policy");
    for (const [key, title] of [["strict", "严格"], ["standard", "标准"], ["observe", "观察"]]) {
      if (policy?.[key]?.labels?.includes(label)) return {title, minimum: policy[key].minimum_f1};
    }
    return {title: "未配置", minimum: null};
  }
  function grade(row, report) {
    const policy = policyFor(report, row.label);
    if (policy.minimum == null) return {text: "不设线", tone: "neutral"};
    if (!row.support) return {text: "无正例，未评估", tone: "warning"};
    return row.f1 >= policy.minimum
      ? {text: `≥ ${policy.minimum.toFixed(2)} 达标`, tone: "good"}
      : {text: `< ${policy.minimum.toFixed(2)} 未达标`, tone: "bad"};
  }
  function invalidReport(box, report, key, name) {
    if (!report[key]?.present) {
      empty(box, `${name}尚未生成${report[key]?.error ? `：${report[key].error}` : "。"}`); return true;
    }
    if (report[key].stale) {
      notice(box, `${name}与当前模型或数据不匹配，不用于当前验收。${(report[key].stale_reasons || []).join("；")}`, "warning");
      return true;
    }
    return false;
  }
  function metric(value) {
    const box = node("div", undefined, "topic-metric"); box.append(node("span", score(value)));
    const track = node("span", undefined, "topic-mini-track"), fill = node("span");
    fill.style.width = `${Math.max(0, Math.min(1, value || 0)) * 100}%`;
    track.append(fill); box.append(track); return box;
  }
  function evaluations(report) {
    heading("质量报告", "查看离线测试结果及需要关注的类目。");
    const data = currentEvaluation(report), summary = section("离线测试表现");
    if (!invalidReport(summary, report, "evaluation", "测试集评测")) {
      const grid = node("div", undefined, "topic-two-columns");
      for (const [key, title, explanation] of [
        ["micro", "micro · 全部标签汇总", "先汇总所有类目的 TP、FP、FN，再计算指标。"],
        ["macro", "macro · 类目等权平均", "分别计算各类指标，再对全部类目取平均。"]]) {
        const card = node("article"); card.append(node("h3", title), node("p", explanation, "muted"));
        const values = node("div", undefined, "topic-metric-values");
        for (const [k, name] of [["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]]) {
          const v = node("div"); v.append(node("strong", score(data[key][k])), node("small", name)); values.append(v);
        }
        card.append(values); grid.append(card);
      }
      summary.append(grid);
      cards([["测试样本", data.size], ["标签集合完全一致", percent(data.exact_match)], ["当前分类阈值", data.threshold],
        ["错例问题", data.errors.wrong_samples]], summary);
      notice(summary, `测试来源：${Object.entries(data.source_counts || {})
        .map(([key, count]) => `${sources[key] || key} ${count} 条`).join("、") || "未记录"}。`
        + "总分不能替代每类分析，也不能代表生产环境真实准确率。");
      const s = section("每类指标与验收标准",
        "P 为精确率，R 为召回率；support 为标准答案含此标签的样本数（TP + FN）。验收线不改变分类阈值。");
      const t = node("table"), head = node("thead"), tr = node("tr");
      for (const text of ["类目", "分档", "P", "R", "F1", "support", "F1 验收"]) tr.append(node("th", text));
      head.append(tr); t.append(head);
      const tbody = node("tbody");
      for (const row of data.per_class) {
        const r = node("tr"), g = grade(row, report);
        for (const value of [row.label, policyFor(report, row.label).title]) r.append(node("td", value));
        for (const key of ["precision", "recall", "f1"]) {
          const cell = node("td"); cell.append(metric(row[key])); r.append(cell);
        }
        r.append(node("td", row.support)); const status = node("td"); status.append(badge(g.text, g.tone));
        r.append(status); tbody.append(r);
      }
      t.append(tbody);
      const wrap = node("div", undefined, "table-wrap topic-evaluation-table"); wrap.append(t); s.append(wrap);
      const policy = body(report, "acceptance_policy");
      notice(s, policy
        ? `严格档 F1 ≥ ${score(policy.strict.minimum_f1)}；标准档 ≥ ${score(policy.standard.minimum_f1)}；观察档不设线，不标记为通过。`
          + "无正例的类目不可判为达标。此标准仅用于当前离线测试，不等于上线批准。"
        : "尚未配置有效的每类 F1 验收标准，仅展示真实指标，不判定通过。");
      const matrices = section("每类混淆矩阵",
        "每个类目独立判断是否命中。行表示实际标签，列表示模型预测；错位通常同时产生一次 FP 和一次 FN。");
      const fp = data.errors.false_positive_labels, fn = data.errors.false_negative_labels;
      notice(matrices, `${data.size} 条问题 × ${data.per_class.length} 类 = ${data.size * data.per_class.length} 次标签判断；`
        + `误报 FP ${fp} 次，漏报 FN ${fn} 次，共 ${fp + fn} 次错误判断，涉及 ${data.errors.wrong_samples} 条问题。`);
      const matrixGrid = node("div", undefined, "topic-matrices");
      for (const row of data.per_class) {
        const c = node("article"); c.append(node("h3", row.label));
        const cells = node("div", undefined, "topic-matrix-cells");
        for (const [value, title, tone] of [
          [row.matrix[0][0], "TN 正确不标", "neutral"], [row.matrix[0][1], "FP 误报", "bad"],
          [row.matrix[1][0], "FN 漏报", "bad"], [row.matrix[1][1], "TP 正确标注", "good"]]) {
          const cell = node("div", undefined, tone);
          cell.append(node("strong", value), node("small", title)); cells.append(cell);
        }
        c.append(cells); matrixGrid.append(c);
      }
      matrices.append(matrixGrid, link("/acceptance/errors", "查看问题复核 →", "topic-action"));
      foldSection(matrices, "每类混淆矩阵与错误统计");
    }
    thresholdScan(report);
  }
  function thresholdScan(report) {
    const box = section("验证集阈值扫描",
      "阈值仅由验证集选择，不使用测试集调参。横条为验证 micro-F1，统一以 0～1 为刻度；页面不会重新运行推理。");
    foldSection(box, "阈值选择记录");
    if (!body(report, "threshold")) { empty(box, "暂无阈值扫描报告。"); return; }
    const data = report.threshold.data, model = body(report, "model"), chosen = data.chosen;
    if (!chosen || !Array.isArray(data.trials)) { empty(box, "阈值报告不完整。"); return; }
    for (const trial of data.trials) {
      const row = node("div", undefined, `topic-scan-row ${trial.threshold === chosen.threshold ? "selected" : ""}`);
      const track = node("span", undefined, "topic-track"), fill = node("span", undefined, "topic-fill");
      fill.style.width = `${Math.max(0, Math.min(1, trial.micro_f1)) * 100}%`; track.append(fill);
      row.append(node("strong", trial.threshold.toFixed(2)), track, node("span", score(trial.micro_f1)),
        node("small", `${trial.threshold === chosen.threshold ? "选定 · " : ""}FP ${trial.fp} / FN ${trial.fn}`,
          "muted"));
      box.append(row);
    }
    const matches = chosen.threshold === model?.threshold;
    notice(box, `验证样本 ${report.threshold.validation_size ?? "未记录"} 条；`
      + `选定 ${chosen.threshold.toFixed(2)}（micro-F1 ${score(chosen.micro_f1)}）；`
      + `当前模型配置 ${model?.threshold ?? "缺失"}，${matches ? "数值一致" : "不一致，请核对模型产物"}。`, matches ? "good" : "warning");
    details(box, "选择规则与版本说明", "当前流程按 micro-F1 优先，其次误报更少，再其次阈值更高。"
      + "扫描文件未保存独立模型版本，阈值数值一致不构成完整版本一致性证明；推理服务状态请在总览核对。");
  }
  function errors(report) {
    heading("问题复核", "对照标准标签与模型预测，定位需要改进的问题。"); const box = section("历史错例 · 来源待核对");
    if (invalidReport(box, report, "errors", "错例报告")) return;
    const rows = report.errors.data, missed = rows.reduce((n, row) => n + row.missed.length, 0),
      extra = rows.reduce((n, row) => n + row.extra.length, 0);
    cards([["错例问题", rows.length], ["漏报 FN", missed], ["误报 FP", extra], ["错误标签判断", missed + extra]], box);
    notice(box, "来源待核对：此错例文件未保存独立模型版本或摘要，汇总数量相同不能证明属于当前模型。"
      + "以下供历史复核，不计入当前验收通过状态。漏标计 FN，多标计 FP；错位同时计两者。", "warning");
    const toolbar = node("div", undefined, "topic-filterbar"), label = node("label", "错误类型", "topic-field"),
      select = node("select");
    select.setAttribute("aria-label", "按错误类型筛选");
    for (const [value, title] of [["all", "全部"], ["missed", "仅漏标"], ["extra", "仅多标"], ["mixed", "错位 / 漏标与多标并存"]]) {
      const o = node("option", title); o.value = value; select.append(o);
    }
    const searchLabel = node("label", "搜索问题", "topic-field"), search = node("input");
    search.type = "search"; search.placeholder = "输入问题关键词"; searchLabel.append(search);
    label.append(select); toolbar.append(searchLabel, label); box.append(toolbar);
    const list = node("div"); box.append(list);
    function render(page = 1) {
      list.replaceChildren();
      const type = row => row.missed.length && row.extra.length ? "mixed" : row.missed.length ? "missed" : "extra";
      const filtered = rows.filter(row => (select.value === "all" || type(row) === select.value)
        && row.text.toLocaleLowerCase().includes(search.value.trim().toLocaleLowerCase()));
      if (!filtered.length) empty(list, "此筛选条件下没有错例。");
      for (const row of filtered.slice((page - 1) * 10, page * 10)) {
        const card = node("article", undefined, "topic-error-card"), top = node("div", undefined, "topic-meta");
        top.append(badge({missed: "漏标", extra: "多标", mixed: "错位"}[type(row)],
          type(row) === "mixed" ? "bad" : "warning"),
          node("span", `${row.missed.length + row.extra.length} 次标签错误`, "muted"));
        card.append(top, node("h3", row.text));
        const expected = node("div", undefined, "topic-label-row");
        expected.append(node("span", "标准"), tags(row.expected, x => row.missed.includes(x) ? "missed" : "good"));
        const predicted = node("div", undefined, "topic-label-row");
        predicted.append(node("span", "预测"), tags(row.predicted, x => row.extra.includes(x) ? "extra" : "good"));
        card.append(expected, predicted,
          node("p", `漏标：${row.missed.join("、") || "无"}；多标：${row.extra.join("、") || "无"}`, "muted"),
          node("p", `${sources[row.source] || row.source || "来源未记录"} · `
            + `复核：${reviews[row.review_status] || row.review_status || "未记录"}`, "muted"));
        details(card, "相关类目分数与记录标识", `${row.id} · ${[...new Set([...row.expected, ...row.predicted])]
          .map(x => `${x} ${score(row.scores?.[x])}`).join(" · ")}`);
        list.append(card);
      }
      pager(list, page, Math.ceil(filtered.length / 10), filtered.length, render);
    }
    select.addEventListener("change", () => render()); search.addEventListener("input", () => render()); render();
  }
  function size(bytes) {
    if (typeof bytes !== "number") return "—"; if (bytes < 1024) return `${bytes} B`;
    return bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }
  function inventory(box, rows) {
    if (!rows) { empty(box, "文件清单需要新版后台接口，请重启 8000 服务后刷新。"); return; }
    tabular(box, ["文件（相对分类器数据目录）", "条数", "大小", "最后修改", "状态"], rows.map(row => [row.path, row.rows ?? "—",
      size(row.size_bytes), date(row.modified_at), row.error || (row.present ? "已生成" : "未生成")]));
  }
  function dataArtifacts(report) {
    heading("技术资料", "供维护人员追溯数据、模型及运行记录。");
    const intro = section("模型维护资料", "以下为只读记录；查看页面不会触发训练、付费调用或重新评测。");
    intro.append(link("/acceptance", "← 返回运行概览", "topic-action"));
    const inv = report.inventory, dataset = body(report, "dataset"), model = body(report, "model");
    const lineage = section("数据流转", "以下为当前本地流水线产物，不将旧参考数据计入本次训练；只有训练集进行增强。");
    const count = name => inv?.corpus?.find(row => row.path === name)?.rows;
    cards([["真实池快照", count("pool-20261013.jsonl"), "已脱敏导出的历史快照，不是实时池总量"],
      ["真实预标 / 模拟样本", `${count("pool-prelabels.jsonl") ?? "—"} / ${count("synthetic-v1.jsonl") ?? "—"}`,
        "候选预标，需抽审后冻结"],
      ["审定 / 近义分组", `${count("corpus-approved-v1.jsonl") ?? "—"} / ${count("corpus-grouped-v2.jsonl") ?? "—"}`,
        "保留原样本与分组溯源"],
      ["训练 / 验证 / 测试", dataset
        ? ["train", "val", "test"].map(k => dataset.after_augmentation?.[k]?.size ?? "—").join(" / ") : "—",
      "整组划分，增强仅应用于训练集"]], lineage);
    if (report.review_summary?.present) {
      notice(lineage, `抽审记录：${report.review_summary.samples} 条；`
        + `${report.review_summary.approved ? "已有明确批准记录" : "尚未批准"}。批次批准不代表每条模拟样本均经过人工逐条审核。`);
    }
    inventory(lineage, inv?.corpus);
    const split = section("数据集划分与增强");
    if (dataset) {
      tabular(split, ["分区", "增强前", "增强后", "增强范围"], [["train", "训练"], ["val", "验证"], ["test", "测试"]]
        .map(([key, title]) => [title, dataset.before_augmentation?.[key]?.size ?? "—",
          dataset.after_augmentation?.[key]?.size ?? "—", key === "train" ? "仅训练样本生成措辞变体" : "保持冻结，不增强"]));
      const overlap = Object.entries(dataset.overlaps || {})
        .map(([key, value]) => `${key}：文本 ${value.texts} / 分组 ${value.groups}`).join("；");
      notice(split, `已记录的交叉重叠：${overlap || "未提供"}。分组属于保守近义处理，不保证消除所有语义近重复。`);
    } else empty(split, "尚无当前数据集划分清单。"); inventory(split, inv?.dataset);
    const artifacts = section("训练模型与运行配置", "权重、分词器、类目顺序和阈值共同组成当前分类器。");
    if (model) {
      cards([["模型版本", model.model_version], ["分类阈值", model.threshold],
        ["可训练参数", model.trainable_parameters?.toLocaleString("zh-CN")], ["最大文本长度", model.max_length]], artifacts);
      notice(artifacts, `基座 ${model.base_model}；${model.labels?.length ?? "—"} 类独立多标签。`
        + "未命中具体类目时按当前策略使用“其他”兜底，不将概率最高的类目强行判为命中。");
    } else empty(artifacts, "尚未生成微调模型配置。"); inventory(artifacts, inv?.model);
    const train = section("训练过程"), training = body(report, "training");
    if (training) {
      const epochs = (training.history || []).filter(row => typeof row.eval_micro_f1 === "number");
      tabular(train, ["轮次", "验证 micro-F1", "验证 macro-F1", "验证 loss"], epochs.map(row => [row.epoch,
        score(row.eval_micro_f1), score(row.eval_macro_f1), score(row.eval_loss)]));
      notice(train, `最佳验证 micro-F1 ${score(training.best_metric)}。`
        + `最佳检查点：${training.best_checkpoint || "未记录"}。不使用测试集选择训练轮次。`);
    } else empty(train, "暂无训练记录。");
    const exports = section("推理格式与环境记录");
    notice(exports, "当前服务使用 Transformers / PyTorch 推理，未提供 ONNX 导出与一致性校验流程，不标记为已导出或校验通过。");
    const environment = body(report, "environment");
    if (environment) {
      tabular(exports, ["训练设备", "历史环境记录", "记录时间"],
        [[environment.device, environment.witness, date(environment.timestamp)]]);
    }
    notice(exports, "环境记录只证明记录时间点的运行情况，不能代替当前服务健康检查。");
    const reports = section("评测与审计文件"); inventory(reports, inv?.reports);
    details(reports, "历史参考数据的使用边界",
      "旧参考语料来源未认证，历史审计报告不等于当前模型独立测试结果。数据和模型默认不上传 GitHub。");
    for (const box of [lineage, split, artifacts, train, exports, reports]) foldSection(box);
  }
  function overview(report, service, distribution) {
    heading("分类模型", "查看分类运行情况与质量，跟进待处理问题。");
    const model = body(report, "model"), evaluation = currentEvaluation(report);
    const sameVersion = service.online && model?.model_version && service.model_version === model.model_version;
    cards([["分类服务", service.online ? (sameVersion ? "在线" : "版本待核对") : "不可用", "用于问题整理，不影响实时客服对话"],
      ["已归类记录", distribution?.total ?? "—"], ["待归类", distribution?.pending_merged ?? "—", "已完成问题归并"],
      ["离线测试 F1", evaluation ? score(evaluation.micro.f1) : "待核对", "不代表生产环境准确率"]]);
    const quality = section("质量关注");
    if (service.online && !sameVersion) {
      notice(quality, "运行中的分类服务与本地模型版本未确认一致，请维护人员核对后再处理新批次。", "warning");
    }
    if (!service.online) notice(quality, "分类服务暂不可用，新增问题暂不能完成批量归类；已有主题结果仍可查看。", "warning");
    const statuses = evaluation?.per_class.map(row => grade(row, report)) || [],
      failed = statuses.filter(x => x.tone === "bad" || x.tone === "warning").length,
      checked = statuses.filter(x => x.tone !== "neutral").length;
    if (evaluation) {
      quality.append(badge(checked ? (failed ? "存在未达标项" : "设线类目达标") : "未配置验收线",
        checked ? (failed ? "warning" : "good") : "neutral"));
      quality.append(node("p", `${checked} 个设线类目中，${failed} 个未达标或缺少正例；观察档不计通过。`));
      notice(quality, `测试样本 ${evaluation.size} 条：${Object.entries(evaluation.source_counts || {})
        .map(([key, count]) => `${sources[key] || key} ${count} 条`).join("、") || "来源未记录"}。`
        + "离线结果不能替代真实业务验证，也不能证明多诉求问题均能正确识别。");
    } else {
      invalidReport(quality, report, "evaluation", "测试报告");
    }
    const actions = node("div", undefined, "topic-page-actions");
    actions.append(link("/acceptance/eval", "查看质量报告 →", "topic-action"),
      link("/acceptance/errors", "复核历史错例 →", "topic-action"));
    quality.append(actions);
    const batch = section("离线批量归类"), last = report.batch;
    if (last?.present) {
      const names = {complete: "已完成", no_pending: "无待归类问题", waiting_for_batch: "待积累批次", failed: "失败",
        started: "进行中 / 结果未确认"};
      const failure = last.status === "complete" || last.failed > 0 ? last.failed
        : ["no_pending", "waiting_for_batch"].includes(last.status) ? "未执行归类" : "未确认";
      tabular(batch, ["最近批次", "已确认保存", "失败记录", "开始时间", "完成时间"], [[names[last.status] || "未知状态",
        last.saved, failure, date(last.started_at), last.finished_at ? date(last.finished_at) : "未记录"]]);
      if (last.status === "started") {
        notice(batch, "最近记录尚未确认结束，可能仍在处理或已中断。请维护人员核对进程及日志；不要仅凭本页重复提交批次。", "warning");
      }
      if (last.model_version && last.model_version !== model?.model_version) {
        notice(batch, "最近批次使用了不同模型版本；历史归类不会自动重算。", "warning");
      }
    } else empty(batch, "暂无批量归类运行记录。");
    if (distribution) {
      notice(batch, `已归并待归类 ${distribution.pending_merged} 条；`
        + `未归类问题共 ${distribution.unclassified} 条。未归并问题不进入当前批次。`);
    }
    const results = node("div", undefined, "topic-page-actions");
    results.append(link("/topics", "查看主题结果 →", "topic-action primary"), link("/review", "前往知识审核 →", "topic-action"));
    batch.append(results);
    details(batch, "处理范围", "分类批次由维护人员手动运行，只处理已归并且未归类的问题；刷新页面不会运行批次，也不会修改历史结果。");
    const diagnostics = section("运行诊断");
    tabular(diagnostics, ["项目", "当前记录"], [["服务模型版本", service.model_version || "不可用"],
      ["本地模型版本", model?.model_version || "未生成"], ["分类阈值", model?.threshold ?? "未生成"]]);
    diagnostics.append(link("/acceptance/data", "模型与数据技术资料 →", "topic-action")); foldSection(diagnostics);
  }
  async function refresh() {
    content.replaceChildren(); byId("status").textContent = "正在读取…"; byId("refresh").disabled = true;
    try {
      if (path === "/topics") await topics(); else if (path === "/topics/questions") await questions();
      else {
        const report = await api("/api/acceptance/reports");
        if (path === "/acceptance/eval") evaluations(report);
        else if (path === "/acceptance/data") dataArtifacts(report);
        else if (path === "/acceptance/errors") errors(report);
        else {
          const [service, distribution] = await Promise.all([api("/api/acceptance/service"),
            api("/api/topics/distribution").catch(() => null)]);
          overview(report, service, distribution);
        }
      }
      byId("status").textContent = `已更新 · ${new Date().toLocaleTimeString("zh-CN", {hour12: false})}`;
    } catch (e) { byId("status").textContent = `加载失败：${e.message}。请检查服务后点击刷新。`; }
    finally { byId("refresh").disabled = false; }
  }
  for (const [href, title] of path.startsWith("/acceptance")
    ? subpages : [["/topics", "主题分布"], ["/acceptance", "分类模型"]]) {
    const a = link(href, title);
    if (href === path || (href === "/topics" && path === "/topics/questions")) a.setAttribute("aria-current", "page");
    byId("subnav").append(a);
  }
  if (path.startsWith("/acceptance")) {
    const tech = link("/acceptance/data", "技术资料", "topic-secondary-link");
    if (path === "/acceptance/data") tech.setAttribute("aria-current", "page"); byId("subnav").append(tech);
    byId("countPolicy").hidden = true;
  }
  byId("refresh").addEventListener("click", refresh); refresh();
})();
