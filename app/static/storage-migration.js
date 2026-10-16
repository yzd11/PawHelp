"use strict";
(() => {
  const markerKey = "pawhelp.storageMigration.version";
  const backupKey = "pawhelp.storageMigration.backup";
  const targets = new Set(["pawhelp.userId", "pawhelp.conversationId", "pawhelp.messageFeedback.server"]);
  function valid(value, kind) {
    if (value === null) return true;
    if (kind === "uuid") return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
    if (kind === "positive_integer") return /^\d+$/.test(value) && Number.isSafeInteger(Number(value)) && Number(value) > 0;
    if (kind === "json_object") {
      try { const item = JSON.parse(value); return item !== null && typeof item === "object" && !Array.isArray(item); } catch { return false; }
    }
    return false;
  }
  function migrate(storage, policy) {
    if (!policy || !Array.isArray(policy.rules) || policy.rules.length > 3) throw new Error("本地存储迁移配置无效。");
    if (!policy.rules.length) return;
    if (typeof policy.id !== "string" || !/^[a-z0-9-]{1,80}$/.test(policy.id)) throw new Error("本地存储迁移版本无效。");
    if (storage.getItem(markerKey) === policy.id) return;
    const used = new Set();
    const changes = policy.rules.map(rule => {
      if (!rule || !targets.has(rule.target) || typeof rule.source !== "string" || !rule.source || rule.source.length > 100 || rule.source === rule.target || targets.has(rule.source) || used.has(rule.target) || !["uuid", "positive_integer", "json_object"].includes(rule.kind)) throw new Error("本地存储迁移规则无效。");
      used.add(rule.target);
      const sourceValue = storage.getItem(rule.source), targetValue = storage.getItem(rule.target);
      if (!valid(sourceValue, rule.kind) || !valid(targetValue, rule.kind)) throw new Error("本地会话数据格式异常，请勿清除浏览器数据，请联系维护人员。");
      if (sourceValue !== null && targetValue !== null && sourceValue !== targetValue) throw new Error("检测到两份不同的本地会话身份，已停止自动合并。请勿清除浏览器数据。");
      return {source: rule.source, target: rule.target, sourceValue, targetValue};
    });
    // Validate everything before mutation and retain a same-origin rollback copy.
    storage.setItem(backupKey, JSON.stringify({id: policy.id, changes}));
    try {
      for (const row of changes) {
        if (row.sourceValue !== null) storage.setItem(row.target, row.sourceValue);
        if (storage.getItem(row.target) !== (row.sourceValue ?? row.targetValue)) throw new Error("Local storage write verification failed");
      }
      for (const row of changes) if (row.sourceValue !== null) storage.removeItem(row.source);
      storage.setItem(markerKey, policy.id);
    } catch {
      for (const row of changes) {
        try {
          if (row.sourceValue !== null) storage.setItem(row.source, row.sourceValue);
          if (row.targetValue === null) storage.removeItem(row.target); else storage.setItem(row.target, row.targetValue);
        } catch { /* The durable local backup remains available for manual recovery. */ }
      }
      throw new Error("本地会话迁移未完成，请勿清除浏览器数据，请联系维护人员。");
    }
  }
  async function prepare() {
    const response = await fetch("/v1/client-storage-migration", {cache: "no-store"});
    if (!response.ok) throw new Error("无法读取本地会话迁移配置，请稍后刷新；请勿清除浏览器数据。");
    migrate(localStorage, await response.json());
  }
  window.PawHelpStorage = Object.freeze({migrate, prepare});
})();
