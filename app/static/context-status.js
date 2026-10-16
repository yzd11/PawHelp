/* Background-only status: never gate sending, never reuse another conversation's response. */
(function () {
  "use strict";
  function create({ element, getContext, userId, intervalMs = 2000, timeoutMs = 4000 }) {
    let timer = null;
    let controller = null;
    let generation = 0;
    let disposed = false;
    async function poll(token) {
      if (disposed || token !== generation) return;
      const context = getContext();
      if (!context.id || document.hidden) {
        element.hidden = true;
        timer = setTimeout(() => poll(token), intervalMs);
        return;
      }
      const request = new AbortController();
      controller = request;
      const timeout = setTimeout(() => request.abort(), timeoutMs);
      const isCurrent = () => !disposed && token === generation &&
        getContext().id === context.id && getContext().epoch === context.epoch;
      try {
        const response = await fetch(`/api/conversations/${context.id}/context-status`, {
          headers: { "X-Demo-User-Id": userId }, cache: "no-store", signal: request.signal,
        });
        if (!response.ok) throw new Error("context status unavailable");
        const status = await response.json();
        if (isCurrent()) element.hidden = status.status !== "summarizing";
      } catch (_) {
        if (isCurrent()) element.hidden = true; // Status failure must not break chat.
      } finally {
        clearTimeout(timeout);
        if (controller === request) controller = null;
        if (!disposed && token === generation) timer = setTimeout(() => poll(token), intervalMs);
      }
    }
    function refresh() {
      if (disposed) return;
      generation += 1;
      clearTimeout(timer);
      if (controller) controller.abort();
      element.hidden = true;
      void poll(generation);
    }
    function stop() {
      disposed = true;
      generation += 1;
      clearTimeout(timer);
      if (controller) controller.abort();
      element.hidden = true;
      document.removeEventListener("visibilitychange", refresh);
    }
    document.addEventListener("visibilitychange", refresh);
    refresh();
    return { refresh, stop };
  }
  window.PawHelpContextStatus = { create };
})();
