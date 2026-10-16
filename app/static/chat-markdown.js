/* Model output is untrusted. Parse, sanitize, then add our own citation controls. */
(() => {
  "use strict";
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  const parser = window.marked && new window.marked.Marked({
    gfm: true, breaks: true, async: false,
    renderer: {
      html: ({ text }) => escapeHtml(text),
      // Do not let model output load tracking images or arbitrary external resources.
      image: ({ text }) => escapeHtml(text || "[图片]"),
    },
  });

  function render(target, text, sources = [], onSource = () => {}) {
    const content = String(text || "");
    target.classList.remove("markdown-content");
    if (parser && window.DOMPurify && window.DOMPurify.isSupported) {
      try {
        const fragment = window.DOMPurify.sanitize(parser.parse(content), {
          RETURN_DOM_FRAGMENT: true,
          ALLOWED_TAGS: ["p", "br", "strong", "em", "del", "h1", "h2", "h3", "h4", "h5", "h6",
            "ul", "ol", "li", "blockquote", "pre", "code", "a", "hr", "table", "thead", "tbody", "tr", "th", "td"],
          ALLOWED_ATTR: ["href", "title", "start", "align"],
          ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false,
        });
        for (const link of fragment.querySelectorAll("a")) {
          const href = link.getAttribute("href");
          let safe = false;
          try { safe = Boolean(href) && ["https:", "http:", "mailto:"].includes(new URL(href, location.href).protocol); } catch (_) {}
          if (!safe) { link.replaceWith(...link.childNodes); continue; }
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }
        for (const table of fragment.querySelectorAll("table")) {
          const wrapper = document.createElement("div");
          wrapper.className = "markdown-table";
          table.replaceWith(wrapper);
          wrapper.appendChild(table);
        }
        target.replaceChildren(fragment);
        target.classList.add("markdown-content");
      } catch (_) {
        target.textContent = content;
      }
    } else {
      // Missing/blocked assets must not break chat or turn raw output into HTML.
      target.textContent = content;
    }

    const byNumber = new Map((Array.isArray(sources) ? sources : [])
      .filter((item) => item && Number.isSafeInteger(item.number) && item.number > 0
        && Number.isSafeInteger(item.chunk_id) && item.chunk_id > 0)
      .map((item) => [item.number, item]));
    if (!byNumber.size) return;
    const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.parentElement.closest("pre, code, a, button")) nodes.push(node);
    }
    for (const node of nodes) {
      const replacement = document.createDocumentFragment();
      for (const part of node.textContent.split(/(\[\d+\])/g)) {
        const match = part.match(/^\[(\d+)]$/);
        const source = match && byNumber.get(Number(match[1]));
        if (!source) { replacement.appendChild(document.createTextNode(part)); continue; }
        const button = document.createElement("button");
        button.type = "button";
        button.className = "citation-ref";
        button.textContent = part;
        button.setAttribute("aria-label", `查看来源 ${source.number}`);
        button.addEventListener("click", () => onSource(source));
        replacement.appendChild(button);
      }
      node.replaceWith(replacement);
    }
  }
  window.PawHelpMarkdown = Object.freeze({ render });
})();
