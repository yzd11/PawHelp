"use strict";
// Separate customer and operator navigation; this is not an authorization layer.
(() => {
  const customerPage = location.pathname === "/";
  const pages = customerPage ? [["/admin", "管理后台"]] : [["/kb", "知识库"], ["/review", "知识审核"], ["/topics", "主题分布"], ["/rag-eval", "检索评测"], ["/observability", "用量与成本"], ["/acceptance", "分类模型"]];
  document.body.classList.add(customerPage ? "customer-page" : "console-page");
  document.querySelectorAll("[data-app-navigation]").forEach(container => {
    container.classList.add("app-nav");
    container.setAttribute("aria-label", "全局页面导航");
    const links = pages.map(([path, label]) => {
      const link = document.createElement("a");
      link.href = path;
      link.textContent = label;
      if (location.pathname.replace(/\/$/, "") === path.replace(/\/$/, "") || (path !== "/" && location.pathname.startsWith(path + "/"))) link.setAttribute("aria-current", "page");
      return link;
    });
    container.replaceChildren(...links);
    if (!customerPage) {
      const sidebar = document.createElement("aside");
      sidebar.className = "console-sidebar";
      sidebar.setAttribute("aria-label", "PawHelp 管理后台");
      const brand = document.createElement("a");
      brand.className = "console-brand";
      brand.href = "/admin";
      brand.textContent = "PawHelp";
      const caption = document.createElement("p");
      caption.className = "console-caption";
      caption.textContent = "客户服务管理";
      const bottom = document.createElement("div");
      bottom.className = "console-sidebar-bottom";
      const chat = document.createElement("a");
      chat.href = "/";
      chat.textContent = "返回客户聊天 ↗";
      const mode = document.createElement("small");
      mode.textContent = "本地演示环境";
      bottom.append(chat, mode);
      sidebar.append(brand, caption, container, bottom);
      document.body.prepend(sidebar);
    }
  });
})();
