// Syntax checks only. Browser behavior and visual checks remain separate.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.resolve(__dirname, "../app/static");
let count = 0;
for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
  if (!entry.isFile()) continue;
  const name = entry.name;
  const source = fs.readFileSync(path.join(root, name), "utf8");
  if (name.endsWith(".js")) {
    new vm.Script(source, { filename: name });
    count++;
  } else if (name.endsWith(".html")) {
    for (const match of source.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/gi)) {
      if (/\bsrc\s*=|\btype\s*=\s*["'](?:application\/json|importmap)/i.test(match[1])) continue;
      new vm.Script(match[2], { filename: name + ":inline" });
      count++;
    }
  }
}
console.log(`Frontend syntax: ${count} scripts checked.`);
