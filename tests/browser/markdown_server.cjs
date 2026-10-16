/* Serve the real frontend with fake SSE/history: no database or model calls. */
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '../../app/static');
const content = '## 订单查询结果\n\n您之前查询的是 **订单 1001**，对应商品为 *猫爬架*。[1]\n\n'
  + '1. 已查询订单\n2. 已查询物流\n\n> 以下是演示信息，不是实时查询。\n\n'
  + '| 项目 | 结果 |\n| --- | --- |\n| 订单 | 1001 |\n| 状态 | 已签收 |\n\n'
  + '行内代码 `order_id=1001 [1]`。\n\n```json\n{"order_id":"1001","citation":"[1]"}\n```\n\n'
  + '[帮助文档](https://example.com/help) 和未登记引用 [99]。';
const sources = [{number: 1, chunk_id: 1, category: '订单'}];
const history = [
  {id: 1, role: 'user', content: '**我的输入保持原文**'},
  {id: 2, role: 'assistant', content, sources, tool_trace: [], suggested_actions: []},
];
const sse = (name, data) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
http.createServer((req, res) => {
  if (req.url === '/' || req.url.startsWith('/static/')) {
    const relative = req.url === '/' ? 'index.html' : decodeURIComponent(req.url.slice(8).split('?')[0]);
    const file = path.resolve(root, relative);
    if (file.startsWith(root + path.sep) && fs.existsSync(file) && fs.statSync(file).isFile()) {
      res.writeHead(200, {'content-type': file.endsWith('.js') ? 'text/javascript; charset=utf-8' : 'text/html; charset=utf-8'});
      return res.end(fs.readFileSync(file));
    }
  }
  if (req.url === '/v1/chat/stream' && req.method === 'POST') {
    req.resume();
    res.writeHead(200, {'content-type': 'text/event-stream; charset=utf-8'});
    res.write(sse('conversation', {conversation_id: 42}));
    res.write(sse('token', {content: '**正在流式渲染**\n\n- 第一项\n'}));
    setTimeout(() => res.end(sse('done', {content, sources, message_id: 2, conversation_id: 42, tool_trace: [], suggested_actions: []})), 1000);
    return;
  }
  res.setHeader('content-type', 'application/json; charset=utf-8');
  if (req.url.startsWith('/v1/conversations/42/messages')) return res.end(JSON.stringify(history));
  if (req.url.startsWith('/v1/conversations/42/pending-')) return res.end(JSON.stringify({pending: false}));
  if (req.url.startsWith('/v1/conversations')) return res.end('[]');
  if (req.url === '/v1/knowledge/chunks/1') return res.end(JSON.stringify({category:'订单', questions:'订单查询', answer:'演示订单 1001，已签收。', section_path:'演示订单'}));
  res.writeHead(404);
  res.end('{}');
}).listen(8879, '127.0.0.1', () => console.log('Markdown browser fixture http://127.0.0.1:8879'));
