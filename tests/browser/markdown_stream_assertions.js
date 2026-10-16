/* Exercise form submission and real SSE dispatch, not just the renderer. */
(async () => {
  const input = document.querySelector('#messageInput');
  input.value = '查询订单1001';
  input.dispatchEvent(new Event('input', {bubbles: true}));
  document.querySelector('#chatForm').requestSubmit();
  async function waitFor(predicate) {
    const until = Date.now() + 5000;
    while (!predicate()) {
      if (Date.now() > until) throw new Error('SSE Markdown assertion timed out');
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
  const lastCopy = () => [...document.querySelectorAll('.message-row.assistant .message-copy')].at(-1);
  await waitFor(() => lastCopy()?.querySelector('strong')?.textContent === '正在流式渲染');
  if (!lastCopy().querySelector('li')) throw new Error('Streaming list missing');
  await waitFor(() => lastCopy()?.querySelector('h2')?.textContent === '订单查询结果');
  if (!lastCopy().querySelector('.citation-ref') || lastCopy().querySelector('.streaming-caret')) throw new Error('Final SSE render/citation/caret invalid');
  return {passed: ['form submits to SSE', 'streaming bold and list', 'done renders Markdown with citation and removes caret']};
})();
