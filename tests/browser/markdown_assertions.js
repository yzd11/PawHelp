/* Run through agent-browser eval --stdin on the real chat page. */
(async () => {
  const checks = [];
  function check(name, condition) {
    if (!condition) throw new Error('FAILED: ' + name);
    checks.push(name);
  }
  const history = await (await fetch('/v1/conversations/42/messages')).json();
  renderHistory(history);
  const copy = document.querySelector('.message-row.assistant .message-copy');
  check('bold and heading in restored history', copy.querySelector('strong').textContent === '订单 1001' && copy.querySelector('h2'));
  check('list, table, blockquote, inline and fenced code', copy.querySelector('ol li') && copy.querySelector('.markdown-table table') && copy.querySelector('blockquote') && copy.querySelector('pre code') && copy.querySelector('code'));
  check('user input remains plain', document.querySelector('.message-row.user .message-copy').textContent === '**我的输入保持原文**' && !document.querySelector('.message-row.user strong'));
  check('only body citations are buttons', copy.querySelectorAll('.citation-ref').length === 1 && copy.querySelector('pre code').textContent.includes('[1]') && copy.textContent.includes('[99]'));
  check('safe links hardened', copy.querySelector('a').getAttribute('rel') === 'noopener noreferrer');
  copy.querySelector('.citation-ref').click();
  await new Promise((resolve) => setTimeout(resolve, 100));
  check('citation modal still reads source', !document.querySelector('#sourceModal').hidden && document.querySelector('#sourceAnswer').textContent.includes('1001'));
  document.querySelector('#sourceClose').click();

  const assistant = addMessage('assistant');
  window.__markdownXss = 0;
  const attack = '<img src="/evil-image" onerror="window.__markdownXss=1">\n\n<script>window.__markdownXss=2</script>\n\n'
    + '[危险](javascript:window.__markdownXss=3)\n\n![跟踪图片](https://example.com/track.png)\n\n'
    + '<svg onload="window.__markdownXss=4"></svg>\n\n[恶意数据](data:text/html,evil)';
  renderFinalAnswer(assistant, attack, [], null);
  check('HTML and images cannot execute or load', window.__markdownXss === 0 && !assistant.copy.querySelector('script, img, svg, iframe, input'));
  check('dangerous links stripped', !assistant.copy.querySelector('a'));
  check('raw HTML stays visible as text', assistant.copy.textContent.includes('<img'));
  renderStreamingText(assistant, '**流式加粗**\n\n- 已返回');
  check('streaming Markdown and caret', assistant.copy.querySelector('strong').textContent === '流式加粗' && assistant.copy.querySelector('ul li') && assistant.copy.querySelector('.streaming-caret'));
  renderStreamingText(assistant, '**尚未结束');
  check('incomplete streaming Markdown remains readable', assistant.copy.textContent.includes('尚未结束'));
  const purifier = window.DOMPurify;
  window.DOMPurify = null;
  renderFinalAnswer(assistant, '**安全降级** <img src=x onerror=evil()> [1]', history[1].sources, null);
  check('missing sanitizer falls back without HTML', !assistant.copy.querySelector('img, strong') && assistant.copy.textContent.includes('**安全降级**') && assistant.copy.querySelector('.citation-ref'));
  window.DOMPurify = purifier;
  renderFinalAnswer(assistant, '| 长内容 | 第二列 | 第三列 |\n| --- | --- | --- |\n| ' + '很长的列'.repeat(30) + ' | 内容 | 内容 |\n\n```\n' + 'long-code-'.repeat(60) + '\n```', [], null);
  const scrollers = [...assistant.copy.querySelectorAll('.markdown-table, pre')];
  check('wide content scrolls inside bubble', scrollers.length === 2 && scrollers.every((el) => getComputedStyle(el).overflowX === 'auto'));
  check('page does not overflow horizontally', document.documentElement.scrollWidth <= innerWidth + 1);
  assistant.row.remove();
  return {passed: checks.length, checks};
})();
