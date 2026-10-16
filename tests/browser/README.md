# Chat Markdown browser acceptance

The fixture serves the **real** chat HTML and local libraries with fixed history
and SSE output. No model calls, database writes, or third-party image requests.

Start in a separate terminal:

```powershell
node tests/browser/markdown_server.cjs
```

Run with the installed agent-browser:

```powershell
agent-browser --session markdown-check open http://127.0.0.1:8879
$markdownCode = Get-Content tests/browser/markdown_assertions.js -Raw -Encoding UTF8
$markdownEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($markdownCode))
agent-browser --session markdown-check eval -b $markdownEncoded

$markdownStreamCode = Get-Content tests/browser/markdown_stream_assertions.js -Raw -Encoding UTF8
$markdownStreamEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($markdownStreamCode))
agent-browser --session markdown-check eval -b $markdownStreamEncoded

agent-browser --session markdown-check set viewport 390 844
agent-browser --session markdown-check eval -b $markdownEncoded
agent-browser --session markdown-check close
```

Expected: 14 checks for each viewport and 3 SSE checks. Assertions throw on
failure. Stop the fixture terminal with Ctrl+C. Refreshing after a successful SSE
submission also reloads the fixed historical Markdown through the normal API.

Coverage: formatting, plain user input, known citations (excluding code/links),
source dialog, no executable raw HTML or auto-loaded images, unsafe URL removal,
partial streams, missing-sanitizer safe fallback, wide-content scrolling.
