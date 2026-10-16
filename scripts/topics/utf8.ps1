# Dot-source this ASCII-only file in the current PowerShell session.
# No system code page, PowerShell profile, model or corpus is changed.
# Python bytes and PowerShell native stdout/stderr decoding must agree.
$topicUtf8Encoding = New-Object System.Text.UTF8Encoding($false)
[Console]::InputEncoding = $topicUtf8Encoding
[Console]::OutputEncoding = $topicUtf8Encoding
$OutputEncoding = $topicUtf8Encoding
if (Get-Variable -Name PSApplicationOutputEncoding -ErrorAction SilentlyContinue) {
    $PSApplicationOutputEncoding = $topicUtf8Encoding
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
