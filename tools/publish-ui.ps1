param([string]$ArticlePath = '', [switch]$SmokeTest, [string]$RenderPath = '')
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()
$root = Split-Path $PSScriptRoot -Parent
$cache = Join-Path $root '.blog-publish'
[System.IO.Directory]::CreateDirectory($cache) | Out-Null
$script:task = $null
$script:inspection = $null
$script:selectedPath = $ArticlePath
$form = New-Object System.Windows.Forms.Form
$form.Text = 'kanosa · Typora 文章发布'
$form.Size = New-Object System.Drawing.Size(790, 690)
$form.MinimumSize = $form.Size
$form.StartPosition = 'CenterScreen'
$form.BackColor = [System.Drawing.Color]::FromArgb(245, 246, 240)
$form.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 10)

function Label($text, $x, $y, $width, $height) {
  $label = New-Object System.Windows.Forms.Label
  $label.Text = $text; $label.SetBounds($x, $y, $width, $height)
  $form.Controls.Add($label); return $label
}
function Button($text, $x, $y, $width) {
  $button = New-Object System.Windows.Forms.Button
  $button.Text = $text; $button.SetBounds($x, $y, $width, 36)
  $button.FlatStyle = 'Flat'; $button.BackColor = [System.Drawing.Color]::White
  $form.Controls.Add($button); return $button
}
$heading = Label '在 Typora 写作，在这里发布。' 25 22 700 36
$heading.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 18, [System.Drawing.FontStyle]::Bold)
$null = Label '选择保存好的 .md 原稿，工具会整理本地图片、检查网页并推送。' 27 67 700 30
$fileBox = New-Object System.Windows.Forms.TextBox
$fileBox.SetBounds(27, 113, 590, 30); $fileBox.ReadOnly = $true
$fileBox.Text = $ArticlePath; $form.Controls.Add($fileBox)
$choose = Button '选择文章…' 630 109 120
$null = Label '文章分类' 27 161 90 28
$category = New-Object System.Windows.Forms.ComboBox
$category.SetBounds(116, 157, 280, 30); $category.DropDownStyle = 'DropDownList'
$category.Items.AddRange(@('沿用原稿（未填写时为经验分享）', '论文介绍', '经验分享', '课程笔记'))
$category.SelectedIndex = 0; $form.Controls.Add($category)
$inspect = Button '重新检查' 420 154 110
$openOriginal = Button '打开原稿' 544 154 100
$summary = Label '尚未选择文章。可以双击发布入口，也可以把 .md 文件拖到入口上。' 27 208 723 110
$summary.BackColor = [System.Drawing.Color]::FromArgb(228, 235, 223)
$summary.Padding = New-Object System.Windows.Forms.Padding(12)
$publish = Button '检查通过后发布' 27 335 205
$publish.BackColor = [System.Drawing.Color]::FromArgb(35, 89, 77)
$publish.ForeColor = [System.Drawing.Color]::White
$publish.Enabled = $false
$visit = Button '查看线上博客 ↗' 248 335 165
$state = Label '等待选择文章' 27 387 720 28
$state.Font = New-Object System.Drawing.Font('Microsoft YaHei UI', 10, [System.Drawing.FontStyle]::Bold)
$logs = New-Object System.Windows.Forms.TextBox
$logs.SetBounds(27, 423, 723, 177); $logs.Multiline = $true; $logs.ReadOnly = $true
$logs.ScrollBars = 'Vertical'; $logs.BackColor = [System.Drawing.Color]::White
$logs.Font = New-Object System.Drawing.Font('Consolas', 9)
$form.Controls.Add($logs)
$null = Label '原稿在博客目录之外时不会被改动；外链图片仍依赖原站。发布期间请勿关闭窗口。' 27 610 725 27

function Read-Log($filename) {
  $stream = New-Object System.IO.FileStream($filename, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
  $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
  try { return $reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Begin-Task($mode) {
  if ($script:task) { return }
  if (-not (Test-Path -LiteralPath $script:selectedPath -PathType Leaf)) { $state.Text = '请先选择保存好的 Markdown 原稿'; return }
  $choose.Enabled = $false; $inspect.Enabled = $false; $category.Enabled = $false; $publish.Enabled = $false
  $logs.Clear()
  $arguments = '"' + (Join-Path $PSScriptRoot 'publish-article.cjs') + '" ' + $mode + ' "' + $script:selectedPath + '"'
  if ($category.SelectedIndex -gt 0) { $arguments += ' --category "' + $category.SelectedItem + '"' }
  if ($mode -eq '--publish') { $arguments += ' --expected-hash ' + $script:inspection.sourceHash }
  $token = [Guid]::NewGuid().ToString('N')
  $stdout = Join-Path $cache ($token + '.stdout.log'); $stderr = Join-Path $cache ($token + '.stderr.log')
  $start = New-Object System.Diagnostics.ProcessStartInfo
  $start.FileName = (Get-Command node -ErrorAction Stop).Source
  $start.Arguments = $arguments; $start.WorkingDirectory = $root
  $start.UseShellExecute = $false; $start.CreateNoWindow = $true
  $start.RedirectStandardOutput = $true; $start.RedirectStandardError = $true
  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $start
  $outFile = New-Object System.IO.FileStream($stdout, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1, [System.IO.FileOptions]::Asynchronous)
  $errFile = New-Object System.IO.FileStream($stderr, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite, 1, [System.IO.FileOptions]::Asynchronous)
  $process.Start() | Out-Null
  $script:task = @{ Process = $process; Mode = $mode; OutFile = $outFile; ErrFile = $errFile; Stdout = $stdout; Stderr = $stderr
    OutCopy = $process.StandardOutput.BaseStream.CopyToAsync($outFile); ErrCopy = $process.StandardError.BaseStream.CopyToAsync($errFile) }
  $state.Text = $(if ($mode -eq '--inspect') { '正在检查原稿与图片…' } else { '正在准备并发布，请稍候…' })
}
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 350
$timer.Add_Tick({
  if (-not $script:task) { return }
  try {
    $task = $script:task
    if ($task.Mode -eq '--publish') {
      $text = (Read-Log $task.Stdout) + (Read-Log $task.Stderr)
      if ($text.Length -gt 20000) { $text = $text.Substring($text.Length - 20000) }
      $logs.Text = $text; $logs.SelectionStart = $logs.TextLength; $logs.ScrollToCaret()
      $stages = [regex]::Matches($text, '\[进度\] ([^\r\n]+)')
      if ($stages.Count) { $state.Text = $stages[$stages.Count - 1].Groups[1].Value }
    }
    if (-not $task.Process.HasExited -or -not $task.OutCopy.IsCompleted -or -not $task.ErrCopy.IsCompleted) { return }
    $task.OutCopy.GetAwaiter().GetResult(); $task.ErrCopy.GetAwaiter().GetResult()
    $task.OutFile.Dispose(); $task.ErrFile.Dispose()
    $output = Read-Log $task.Stdout; $errorText = Read-Log $task.Stderr
    $exitCode = $task.Process.ExitCode; $task.Process.Dispose(); $script:task = $null
    $choose.Enabled = $true; $inspect.Enabled = $true; $category.Enabled = $true
    if ($exitCode -ne 0) {
      $state.Text = '未发布：请按下方提示处理后重新检查'
      $logs.Text = $errorText + "`r`n" + $output; $script:inspection = $null
    } elseif ($task.Mode -eq '--inspect') {
      $script:inspection = $output | ConvertFrom-Json
      $item = $script:inspection
      $kind = $(if ($item.exists) { '更新同名文章（保持原发布日期）' } else { '新文章' })
      $summary.Text = $item.title + "`r`n" + $item.date + '  ·  ' + ($item.categories -join ' / ') + '  ·  ' + $kind + "`r`n本地图片 " + $item.localImages + ' 处 / 去重后 ' + $item.copiedFiles + ' 个文件；外链图片 ' + $item.externalImages + ' 处'
      $state.Text = '检查通过：本地图片将随文章一起上传'
      $publish.Text = $(if ($item.exists) { '更新并发布' } else { '发布文章' }); $publish.Enabled = $true
    } else {
      $state.Text = '已成功推送；GitHub Pages 正在更新网站'
      $logs.Text = $output + "`r`n" + $errorText; $publish.Enabled = $true
    }
  } catch {
    $state.Text = '操作未完成'; $logs.Text = $_.Exception.Message
    $choose.Enabled = $true; $inspect.Enabled = $true; $category.Enabled = $true
  }
})
$choose.Add_Click({
  $dialog = New-Object System.Windows.Forms.OpenFileDialog
  $dialog.Filter = 'Markdown 原稿 (*.md)|*.md'; $dialog.Title = '选择在 Typora 中保存好的文章'
  if ($dialog.ShowDialog() -eq 'OK') { $script:selectedPath = $dialog.FileName; $fileBox.Text = $script:selectedPath; Begin-Task '--inspect' }
  $dialog.Dispose()
})
$inspect.Add_Click({ Begin-Task '--inspect' })
$category.Add_SelectedIndexChanged({ if ($script:selectedPath -and -not $script:task) { Begin-Task '--inspect' } })
$openOriginal.Add_Click({ if ($script:selectedPath) { Start-Process -FilePath $script:selectedPath } })
$visit.Add_Click({ Start-Process 'https://kanosa0101.github.io/' })
$publish.Add_Click({
  if (-not $script:inspection) { return }
  if ($script:inspection.exists) {
    if ([System.Windows.Forms.MessageBox]::Show('将更新博客中的同名文章：' + $script:inspection.filename + "`r`n是否继续？", '更新文章', 'YesNo', 'Question') -ne 'Yes') { return }
  }
  Begin-Task '--publish'
})
$form.Add_FormClosing({
  if ($script:task -and $script:task.Mode -eq '--publish') { $_.Cancel = $true; [System.Windows.Forms.MessageBox]::Show('文章正在发布，请等待完成后再关闭。') | Out-Null }
})
if ($SmokeTest) {
  if ($ArticlePath) {
    $timer.Start(); Begin-Task '--inspect'; $deadline = [DateTime]::Now.AddSeconds(20)
    while ($script:task -and [DateTime]::Now -lt $deadline) { [System.Windows.Forms.Application]::DoEvents(); Start-Sleep -Milliseconds 50 }
    $timer.Stop()
    if (-not $script:inspection) { throw ('界面检查未通过：' + $logs.Text) }
  }
  [pscustomobject]@{ Title = $form.Text; Controls = $form.Controls.Count; FilePicker = $choose.Text; Publish = $publish.Text; Inspection = $script:inspection } | ConvertTo-Json -Compress
  if ($RenderPath) {
    $form.ShowInTaskbar = $false; $form.StartPosition = 'Manual'; $form.Location = New-Object System.Drawing.Point(-3000, -3000); $form.Show(); [System.Windows.Forms.Application]::DoEvents(); $form.PerformLayout()
    $bitmap = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
    $form.DrawToBitmap($bitmap, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
    $bitmap.Save($RenderPath); $bitmap.Dispose()
  }
  $form.Dispose(); exit 0
}
$timer.Start()
$form.Add_Shown({ if ($script:selectedPath) { Begin-Task '--inspect' } })
[System.Windows.Forms.Application]::Run($form)
$timer.Dispose(); $form.Dispose()
