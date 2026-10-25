const http = require('node:http');
const fs = require('node:fs/promises');
const fsSync = require('node:fs');
const path = require('node:path');
const { spawn, execFile } = require('node:child_process');
const yaml = require('js-yaml');
const { prepareArticle } = require('./prepare-article.cjs');
const { publishArticle } = require('./publish-site.cjs');

const root = path.resolve(__dirname, '..');
const edgePath = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

// Helper to scan all existing categories across the blog
async function getBlogCategories() {
  const postsDir = path.join(root, 'source/_posts');
  const categories = new Set(['笔记 / CS336', '经验分享', '论文阅读', '技术教程']);
  try {
    const files = await fs.readdir(postsDir);
    for (const file of files) {
      if (file.endsWith('.md')) {
        try {
          const content = await fs.readFile(path.join(postsDir, file), 'utf8');
          const match = content.match(/^---\n([\s\S]*?)\n---/);
          if (match) {
            const meta = yaml.load(match[1]) || {};
            if (meta.categories) {
              if (Array.isArray(meta.categories)) {
                categories.add(meta.categories.join(' / '));
              } else {
                categories.add(String(meta.categories));
              }
            }
          }
        } catch (_) {}
      }
    }
  } catch (_) {}
  return Array.from(categories);
}

// Open native Windows file dialog (using Base64 for 100% reliable Unicode/Chinese path handling)
function openFileDialog() {
  return new Promise((resolve) => {
    const psScript = `
      Add-Type -AssemblyName System.Windows.Forms
      $d = New-Object System.Windows.Forms.OpenFileDialog
      $d.Filter = 'Markdown 文章 (*.md)|*.md'
      $d.Title = '选择要发布的 Markdown 文章原稿'
      $d.InitialDirectory = 'C:\\Users\\14376\\Desktop\\笔记'
      if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($d.FileName)
        [Console]::WriteLine([Convert]::ToBase64String($bytes))
      }
    `;
    execFile('powershell', ['-NoProfile', '-STA', '-Command', psScript], { windowsHide: true }, (err, stdout) => {
      if (err || !stdout.trim()) {
        resolve(null);
      } else {
        try {
          const decoded = Buffer.from(stdout.trim(), 'base64').toString('utf8');
          resolve(decoded);
        } catch (_) {
          resolve(null);
        }
      }
    });
  });
}

// Open file with default app
function openFile(filePath) {
  if (filePath && fsSync.existsSync(filePath)) {
    execFile('cmd.exe', ['/c', 'start', '""', filePath], {
      windowsHide: true
    });
  }
}

// Open URL with default browser
function openUrl(url) {
  execFile('cmd.exe', ['/c', 'start', '""', url], {
    windowsHide: true
  });
}

let initialArticle = null;
if (process.argv[2] && process.argv[2].trim()) {
  try {
    const candidate = path.resolve(process.argv[2].trim());
    if (candidate.toLowerCase().endsWith('.md') && fsSync.existsSync(candidate)) {
      initialArticle = candidate;
    }
  } catch (_) {}
}
let lastHeartbeat = Date.now();
let isPublishing = false;

const HTML_CONTENT = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>kanosa · 文章发布与更新中心</title>
  <style>
    :root {
      --primary: #059669;
      --primary-hover: #047857;
      --primary-light: #ecfdf5;
      --primary-border: #a7f3d0;
      --slate-50: #f8fafc;
      --slate-100: #f1f5f9;
      --slate-200: #e2e8f0;
      --slate-300: #cbd5e1;
      --slate-400: #94a3b8;
      --slate-600: #475569;
      --slate-700: #334155;
      --slate-800: #1e293b;
      --slate-900: #0f172a;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.07);
      --shadow-lg: 0 10px 25px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.05);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
      color: var(--slate-800);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      user-select: none;
      line-height: 1.5;
    }
    /* App Header */
    header {
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--slate-200);
      padding: 14px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #10b981 0%, #059669 100%);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-size: 20px;
      box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3);
    }
    .brand-text h1 {
      font-size: 16px;
      font-weight: 700;
      color: var(--slate-900);
      letter-spacing: -0.3px;
    }
    .brand-text p {
      font-size: 12px;
      color: var(--slate-400);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .btn-header {
      background: white;
      border: 1px solid var(--slate-200);
      color: var(--slate-600);
      padding: 7px 14px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn-header:hover {
      background: var(--slate-100);
      border-color: var(--slate-300);
      color: var(--slate-900);
    }

    /* Container */
    main {
      flex: 1;
      max-width: 860px;
      width: 100%;
      margin: 24px auto;
      padding: 0 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    /* Card */
    .card {
      background: white;
      border-radius: var(--radius-md);
      border: 1px solid var(--slate-200);
      box-shadow: var(--shadow-sm);
      padding: 22px;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    .card:hover {
      box-shadow: var(--shadow-md);
    }

    /* File Drop Zone */
    .file-zone {
      border: 2px dashed var(--slate-300);
      border-radius: var(--radius-md);
      background: white;
      padding: 26px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .file-zone:hover, .file-zone.dragover {
      border-color: var(--primary);
      background: var(--primary-light);
    }
    .file-zone-content {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
    }
    .file-zone-icon {
      font-size: 36px;
      color: var(--primary);
      margin-bottom: 2px;
    }
    .file-path-display {
      font-size: 13px;
      font-family: Consolas, monospace;
      color: var(--slate-600);
      background: var(--slate-100);
      padding: 6px 14px;
      border-radius: 6px;
      max-width: 90%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .btn-browse {
      background: var(--slate-900);
      color: white;
      border: none;
      padding: 8px 18px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      transition: opacity 0.15s;
    }
    .btn-browse:hover { opacity: 0.9; }

    /* Category Section (Key Fix) */
    .category-section {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .section-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--slate-900);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .category-preview-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      background: var(--slate-50);
      border: 1px solid var(--slate-200);
      padding: 10px 14px;
      border-radius: var(--radius-sm);
    }
    .preview-label {
      font-size: 12px;
      color: var(--slate-500);
      font-weight: 600;
    }
    .category-crumbs {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .crumb-pill {
      background: #e0e7ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
    }
    .crumb-divider {
      color: var(--slate-400);
      font-size: 12px;
      font-weight: 700;
    }

    .category-pills {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .cat-btn {
      background: var(--slate-100);
      border: 1px solid var(--slate-200);
      color: var(--slate-700);
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s;
    }
    .cat-btn:hover {
      background: var(--slate-200);
      border-color: var(--slate-300);
    }
    .cat-btn.active {
      background: var(--primary);
      border-color: var(--primary);
      color: white;
    }

    .custom-cat-input-wrap {
      display: flex;
      gap: 10px;
      align-items: center;
    }
    .custom-cat-input {
      flex: 1;
      padding: 8px 12px;
      border: 1px solid var(--slate-200);
      border-radius: var(--radius-sm);
      font-size: 13px;
      color: var(--slate-800);
      outline: none;
      transition: border-color 0.2s;
    }
    .custom-cat-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.1);
    }

    /* Summary Details Grid */
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 14px;
    }
    .summary-box {
      background: var(--slate-50);
      border: 1px solid var(--slate-200);
      padding: 12px 14px;
      border-radius: var(--radius-sm);
    }
    .summary-box .k {
      font-size: 11px;
      color: var(--slate-400);
      text-transform: uppercase;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .summary-box .v {
      font-size: 13px;
      font-weight: 600;
      color: var(--slate-800);
    }
    .badge-status {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
    }
    .badge-update { background: #dbeafe; color: #1e40af; }
    .badge-new { background: #dcfce7; color: #166534; }

    /* Action & Progress Bar */
    .action-panel {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .btn-publish {
      background: linear-gradient(135deg, #059669 0%, #047857 100%);
      color: white;
      border: none;
      padding: 14px 28px;
      border-radius: var(--radius-md);
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      box-shadow: 0 4px 14px rgba(5, 150, 105, 0.35);
      transition: all 0.2s ease;
    }
    .btn-publish:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 18px rgba(5, 150, 105, 0.45);
    }
    .btn-publish:disabled {
      background: var(--slate-300);
      box-shadow: none;
      cursor: not-allowed;
      opacity: 0.7;
    }

    /* Stepper */
    .stepper {
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: relative;
      margin: 8px 0;
    }
    .step-item {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--slate-400);
      z-index: 2;
    }
    .step-dot {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: white;
      border: 2px solid var(--slate-300);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 10px;
      font-weight: 700;
      color: var(--slate-400);
      transition: all 0.2s;
    }
    .step-item.active .step-dot {
      border-color: var(--primary);
      background: var(--primary);
      color: white;
    }
    .step-item.active { color: var(--primary); font-weight: 600; }
    .stepper-line {
      position: absolute;
      top: 11px;
      left: 10%;
      right: 10%;
      height: 2px;
      background: var(--slate-200);
      z-index: 1;
    }

    /* Terminal Logs Drawer */
    .logs-drawer {
      border: 1px solid var(--slate-200);
      border-radius: var(--radius-md);
      overflow: hidden;
      background: var(--slate-900);
    }
    .logs-header {
      background: #1e293b;
      padding: 10px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      cursor: pointer;
      color: var(--slate-300);
      font-size: 12px;
      font-weight: 600;
    }
    .logs-body {
      padding: 14px;
      font-family: Consolas, monospace;
      font-size: 11px;
      color: #94a3b8;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-all;
      line-height: 1.6;
    }
    .log-success { color: #34d399; font-weight: 700; }
    .log-stage { color: #38bdf8; font-weight: 600; }
    .log-error { color: #f87171; font-weight: 700; }

    /* Toast */
    .toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: var(--slate-900);
      color: white;
      padding: 12px 20px;
      border-radius: var(--radius-sm);
      font-size: 13px;
      font-weight: 500;
      box-shadow: var(--shadow-lg);
      display: flex;
      align-items: center;
      gap: 10px;
      opacity: 0;
      transform: translateY(10px);
      transition: all 0.25s ease;
      pointer-events: none;
      z-index: 100;
    }
    .toast.show {
      opacity: 1;
      transform: translateY(0);
    }
    .toast.success { background: #065f46; border: 1px solid #059669; }
    .toast.error { background: #991b1b; border: 1px solid #dc2626; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">🦋</div>
      <div class="brand-text">
        <h1>kanosa · 博客发布中心</h1>
        <p>Typora / Markdown 文章一键同步与发布</p>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn-header" id="btnOpenOriginal" title="在默认编辑器（如 Typora）中打开原稿">📖 打开原稿</button>
      <button class="btn-header" id="btnOpenBlog" title="在浏览器打开你的博客">🌐 查看线上博客 ↗</button>
    </div>
  </header>

  <main>
    <!-- File Zone -->
    <div class="card file-zone" id="dropZone">
      <div class="file-zone-content">
        <div class="file-zone-icon">📄</div>
        <h3 id="fileZoneTitle" style="font-size: 15px; font-weight: 700;">点击浏览或将 .md 文件拖拽到此处</h3>
        <p id="fileZoneSub" style="font-size: 12px; color: var(--slate-400);">支持任意文件夹下的 Typora / Markdown 原稿，自动整理本地图片</p>
        <div class="file-path-display" id="filePathDisplay" style="display:none;"></div>
        <div style="display: flex; align-items: center; gap: 10px; margin-top: 4px;">
          <button class="btn-browse" id="btnBrowse" type="button">选择文章…</button>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; width: 100%; max-width: 640px; margin-top: 10px;" onclick="event.stopPropagation()">
          <input type="text" class="custom-cat-input" id="manualPathInput" placeholder="或直接在此粘贴完整路径（支持中英文路径，按回车解析）" style="font-family: Consolas, monospace; font-size: 12px;" />
          <button class="btn-header" id="btnLoadManualPath" type="button" style="white-space: nowrap; padding: 7px 14px;">解析路径</button>
        </div>
      </div>
    </div>

    <!-- Category Section -->
    <div class="card category-section" id="categoryCard">
      <div class="section-title">
        <span>🏷️ 文章层级分类</span>
        <span style="font-size: 12px; font-weight: 400; color: var(--slate-400);">（决定在博客中的所属主分类与子分类）</span>
      </div>

      <!-- Live Category Breadcrumb Preview -->
      <div class="category-preview-bar">
        <span class="preview-label">最终分类层级：</span>
        <div class="category-crumbs" id="categoryCrumbs">
          <span class="crumb-pill">笔记</span>
          <span class="crumb-divider">❯</span>
          <span class="crumb-pill">CS336</span>
        </div>
      </div>

      <!-- Quick category pills from blog -->
      <div style="font-size: 12px; color: var(--slate-500); margin-top: 4px;">点选已有分类：</div>
      <div class="category-pills" id="categoryPills">
        <!-- Rendered by JS -->
      </div>

      <!-- Custom category input -->
      <div class="custom-cat-input-wrap">
        <input type="text" class="custom-cat-input" id="customCatInput" placeholder="自定义输入多级分类（例如：笔记 / CS336 或 课程笔记 / Stanford）" />
        <button class="btn-header" id="btnApplyCustomCat">应用分类</button>
      </div>
    </div>

    <!-- Article Summary Card -->
    <div class="card" id="summaryCard" style="display: none;">
      <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 14px;">
        <div>
          <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
            <span class="badge-status badge-new" id="badgeExists">新文章</span>
            <span style="font-size: 12px; color: var(--slate-400);" id="articleDate">2026-10-25</span>
          </div>
          <h2 id="articleTitle" style="font-size: 17px; font-weight: 700; color: var(--slate-900); line-height: 1.4;"></h2>
        </div>
      </div>

      <div class="summary-grid">
        <div class="summary-box">
          <div class="k">目标文件名</div>
          <div class="v" id="metaFilename">-</div>
        </div>
        <div class="summary-box">
          <div class="k">本地图片资源</div>
          <div class="v" id="metaLocalImages">0 处</div>
        </div>
        <div class="summary-box">
          <div class="k">外链图片</div>
          <div class="v" id="metaExternalImages">0 处</div>
        </div>
        <div class="summary-box">
          <div class="k">公式引擎支持</div>
          <div class="v" style="color: #059669;" id="metaMath">MathJax 就绪</div>
        </div>
      </div>
    </div>

    <!-- Publish Action & Stepper -->
    <div class="card action-panel">
      <div class="stepper">
        <div class="stepper-line"></div>
        <div class="step-item active" id="step1">
          <div class="step-dot">1</div>
          <span>原稿检查</span>
        </div>
        <div class="step-item" id="step2">
          <div class="step-dot">2</div>
          <span>页面生成</span>
        </div>
        <div class="step-item" id="step3">
          <div class="step-dot">3</div>
          <span>站点校验</span>
        </div>
        <div class="step-item" id="step4">
          <div class="step-dot">4</div>
          <span>推送上线</span>
        </div>
      </div>

      <button class="btn-publish" id="btnPublish" disabled>
        <span id="publishIcon">🚀</span>
        <span id="publishText">请先选择 Markdown 文章</span>
      </button>
    </div>

    <!-- Terminal Drawer -->
    <div class="logs-drawer">
      <div class="logs-header" id="logsToggle">
        <span>📋 构建与推送终端日志</span>
        <span id="logsToggleText" style="font-weight: 400; opacity: 0.8;">展开日志 ▾</span>
      </div>
      <div class="logs-body" id="logsBody" style="display: none;">等待操作开始...</div>
    </div>
  </main>

  <div class="toast" id="toast">提示信息</div>

  <script>
    let currentPath = null;
    let inspectionData = null;
    let blogCategories = [];
    let selectedCategory = '';

    const el = id => document.getElementById(id);

    function showToast(msg, type = 'info') {
      const toast = el('toast');
      toast.textContent = msg;
      toast.className = 'toast show ' + type;
      setTimeout(() => toast.className = 'toast', 3500);
    }

    // Heartbeat to let Node server know window is open
    setInterval(() => {
      fetch('/api/heartbeat', { method: 'POST' }).catch(() => {});
    }, 1500);

    // Initial setup
    async function init() {
      try {
        const res = await fetch('/api/init');
        const data = await res.json();
        blogCategories = data.categories || [];
        renderCategoryPills();

        if (data.initialPath) {
          inspectFile(data.initialPath);
        }
      } catch (err) {
        showToast('初始化异常：' + err.message, 'error');
      }
    }

    function renderCategoryPills() {
      const container = el('categoryPills');
      container.innerHTML = '';

      // Default: "沿用原稿"
      const defaultBtn = document.createElement('button');
      defaultBtn.className = 'cat-btn' + (!selectedCategory ? ' active' : '');
      defaultBtn.textContent = '📄 沿用原稿';
      defaultBtn.onclick = () => {
        selectedCategory = '';
        updateCategorySelection();
      };
      container.appendChild(defaultBtn);

      blogCategories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'cat-btn' + (selectedCategory === cat ? ' active' : '');
        btn.textContent = cat;
        btn.onclick = () => {
          selectedCategory = cat;
          updateCategorySelection();
        };
        container.appendChild(btn);
      });
    }

    function updateCategorySelection() {
      renderCategoryPills();
      updateBreadcrumb();
      if (currentPath) {
        inspectFile(currentPath, false);
      }
    }

    function updateBreadcrumb() {
      const crumbsBox = el('categoryCrumbs');
      let catList = [];
      if (selectedCategory) {
        catList = selectedCategory.split(/[/,，、>]/).map(s => s.trim()).filter(Boolean);
      } else if (inspectionData && inspectionData.categories) {
        catList = inspectionData.categories;
      } else {
        catList = ['未指定 (默认: 经验分享)'];
      }

      crumbsBox.innerHTML = '';
      catList.forEach((c, idx) => {
        const pill = document.createElement('span');
        pill.className = 'crumb-pill';
        pill.textContent = c;
        crumbsBox.appendChild(pill);
        if (idx < catList.length - 1) {
          const div = document.createElement('span');
          div.className = 'crumb-divider';
          div.textContent = '❯';
          crumbsBox.appendChild(div);
        }
      });
    }

    // Inspect file
    async function inspectFile(filePath, showLogs = true) {
      currentPath = filePath;
      if (el('manualPathInput')) el('manualPathInput').value = filePath;
      el('filePathDisplay').style.display = 'block';
      el('filePathDisplay').textContent = filePath;
      el('fileZoneTitle').textContent = filePath.split(/[\\\\/]/).pop();
      el('fileZoneSub').textContent = '点击可重新选择文件';

      if (showLogs) {
        appendLog('\\n[检查] 正在解析原稿与本地资源: ' + filePath);
      }

      try {
        const res = await fetch('/api/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: filePath, category: selectedCategory || undefined })
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        inspectionData = data;
        el('summaryCard').style.display = 'block';
        el('articleTitle').textContent = data.title;
        el('articleDate').textContent = data.date;
        el('metaFilename').textContent = data.filename;
        el('metaLocalImages').textContent = data.localImages + ' 处 (' + data.copiedFiles + ' 个文件)';
        el('metaExternalImages').textContent = data.externalImages + ' 处';
        el('badgeExists').textContent = data.exists ? '更新已有文章' : '新文章';
        el('badgeExists').className = 'badge-status ' + (data.exists ? 'badge-update' : 'badge-new');

        updateBreadcrumb();

        el('btnPublish').disabled = false;
        el('publishIcon').textContent = data.exists ? '🔄' : '🚀';
        el('publishText').textContent = data.exists ? '更新并发布到博客' : '一键发布文章到博客';
        updateStep(1);

        if (showLogs) {
          appendLog('[通过] 文章检查无误，就绪可发布！');
        }
      } catch (err) {
        showToast('文章解析失败：' + err.message, 'error');
        appendLog('[错误] ' + err.message);
        el('btnPublish').disabled = true;
      }
    }

    // Stepper
    function updateStep(step) {
      for (let i = 1; i <= 4; i++) {
        const item = el('step' + i);
        if (i <= step) item.classList.add('active');
        else item.classList.remove('active');
      }
    }

    // Logs
    function appendLog(text) {
      const body = el('logsBody');
      body.textContent += '\\n' + text;
      body.scrollTop = body.scrollHeight;
    }

    // Manual path loading
    function handleManualPath() {
      const val = el('manualPathInput').value.trim().replace(/^["']|["']$/g, '');
      if (!val) {
        showToast('请输入有效的 .md 文件路径', 'info');
        return;
      }
      inspectFile(val);
    }

    el('btnLoadManualPath').onclick = (e) => {
      e.stopPropagation();
      handleManualPath();
    };

    el('manualPathInput').onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        handleManualPath();
      }
    };

    // Paste event support (Ctrl+V)
    window.addEventListener('paste', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const text = (e.clipboardData || window.clipboardData)?.getData('text');
      if (text && text.trim()) {
        const clean = text.trim().replace(/^["']|["']$/g, '');
        if (clean.toLowerCase().endsWith('.md')) {
          el('manualPathInput').value = clean;
          inspectFile(clean);
          showToast('已从剪贴板载入文章路径', 'success');
        }
      }
    });

    // Browse Button
    el('btnBrowse').onclick = async (e) => {
      e.stopPropagation();
      const res = await fetch('/api/browse', { method: 'POST' });
      const data = await res.json();
      if (data.path) {
        inspectFile(data.path);
      }
    };

    el('dropZone').onclick = () => el('btnBrowse').click();

    // Drag and Drop
    el('dropZone').ondragover = e => { e.preventDefault(); el('dropZone').classList.add('dragover'); };
    el('dropZone').ondragleave = () => el('dropZone').classList.remove('dragover');
    el('dropZone').ondrop = e => {
      e.preventDefault();
      el('dropZone').classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.path) {
          inspectFile(file.path);
        } else {
          showToast('已检测到文件，请点击“选择文章”确认本机完整路径', 'info');
          el('btnBrowse').click();
        }
      }
    };

    // Custom Category Input
    el('btnApplyCustomCat').onclick = () => {
      const val = el('customCatInput').value.trim();
      if (val) {
        selectedCategory = val;
        updateCategorySelection();
        showToast('已设定分类为：' + val, 'success');
      }
    };

    // Publish Action
    el('btnPublish').onclick = async () => {
      if (!currentPath || !inspectionData) return;
      if (inspectionData.exists) {
        if (!confirm('博客已存在同名文章：' + inspectionData.filename + '\\n\\n确认要更新覆盖线上版本吗？')) {
          return;
        }
      }

      el('btnPublish').disabled = true;
      el('publishText').textContent = '正在发布中，请稍候...';
      el('logsBody').style.display = 'block';
      el('logsToggleText').textContent = '收起日志 ▴';
      appendLog('\\n================ 开始发布任务 ================');

      try {
        updateStep(2);
        const res = await fetch('/api/publish', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            path: currentPath,
            category: selectedCategory || undefined,
            expectedHash: inspectionData.sourceHash
          })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const text = decoder.decode(value);
          appendLog(text);
          if (text.includes('构建网页')) updateStep(2);
          if (text.includes('站点检查通过')) updateStep(3);
          if (text.includes('推送') || text.includes('发布生成的网页')) updateStep(4);
        }

        updateStep(4);
        showToast('🎉 文章已成功推送到博客！GitHub Pages 正在构建上线', 'success');
        el('publishText').textContent = '已成功推送！可再次点击更新';
        el('btnPublish').disabled = false;
      } catch (err) {
        showToast('发布异常：' + err.message, 'error');
        appendLog('\\n[发布失败] ' + err.message);
        el('publishText').textContent = '重试发布';
        el('btnPublish').disabled = false;
      }
    };

    // Header buttons
    el('btnOpenOriginal').onclick = () => {
      if (!currentPath) { showToast('请先选择一篇文章', 'info'); return; }
      fetch('/api/open-original', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: currentPath })
      });
    };

    el('btnOpenBlog').onclick = () => {
      fetch('/api/open-blog', { method: 'POST' });
    };

    el('logsToggle').onclick = () => {
      const body = el('logsBody');
      const isHidden = body.style.display === 'none';
      body.style.display = isHidden ? 'block' : 'none';
      el('logsToggleText').textContent = isHidden ? '收起日志 ▴' : '展开日志 ▾';
    };

    window.onload = init;
  </script>
</body>
</html>
`;

// HTTP Server
const server = http.createServer(async (req, res) => {
  lastHeartbeat = Date.now();

  if (req.method === 'GET' && req.url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    return res.end(HTML_CONTENT);
  }

  if (req.method === 'GET' && req.url === '/api/init') {
    const categories = await getBlogCategories();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({
      initialPath: initialArticle,
      categories
    }));
  }

  if (req.method === 'POST' && req.url === '/api/heartbeat') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ok: true }));
  }

  if (req.method === 'POST' && req.url === '/api/browse') {
    const selected = await openFileDialog();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ path: selected }));
  }

  if (req.method === 'POST' && req.url === '/api/inspect') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { path: filePath, category } = JSON.parse(body);
        const article = await prepareArticle({ root, input: filePath, category });
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          title: article.meta.title,
          date: article.meta.date,
          categories: article.meta.categories,
          filename: article.filename,
          exists: article.exists,
          sourceHash: article.sourceHash,
          localImages: article.localImages,
          copiedFiles: article.assets.length,
          externalImages: article.externalImages
        }));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/api/publish') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const { path: filePath, category, expectedHash } = JSON.parse(body);
        isPublishing = true;
        res.writeHead(200, {
          'Content-Type': 'text/plain; charset=utf-8',
          'Transfer-Encoding': 'chunked',
          'Cache-Control': 'no-cache'
        });

        await publishArticle({
          root,
          input: filePath,
          category,
          expectedHash,
          onProgress: progress => {
            if (progress.stage) res.write('\\n[进度] ' + progress.stage + '\\n');
            if (progress.log) res.write(progress.log);
          }
        });

        res.write('\\n\\n[成功] 文章发布与推送已完成！\\n');
        res.end();
      } catch (err) {
        res.write('\\n[错误] 发布失败: ' + err.message + '\\n');
        res.end();
      } finally {
        isPublishing = false;
      }
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/api/open-original') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const { path: filePath } = JSON.parse(body);
        openFile(filePath);
      } catch (_) {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    });
    return;
  }

  if (req.method === 'POST' && req.url === '/api/open-blog') {
    openUrl('https://kanosa0101.github.io/');
    res.writeHead(200, { 'Content-Type': 'application/json' });
    return res.end(JSON.stringify({ ok: true }));
  }

  res.writeHead(404);
  res.end('Not found');
});

server.listen(0, '127.0.0.1', () => {
  const port = server.address().port;
  console.log('Publisher server running on port:', port);

  // Launch Edge App window (Zero terminal, standalone window!)
  const edgeApp = spawn(edgePath, [
    `--app=http://127.0.0.1:${port}`,
    '--window-size=920,740',
    `--user-data-dir=${path.join(root, '.blog-publish/profile')}`,
    '--app-id=kanosa-publisher'
  ], { detached: true, stdio: 'ignore' });
  edgeApp.unref();

  // Watch for disconnect when window is closed
  const checkTimer = setInterval(() => {
    // If no heartbeat for 5 seconds and not currently publishing, exit cleanly
    if (!isPublishing && Date.now() - lastHeartbeat > 5000) {
      clearInterval(checkTimer);
      server.close(() => process.exit(0));
    }
  }, 1500);
});
