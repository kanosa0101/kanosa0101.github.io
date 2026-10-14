const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../public');
const failures = [];
const postsRoot = path.resolve(__dirname, '../source/_posts');
for (const file of fs.readdirSync(postsRoot, { recursive: true }).filter(file => file.endsWith('.md'))) {
  const frontMatter = fs.readFileSync(path.join(postsRoot, file), 'utf8').split('---')[1] || '';
  if (!/^date:[ \t]*\S/m.test(frontMatter)) failures.push(`${file}: 缺少显式发布日期，重新部署后链接可能改变`);
}
const pages = fs.readdirSync(root, { recursive: true })
  .filter(file => file.endsWith('.html'));

for (const file of pages) {
  const html = fs.readFileSync(path.join(root, file), 'utf8');
  for (const [, attribute, url] of html.matchAll(/\b(href|src)=["']([^"']*)["']/g)) {
    if (/^(?:#|https?:|\/\/|data:|mailto:|tel:|javascript:|blob:)/i.test(url)) continue;
    if (!url && attribute === 'href') continue;
    if (/^(?:file:|[a-z]:[\\/])/i.test(url)) { failures.push(`${file}: 引用了本机文件 ${url}`); continue; }
    let pathname;
    try { pathname = decodeURIComponent(new URL(url, 'https://blog.local/' + file.split(path.sep).join('/')).pathname); }
    catch { failures.push(`${file}: 无效的资源地址 ${url}`); continue; }
    let target = path.resolve(root, '.' + pathname);
    if (target !== root && !target.startsWith(root + path.sep)) { failures.push(`${file}: 资源超出站点目录 ${url}`); continue; }
    if (fs.existsSync(target) && fs.statSync(target).isDirectory()) {
      target = path.join(target, 'index.html');
    }
    if (!fs.existsSync(target)) failures.push(`${file}: ${url} 不存在`);
    if (attribute === 'src' && (!url || target.endsWith('.html'))) {
      failures.push(`${file}: ${url} 将 HTML 作为资源加载`);
    }
  }
  for (const [, url] of html.matchAll(/url\((\/(?!\/)[^\s)'";]+)\)/g)) {
    if (!fs.existsSync(path.join(root, url))) failures.push(`${file}: 背景 ${url} 不存在`);
  }
}

const home = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const init = home.match(/L2Dwidget\.init\((.*?)\);/);
if (!init) failures.push('Live2D 初始化脚本缺失');
if (init) {
  const modelPath = JSON.parse(init[1]).model?.jsonPath;
  if (!modelPath || !fs.existsSync(path.join(root, modelPath))) {
    failures.push('Live2D 模型未配置或文件不存在');
  } else {
    const model = JSON.parse(fs.readFileSync(path.join(root, modelPath), 'utf8'));
    const assets = [model.model, model.physics, ...(model.textures || []),
      ...Object.values(model.motions || {}).flat().map(motion => motion.file)];
    for (const asset of assets.filter(Boolean)) {
      if (!fs.existsSync(path.resolve(root, path.dirname(modelPath).slice(1), asset))) {
        failures.push(`Live2D 关联资源不存在: ${asset}`);
      }
    }
  }
}
for (const [route, marker] of [['tags', 'tag-cloud-list text-center'], ['categories', 'category-lists'], ['link', 'flink']]) {
  const html = fs.readFileSync(path.join(root, route, 'index.html'), 'utf8');
  if (!html.includes(`class="${marker}"`)) failures.push(`${route}: 页面模板未生效`);
}
if (home.includes('https://github.com/xxxxxx')) failures.push('作者链接仍为占位地址');
assert.equal(failures.length, 0, failures.join('\n'));
console.log(`站点检查通过：${pages.length} 个页面，本地链接、资源、页面模板与 Live2D 模型完整。`);
