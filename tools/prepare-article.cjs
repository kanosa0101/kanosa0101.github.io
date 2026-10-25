const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const { fileURLToPath } = require('node:url');
const yaml = require('js-yaml');

const hash = value => crypto.createHash('sha256').update(value).digest('hex');
const dateText = value => value instanceof Date ? value.toISOString().slice(0, 19).replace('T', ' ') : String(value || '');
const localDate = () => new Intl.DateTimeFormat('sv-SE', { timeZone: 'Asia/Shanghai', dateStyle: 'short', timeStyle: 'medium' }).format(new Date());
function parseDocument(text) {
  const match = text.match(/^---\n([\s\S]*?)\n---(?:\n|$)([\s\S]*)$/);
  return match ? { meta: yaml.load(match[1]) || {}, body: match[2] } : { meta: {}, body: text };
}

function imageType(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) return 'png';
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'jpg';
  if (/^GIF8[79]a/.test(bytes.subarray(0, 6).toString())) return 'gif';
  if (bytes.subarray(0, 4).toString() === 'RIFF' && bytes.subarray(8, 12).toString() === 'WEBP') return 'webp';
  if (bytes.subarray(4, 8).toString() === 'ftyp' && /avif|avis/.test(bytes.subarray(8, 32).toString())) return 'avif';
  if (bytes.subarray(0, 2).toString() === 'BM') return 'bmp';
  if (/<svg(?:\s|>)/i.test(bytes.subarray(0, 4096).toString('utf8'))) return 'svg';
  throw new Error('图片格式无法识别，请使用 PNG、JPEG、GIF、WebP、AVIF、BMP 或 SVG 图片');
}

async function prepareArticle({ root, input, category, expectedHash }) {
  input = path.resolve(input);
  if (path.extname(input).toLowerCase() !== '.md') throw new Error('请选择 Markdown（.md）原稿');
  const original = await fs.readFile(input, 'utf8');
  const sourceHash = hash(original);
  if (expectedHash && expectedHash !== sourceHash) throw new Error('原稿已修改，请重新检查图片后再发布');
  const { meta: supplied, body } = parseDocument(original.replace(/^\uFEFF/, '').replaceAll('\r\n', '\n'));
  if (typeof supplied !== 'object' || Array.isArray(supplied)) throw new Error('文章开头的 YAML 信息格式不正确');
  const filename = path.basename(input, path.extname(input)).replace(/[<>:"/\\|?*\x00-\x1f]/g, '-').trim() + '.md';
  if (filename === '.md' || filename.startsWith('.')) throw new Error('文章文件名不正确，请重命名原稿');
  const target = path.join(root, 'source/_posts', filename);
  let previous = {}, targetHash = null;
  try { const old = await fs.readFile(target, 'utf8'); previous = parseDocument(old.replaceAll('\r\n', '\n')).meta; targetHash = hash(old); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  const meta = { ...previous, ...supplied };
  meta.title = String(supplied.title || body.match(/^#\s+(.+)$/m)?.[1] || previous.title || filename.slice(0, -3));
  meta.date = dateText(supplied.date || previous.date) || localDate();
  if (Number.isNaN(Date.parse(meta.date))) throw new Error('文章发布日期无效，请检查 date 字段');
  if (category) {
    meta.categories = Array.isArray(category)
      ? category
      : String(category).split(/[/,，、>]/).map(s => s.trim()).filter(Boolean);
  } else {
    const rawCat = supplied.categories || previous.categories || ['经验分享'];
    meta.categories = Array.isArray(rawCat)
      ? rawCat
      : String(rawCat).split(/[/,，、>]/).map(s => s.trim()).filter(Boolean);
  }
  const rootSetting = String(supplied['typora-root-url'] || '');
  if (/^\\\\/.test(rootSetting) || /^file:\/\/[^/]/i.test(rootSetting)) throw new Error('请先把网络共享图片复制到本机目录后发布');
  const imageRoot = rootSetting ? (/^file:/i.test(rootSetting) ? fileURLToPath(rootSetting) : path.resolve(path.dirname(input), rootSetting)) : null;
  delete meta['typora-root-url'];
  delete meta['typora-copy-images-to'];
  const assets = new Map();
  let localImages = 0, externalImages = 0;
  async function imageUrl(href) {
    if (/^(?:https?:|\/\/|data:image\/)/i.test(href)) { externalImages++; return href; }
    if (/^\\\\/.test(href) || (/^file:/i.test(href) && new URL(href).hostname && new URL(href).hostname !== 'localhost')) {
      throw new Error('请先把网络共享图片复制到本机后发布：' + href);
    }
    let local = href;
    if (/^file:/i.test(local)) local = fileURLToPath(local);
    else { try { local = decodeURIComponent(local); } catch { throw new Error('图片路径编码有误：' + href); } }
    const candidates = [];
    if (imageRoot && /^[\/]/.test(local)) candidates.push(path.join(imageRoot, local.replace(/^[\\/]+/, '')));
    if (/^[\/]/.test(local)) candidates.push(path.join(root, 'source', local.replace(/^[\\/]+/, '')));
    candidates.push(path.resolve(path.dirname(input), local));
    if (/[?#]/.test(local)) candidates.push(path.resolve(path.dirname(input), local.split(/[?#]/)[0]));
    let bytes, location;
    for (const candidate of candidates) {
      try { bytes = await fs.readFile(candidate); location = candidate; break; }
      catch (error) { if (!['ENOENT', 'ENOTDIR'].includes(error.code)) throw error; }
    }
    if (!bytes) throw new Error('找不到图片：' + href + '\n请在 Typora 中确认图片能显示，再重新检查。');
    const extension = imageType(bytes);
    if (extension === 'svg') {
      const svg = bytes.toString('utf8');
      const references = [...svg.matchAll(/(?:href=["']([^"']+)["']|url\(["']?([^\s)'";]+))/gi)].map(match => match[1] || match[2]);
      if (references.some(value => !/^(?:#|data:|https?:|\/\/)/i.test(value))) {
        throw new Error('SVG 图片还引用了本地资源，请在 Typora 中改用已嵌入资源的 SVG 或 PNG 后发布：' + href);
      }
    }
    const name = hash(bytes).slice(0, 32) + '.' + extension;
    const url = '/images/posts/' + name;
    if (!assets.has(url)) assets.set(url, { url, relative: 'source' + url, source: location, bytes });
    localImages++;
    return url;
  }

  const { fromMarkdown } = await import('mdast-util-from-markdown');
  const { parseFragment } = await import('parse5');
  const tree = fromMarkdown(body);
  const nodes = [];
  function walk(node) { nodes.push(node); for (const child of node.children || []) walk(child); }
  walk(tree);
  const usedReferences = new Set(nodes.filter(node => node.type === 'imageReference').map(node => node.identifier));
  const replacements = [];
  const quote = text => String(text).replaceAll('\\', '\\\\').replaceAll('"', '\\"');
  for (const node of nodes) {
    if (node.type === 'image' || (node.type === 'definition' && usedReferences.has(node.identifier))) {
      const url = await imageUrl(node.url);
      const title = node.title ? ` "${quote(node.title)}"` : '';
      const text = node.type === 'image' ? `![${(node.alt || '').replace(/[\\[\]]/g, '\\$&')}](${url}${title})` : `[${node.label || node.identifier}]: ${url}${title}`;
      replacements.push({ start: node.position.start.offset, end: node.position.end.offset, text });
    }
    if (node.type === 'html') {
      const fragment = parseFragment(node.value, { sourceCodeLocationInfo: true });
      const htmlNodes = [];
      const visit = node => { htmlNodes.push(node); for (const child of node.childNodes || []) visit(child); };
      visit(fragment);
      for (const element of htmlNodes.filter(element => element.nodeName === 'img')) {
        if (element.attrs.some(attr => attr.name === 'srcset')) throw new Error('请把 HTML 图片的 srcset 改为普通 src 链接后再发布');
        const src = element.attrs.find(attr => attr.name === 'src');
        if (!src) throw new Error('HTML 图片缺少 src 地址');
        const url = await imageUrl(src.value);
        const location = element.sourceCodeLocation.attrs.src;
        replacements.push({ start: node.position.start.offset + location.startOffset,
          end: node.position.start.offset + location.endOffset, text: `src="${url.replaceAll('&', '&amp;').replaceAll('"', '&quot;')}"` });
      }
    }
  }
  for (const key of ['cover', 'top_img', 'thumbnail', 'photos', 'image']) {
    if (typeof meta[key] === 'string' && meta[key]) meta[key] = await imageUrl(meta[key]);
    else if (Array.isArray(meta[key])) meta[key] = await Promise.all(meta[key].map(value => typeof value === 'string' ? imageUrl(value) : value));
  }
  let converted = body;
  for (const change of replacements.sort((a, b) => b.start - a.start)) converted = converted.slice(0, change.start) + change.text + converted.slice(change.end);
  if (path.resolve(target) === input) meta['typora-root-url'] = '..';
  const content = `---\n${yaml.dump(meta, { lineWidth: -1 })}---\n\n${converted.trim()}\n`;
  return { input, filename, target, meta, content, sourceHash, targetHash, exists: targetHash !== null,
    localImages, externalImages, assets: [...assets.values()] };
}

module.exports = { prepareArticle };
