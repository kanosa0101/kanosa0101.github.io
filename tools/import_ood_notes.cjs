const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const yaml = require('js-yaml');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = 'E:\\homework\\面向对象\\复习\\学习笔记';
const POSTS_DIR = path.join(ROOT, 'source', '_posts');
const IMAGES_DIR = path.join(ROOT, 'source', 'images', 'posts');

if (!fs.existsSync(POSTS_DIR)) fs.mkdirSync(POSTS_DIR, { recursive: true });
if (!fs.existsSync(IMAGES_DIR)) fs.mkdirSync(IMAGES_DIR, { recursive: true });

function hash(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function imageType(bytes) {
  if (bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) return 'png';
  if (bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return 'jpg';
  if (/^GIF8[79]a/.test(bytes.subarray(0, 6).toString())) return 'gif';
  if (bytes.subarray(0, 4).toString() === 'RIFF' && bytes.subarray(8, 12).toString() === 'WEBP') return 'webp';
  if (bytes.subarray(4, 8).toString() === 'ftyp' && /avif|avis/.test(bytes.subarray(8, 32).toString())) return 'avif';
  if (bytes.subarray(0, 2).toString() === 'BM') return 'bmp';
  if (/<svg(?:\s|>)/i.test(bytes.subarray(0, 4096).toString('utf8'))) return 'svg';
  return 'png';
}

const fileOrder = [
  '第1-4章.md',
  '第5章 用况图.md',
  '第6章 发现对象、定义对象.md',
  '第7章 定义对象的属性和操作.md',
  '第8章 定义对象间的关系.md',
  '第9章 UML中的几种其他图(a).md',
  '第9章 UML中的几种其他图(b).md',
  '第10章 模型规约.md',
  '第11章 问题域部分的设计.md',
  '第12章 人机交互部分的设计.md',
  '第13章 控制驱动部分的设计.md',
  '第14章 数据接口部分的设计.md'
];

async function run() {
  console.log('Starting Object-Oriented Analysis and Design notes import...');
  const existingFiles = fs.readdirSync(SOURCE_DIR).filter(f => f.endsWith('.md'));
  console.log(`Found ${existingFiles.length} files in source directory.`);

  for (const f of existingFiles) {
    if (!fileOrder.includes(f)) {
      fileOrder.push(f);
    }
  }

  let totalImagesProcessed = 0;
  const uniqueImageHashes = new Set();
  let totalLinksSanitized = 0;

  for (let i = 0; i < fileOrder.length; i++) {
    const fileName = fileOrder[i];
    const filePath = path.join(SOURCE_DIR, fileName);
    if (!fs.existsSync(filePath)) {
      console.warn(`File not found: ${filePath}, skipping`);
      continue;
    }

    let raw = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '').replaceAll('\r\n', '\n');

    // Extract title from H1 or filename
    const h1Match = raw.match(/^#\s+(.+)$/m);
    const title = h1Match ? h1Match[1].trim() : fileName.replace('.md', '');

    // Extract description from first substantial paragraph
    let desc = '';
    const lines = raw.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed.startsWith('#') || trimmed.startsWith('!') || trimmed.startsWith('|') || trimmed.startsWith('---')) continue;
      if (/^\*.*(?:课件|页码|页脚|页).*?\*$/.test(trimmed)) continue;
      if (/^(?:\d+(?:\.\d+)*\s*)?[*·_~`\s]*(?:课件|页码|页脚|对应).*$/i.test(trimmed)) continue;
      if (trimmed.length <= 15) continue;
      desc = trimmed.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*_`]/g, '').slice(0, 180);
      break;
    }

    // Minutes offset: 10:01 to 10:12
    const minute = String(i + 1).padStart(2, '0');
    const dateStr = `2026-11-24 10:${minute}:00`;

    // 1. Process images: ![alt](url) or ![alt](<url>)
    const imgRegex = /!\[([^\]]*)\]\(<?([^)>]+)>?\)/g;
    raw = raw.replace(imgRegex, (match, alt, imgUrl) => {
      let cleanUrl = imgUrl.trim();
      try { cleanUrl = decodeURIComponent(cleanUrl); } catch {}
      const fullImgPath = path.resolve(path.dirname(filePath), cleanUrl);
      if (!fs.existsSync(fullImgPath)) {
        throw new Error(`Image not found: ${imgUrl} in ${fileName} (tried ${fullImgPath})`);
      }
      const bytes = fs.readFileSync(fullImgPath);
      const ext = imageType(bytes);
      const name = hash(bytes).slice(0, 32) + '.' + ext;
      const destPath = path.join(IMAGES_DIR, name);
      if (!fs.existsSync(destPath)) {
        fs.writeFileSync(destPath, bytes);
      }
      uniqueImageHashes.add(name);
      totalImagesProcessed++;
      return `![${alt}](/images/posts/${name})`;
    });

    // 2. Sanitize local non-image links: [text](url) or [text](<url>)
    const linkRegex = /(?<!!)\[([^\]]+)\]\(<?([^)>]+)>?\)/g;
    raw = raw.replace(linkRegex, (match, text, targetUrl) => {
      const trimmedUrl = targetUrl.trim();
      // Keep anchors and web links
      if (trimmedUrl.startsWith('#') || trimmedUrl.startsWith('http://') || trimmedUrl.startsWith('https://')) {
        return match;
      }
      totalLinksSanitized++;
      return `**${text}**`;
    });

    // 3. Prepare Front-matter
    const frontMatterObj = {
      title,
      date: dateStr,
      categories: [
        '面向对象分析与设计'
      ],
      tags: [
        '面向对象分析与设计',
        '课程笔记'
      ],
      description: desc,
      mathjax: true
    };

    const frontMatterStr = `---\n${yaml.dump(frontMatterObj, { lineWidth: -1 })}---\n\n`;

    const finalContent = frontMatterStr + raw.trim() + '\n';
    const destPostPath = path.join(POSTS_DIR, fileName);
    fs.writeFileSync(destPostPath, finalContent, 'utf8');
    console.log(`[${i + 1}/${fileOrder.length}] Processed: ${fileName} -> ${title}`);
  }

  console.log('\n--- Summary ---');
  console.log(`Imported articles: ${fileOrder.length}`);
  console.log(`Images processed: ${totalImagesProcessed}`);
  console.log(`Unique image files created: ${uniqueImageHashes.size}`);
  console.log(`Local links sanitized: ${totalLinksSanitized}`);
}

run().catch(err => {
  console.error('Error during import:', err);
  process.exit(1);
});
