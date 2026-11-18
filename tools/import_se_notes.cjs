const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const yaml = require('js-yaml');

const ROOT = path.resolve(__dirname, '..');
const SOURCE_DIR = 'E:\\homework\\软件工程\\笔记';
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
  return 'png'; // default fallback
}

// 19 files in logical sequence
const fileOrder = [
  '第01章 软件与软件工程.md',
  '第02章 软件过程结构.md',
  '第03章 敏捷与敏捷过程.md',
  '第05章 软件工程的人员方面.md',
  '第06章 理解需求.md',
  '第07-1章 需求建模（基于场景的方法）.md',
  '第07-2章 需求建模（基于类的方法）.md',
  '第07-3章 需求建模（行为和模式）.md',
  '第08章 设计概念.md',
  '第09章 体系结构设计.md',
  '第10章 构件级设计.md',
  '第15章 软件测试（构件级）.md',
  '第16章 软件测试（集成级）.md',
  '第17章 软件配置管理.md',
  '第19章 项目管理概念.md',
  '第20章 过程度量和项目度量.md',
  '第24章 软件项目估算.md',
  '第25章 项目进度安排.md',
  '第26章 风险管理.md'
];

async function run() {
  console.log('Starting Software Engineering notes import...');
  const existingFiles = fs.readdirSync(SOURCE_DIR).filter(f => f.endsWith('.md'));
  console.log(`Found ${existingFiles.length} files in source directory.`);

  let totalImagesProcessed = 0;
  const uniqueImageHashes = new Set();
  let totalLinksSanitized = 0;

  for (let i = 0; i < fileOrder.length; i++) {
    const fileName = fileOrder[i];
    const filePath = path.join(SOURCE_DIR, fileName);
    if (!fs.existsSync(filePath)) {
      throw new Error(`File not found: ${filePath}`);
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
      if (/^(?:\d+(?:\.\d+)*\s*)?[*·_~`\s]*(?:课件|页码|页脚).*$/i.test(trimmed)) continue;
      if (trimmed.length <= 15) continue;
      desc = trimmed.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[*_`]/g, '').slice(0, 180);
      break;
    }

    // Minutes offset: 10:01 to 10:19
    const minute = String(i + 1).padStart(2, '0');
    const dateStr = `2026-11-18 10:${minute}:00`;

    // 1. Process images
    // Replace markdown images: ![alt](url)
    const imgRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    raw = raw.replace(imgRegex, (match, alt, imgUrl) => {
      // Decode URL if needed
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

    // 2. Sanitize local non-image links:
    // Any [text](url) where url points to ../课件, ../作业, ../复习材料, or cross-chapter .md
    const linkRegex = /(?<!!)\[([^\]]+)\]\(([^)]+)\)/g;
    raw = raw.replace(linkRegex, (match, text, targetUrl) => {
      const trimmedUrl = targetUrl.trim();
      // Keep anchors and web links
      if (trimmedUrl.startsWith('#') || trimmedUrl.startsWith('http://') || trimmedUrl.startsWith('https://')) {
        return match;
      }
      // Out of bounds local files or cross-chapter links
      totalLinksSanitized++;
      return `**${text}**`;
    });

    // 3. Prepare Front-matter
    const frontMatterObj = {
      title,
      date: dateStr,
      categories: [
        '软件工程'
      ],
      tags: [
        '软件工程',
        '课程笔记'
      ],
      description: desc,
      mathjax: true
    };

    const frontMatterStr = `---\n${yaml.dump(frontMatterObj, { lineWidth: -1 })}---\n\n`;

    // Ensure we don't duplicate the front-matter or corrupt the file
    const finalContent = frontMatterStr + raw.trim() + '\n';
    const destPostPath = path.join(POSTS_DIR, fileName);
    fs.writeFileSync(destPostPath, finalContent, 'utf8');
    console.log(`[${i + 1}/19] Processed: ${fileName} -> ${title}`);
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
