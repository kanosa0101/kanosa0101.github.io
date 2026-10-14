const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
const { prepareArticle } = require('./prepare-article.cjs');
const png = Buffer.from('89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000b49444154789c636000020000050001a5f645400000000049454e44ae426082', 'hex');

async function setup(t) {
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'typora-publish-'));
  t.after(() => fs.rm(folder, { recursive: true, force: true }));
  const root = path.join(folder, 'blog');
  const notes = path.join(folder, '我的笔记');
  await fs.mkdir(path.join(root, 'source/_posts'), { recursive: true });
  await fs.mkdir(path.join(notes, '文章.assets'), { recursive: true });
  const image = path.join(notes, '文章.assets/图 1.png');
  await fs.writeFile(image, png);
  return { root, notes, image, input: path.join(notes, '论文笔记.md') };
}

test('复制 Typora 相对、绝对、file URL、HTML 和引用式图片，去重并保留原稿', async t => {
  const f = await setup(t);
  const body = `---\ntitle: 原标题\ndate: 2025-11-23 18:10:44\ncover: 文章.assets/图 1.png\ntypora-copy-images-to: 文章.assets\n---\n# 内容\n\n![相对](<文章.assets/图 1.png>)\n![绝对](<${f.image}>)\n![文件](<${pathToFileURL(f.image)}>)\n<img src="文章.assets/图%201.png" width="300">\n![引用][figure]\n\n[figure]: <文章.assets/图 1.png>\n\n![外链](https://example.com/remote.png)\n\n\`![示例](missing.png)\`\n\n\`\`\`md\n![示例](missing.png)\n\`\`\`\n`;
  await fs.writeFile(f.input, body);
  const result = await prepareArticle({ root: f.root, input: f.input });
  assert.equal(result.assets.length, 1);
  assert.equal(result.localImages, 6);
  assert.equal(result.externalImages, 1);
  assert.equal(await fs.readFile(f.input, 'utf8'), body);
  assert.match(result.content, /date: ['"]?2025-11-23/);
  assert.doesNotMatch(result.content, /typora-copy-images-to/);
  assert.doesNotMatch(result.content, /file:\/\/|文章\.assets/);
  assert.match(result.content, /width="300"/);
  assert.match(result.content, /\[figure\]: \/images\/posts\//);
  assert.match(result.content, /!\[示例\]\(missing.png\)/);
});

test('遵循 typora-root-url，中文与空格路径保持可用', async t => {
  const f = await setup(t);
  await fs.writeFile(f.input, '---\ntypora-root-url: .\n---\n![图](</文章.assets/图 1.png>)');
  const result = await prepareArticle({ root: f.root, input: f.input });
  assert.equal(result.assets.length, 1);
  assert.equal(result.filename, '论文笔记.md');
  assert.ok(result.meta.date);
});

test('缺图或伪装成图片的文件阻止发布，检查不写入源码目录', async t => {
  const f = await setup(t);
  await fs.writeFile(f.input, '![图](missing.png)');
  await assert.rejects(prepareArticle({ root: f.root, input: f.input }), /找不到图片/);
  await fs.writeFile(f.image, 'this is not an image');
  await fs.writeFile(f.input, '![图](<文章.assets/图 1.png>)');
  await assert.rejects(prepareArticle({ root: f.root, input: f.input }), /图片格式/);
  assert.deepEqual(await fs.readdir(path.join(f.root, 'source/_posts')), []);
});

test('更新已有文章保持原有日期；源稿变化后要求重新检查', async t => {
  const f = await setup(t);
  await fs.writeFile(path.join(f.root, 'source/_posts/论文笔记.md'), '---\ntitle: 旧标题\ndate: 2024-01-02 03:04:05\n---\n旧正文');
  await fs.writeFile(f.input, '# 新标题\n\n新正文');
  const result = await prepareArticle({ root: f.root, input: f.input });
  assert.equal(result.exists, true);
  assert.match(result.meta.date, /^2024-01-02/);
  await fs.appendFile(f.input, '\n又改了');
  await assert.rejects(prepareArticle({ root: f.root, input: f.input, expectedHash: result.sourceHash }), /重新检查/);
});
