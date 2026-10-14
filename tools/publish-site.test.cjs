const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const { execFileSync } = require('node:child_process');
const { publishArticle } = require('./publish-site.cjs');

test('真实构建失败不写入源码或推送；图片随文章进入两个本地远端分支，原稿不变', { timeout: 180000 }, async t => {
  const project = path.resolve(__dirname, '..');
  const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'typora-release-'));
  const remote = path.join(temp, 'remote.git'), root = path.join(temp, 'workspace');
  const git = (args, cwd = temp) => execFileSync('git', ['-c', 'core.quotepath=false', ...args], { cwd, windowsHide: true, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  t.after(async () => {
    for (const file of [path.join(root, 'node_modules'), path.join(root, '.blog-publish/build/node_modules')]) {
      try { await fs.unlink(file); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    }
    await fs.rm(temp, { recursive: true, force: true });
  });
  git(['init', '--bare', remote]);
  git(['--git-dir=' + remote, 'fetch', project, 'source:source']);
  const seed = path.join(temp, 'main-seed');
  git(['init', '-b', 'main', seed]);
  git(['config', 'user.name', 'Publisher Test'], seed); git(['config', 'user.email', 'publisher@example.invalid'], seed);
  await fs.writeFile(path.join(seed, 'index.html'), '<!doctype html><title>Initial site</title>');
  git(['add', 'index.html'], seed); git(['commit', '-m', 'Seed local main branch'], seed);
  git(['push', remote, 'main'], seed);
  git(['clone', '--branch', 'source', remote, root]);
  git(['config', 'user.name', 'Publisher Test'], root); git(['config', 'user.email', 'publisher@example.invalid'], root);
  await fs.appendFile(path.join(root, '.gitignore'), '\n.blog-publish/\n');
  await fs.copyFile(path.join(project, 'tools/check-site.cjs'), path.join(root, 'tools/check-site.cjs'));
  git(['add', '.gitignore', 'tools/check-site.cjs'], root); git(['commit', '-m', 'Prepare isolated publisher test'], root); git(['push', 'origin', 'source'], root);
  await fs.cp(path.join(project, 'themes/butterfly'), path.join(root, 'themes/butterfly'), { recursive: true,
    filter: file => !['.git', 'node_modules'].includes(path.basename(file)) });
  await fs.symlink(path.join(project, 'node_modules'), path.join(root, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
  const input = path.join(temp, '测试原稿.md');
  await fs.copyFile(path.join(project, 'source/images/avatar.jpg'), path.join(temp, '本地图片.jpg'));
  const original = '# 隔离发布测试\n\n![本地图片](本地图片.jpg)\n';
  await fs.writeFile(input, original + '\n[无效链接](./missing-page/)');
  const sourceBefore = git(['--git-dir=' + remote, 'rev-parse', 'source']);
  const mainBefore = git(['--git-dir=' + remote, 'rev-parse', 'main']);
  const progress = value => { if (value.stage) console.log('测试：' + value.stage); };
  await assert.rejects(publishArticle({ root, input, onProgress: progress }), /不存在/);
  assert.equal(git(['--git-dir=' + remote, 'rev-parse', 'source']), sourceBefore);
  assert.equal(git(['--git-dir=' + remote, 'rev-parse', 'main']), mainBefore);
  await assert.rejects(fs.access(path.join(root, 'source/_posts/测试原稿.md')));
  await fs.writeFile(input, original);
  await publishArticle({ root, input, onProgress: progress });
  assert.equal(await fs.readFile(input, 'utf8'), original);
  const published = git(['--git-dir=' + remote, 'show', 'source:source/_posts/测试原稿.md']);
  const url = published.match(/\/images\/posts\/[a-f0-9]+\.jpg/)[0];
  assert.ok(git(['--git-dir=' + remote, 'ls-tree', '-r', '--name-only', 'main']).includes(url.slice(1)));
  assert.ok(git(['--git-dir=' + remote, 'ls-tree', '-r', '--name-only', 'source']).includes('source' + url));
  assert.notEqual(git(['--git-dir=' + remote, 'rev-parse', 'main']), mainBefore);
  assert.doesNotMatch(published, /本地图片\.jpg/);
  const articlePage = git(['--git-dir=' + remote, 'ls-tree', '-r', '--name-only', 'main']).split('\n').find(file => file.endsWith('/测试原稿/index.html'));
  assert.ok(articlePage);
  assert.ok(git(['--git-dir=' + remote, 'show', 'main:' + articlePage]).includes('src="' + url + '"'));
  const publishedImage = execFileSync('git', ['--git-dir=' + remote, 'show', 'main:' + url.slice(1)], { windowsHide: true });
  assert.deepEqual(publishedImage, await fs.readFile(path.join(temp, '本地图片.jpg')));
});
