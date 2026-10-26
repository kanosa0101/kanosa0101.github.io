const fs = require('node:fs/promises');
const path = require('node:path');
const { spawn } = require('node:child_process');
const { prepareArticle } = require('./prepare-article.cjs');
const crypto = require('node:crypto');

async function publishArticle({ root, input, category, expectedHash, onProgress = () => {} }) {
  const cache = path.join(root, '.blog-publish');
  await fs.mkdir(cache, { recursive: true });
  const lockPath = path.join(cache, 'publish.lock');
  let lock;
  try { lock = await fs.open(lockPath, 'wx'); }
  catch (error) {
    if (error.code !== 'EEXIST') throw error;
    const previous = Number(await fs.readFile(lockPath, 'utf8'));
    try { process.kill(previous, 0); throw new Error('另一个发布任务正在运行，请等待完成'); }
    catch (error) { if (error.code !== 'ESRCH') throw error; }
    await fs.unlink(lockPath);
    lock = await fs.open(lockPath, 'wx');
  }
  await lock.writeFile(String(process.pid));
  let deploy;
  try {
    let proxy = '';
    try { proxy = JSON.parse((await fs.readFile(path.join(cache, 'settings.json'), 'utf8')).replace(/^\uFEFF/, '')).proxy || ''; }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    const run = (command, args, cwd = root, allowFailure = false) => new Promise((resolve, reject) => {
      const child = spawn(command, args, { cwd, windowsHide: true,
        env: { ...process.env, GIT_TERMINAL_PROMPT: '0', GCM_INTERACTIVE: 'never' } });
      let output = '';
      for (const stream of [child.stdout, child.stderr]) stream.on('data', data => {
        output += data; onProgress({ log: String(data) });
      });
      child.on('error', reject);
      child.on('close', code => code === 0 || allowFailure ? resolve({ code, output: output.trimEnd() })
        : reject(new Error(output.trim().slice(-2500) || `命令执行失败：${command}`)));
    });
    const git = (args, cwd = root, allowFailure = false) => run('git', ['-c', 'core.quotepath=false',
      ...(proxy ? ['-c', `http.proxy=${proxy}`] : []), ...args], cwd, allowFailure);
    onProgress({ stage: '检查原稿与图片' });
    const article = await prepareArticle({ root, input, category, expectedHash });
    const post = 'source/_posts/' + article.filename;
    const files = [post, ...article.assets.map(asset => asset.relative)];
    if ((await git(['branch', '--show-current'])).output !== 'source') throw new Error('当前不是博客 source 分支，请先恢复源代码分支');
    const changes = (await git(['status', '--porcelain', '--untracked-files=all', '--ignore-submodules=all'])).output;
    if (changes.split('\n').filter(Boolean).some(line => !files.includes(line.slice(3).replaceAll('"', '')))) {
      throw new Error('站点有其他未提交修改，已停止发布以免混入文章；Typora 原稿不会丢失。请先保存这些站点修改。');
    }
    const staged = (await git(['diff', '--cached', '--name-only'])).output;
    if (staged.split('\n').filter(Boolean).some(file => !files.includes(file))) throw new Error('仓库已有其他待提交修改，请先处理后再发布');
    const upstream = (await git(['remote', 'get-url', 'origin'])).output;
    await git(['fetch', 'origin', 'source']);
    if (Number((await git(['rev-list', '--count', 'HEAD..origin/source'])).output)) throw new Error('云端博客源代码有新版本，请先同步本机项目再发布');
    onProgress({ stage: `准备发布副本：${article.localImages} 处本地图片，${article.assets.length} 个去重文件` });
    const build = path.join(cache, 'build');
    await fs.mkdir(build, { recursive: true });
    for (const directory of ['source', 'themes', 'scripts']) {
      await fs.rm(path.join(build, directory), { recursive: true, force: true });
      await fs.cp(path.join(root, directory), path.join(build, directory), { recursive: true,
        filter: file => !['.git', 'node_modules'].includes(path.basename(file)) });
    }
    for (const file of ['_config.yml', '_config.butterfly.yml', 'package.json']) await fs.copyFile(path.join(root, file), path.join(build, file));
    await fs.mkdir(path.join(build, 'tools'), { recursive: true });
    await fs.copyFile(path.join(root, 'tools/check-site.cjs'), path.join(build, 'tools/check-site.cjs'));
    await fs.writeFile(path.join(build, post), article.content);
    for (const asset of article.assets) {
      await fs.mkdir(path.dirname(path.join(build, asset.relative)), { recursive: true });
      await fs.writeFile(path.join(build, asset.relative), asset.bytes);
    }
    try { await fs.symlink(path.join(root, 'node_modules'), path.join(build, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir'); }
    catch (error) { if (error.code !== 'EEXIST') throw error; }
    onProgress({ stage: '构建网页并核对链接和图片' });
    const hexo = path.join(root, 'node_modules/hexo/bin/hexo');
    await run(process.execPath, [hexo, 'clean'], build);
    await run(process.execPath, [hexo, 'generate'], build);
    await run(process.execPath, [path.join(build, 'tools/check-site.cjs')], build);
    const currentHash = crypto.createHash('sha256').update(await fs.readFile(article.input)).digest('hex');
    if (currentHash !== article.sourceHash) throw new Error('构建期间原稿被修改，请重新检查后发布');
    let targetHash = null;
    try { targetHash = crypto.createHash('sha256').update(await fs.readFile(article.target)).digest('hex'); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (targetHash !== article.targetHash) throw new Error('博客中的同名文章已被其他操作修改，请重新检查后发布');
    onProgress({ stage: '保存文章和图片到博客源代码' });
    if (path.resolve(input) === path.resolve(article.target)) {
      await fs.mkdir(path.join(cache, 'originals'), { recursive: true });
      await fs.copyFile(article.target, path.join(cache, 'originals', Date.now() + '-' + article.filename));
    }
    await fs.writeFile(article.target, article.content);
    for (const asset of article.assets) {
      await fs.mkdir(path.dirname(path.join(root, asset.relative)), { recursive: true });
      await fs.writeFile(path.join(root, asset.relative), asset.bytes);
    }
    await git(['add', '--', ...files]);
    if ((await git(['diff', '--cached', '--quiet', '--', ...files], root, true)).code !== 0) {
      await git(['commit', '-m', `发布文章：${article.meta.title}`, '--only', '--', ...files]);
    }
    onProgress({ stage: '推送文章源文件和图片' });
    await git(['push', 'origin', 'source']);
    onProgress({ stage: '发布生成的网页' });
    deploy = await fs.mkdtemp(path.join(cache, 'deploy-'));
    await git(['clone', '--branch', 'main', '--single-branch', upstream, deploy]);
    for (const entry of await fs.readdir(path.join(build, 'public'))) {
      await fs.cp(path.join(build, 'public', entry), path.join(deploy, entry), { recursive: true, force: true });
    }
    await fs.writeFile(path.join(deploy, '.nojekyll'), '');
    await git(['add', '--all'], deploy);
    if ((await git(['diff', '--cached', '--quiet'], deploy, true)).code !== 0) {
      await git(['commit', '-m', `更新博客：${article.meta.title}`], deploy);
    }
    await git(['push', 'origin', 'HEAD:main'], deploy);
    const result = { title: article.meta.title, filename: article.filename,
      sourceCommit: (await git(['rev-parse', 'HEAD'])).output, mainCommit: (await git(['rev-parse', 'HEAD'], deploy)).output };
    await fs.writeFile(path.join(cache, 'last-publication.json'), JSON.stringify(result, null, 2));
    onProgress({ stage: '已成功推送，GitHub Pages 正在更新网站', done: true });
    return result;
  } finally {
    if (deploy) await fs.rm(deploy, { recursive: true, force: true }).catch(() => {});
    await lock.close();
    await fs.unlink(lockPath);
  }
}

async function buildPreviewArticle({ root, input, category, onProgress = () => {} }) {
  const cache = path.join(root, '.blog-publish');
  await fs.mkdir(cache, { recursive: true });
  onProgress({ stage: '检查原稿与图片' });
  const article = await prepareArticle({ root, input, category });
  const post = 'source/_posts/' + article.filename;

  onProgress({ stage: `准备预览副本：${article.localImages} 处本地图片，${article.assets.length} 个去重文件` });
  const build = path.join(cache, 'build');
  await fs.mkdir(build, { recursive: true });
  for (const directory of ['source', 'themes', 'scripts']) {
    await fs.rm(path.join(build, directory), { recursive: true, force: true });
    await fs.cp(path.join(root, directory), path.join(build, directory), {
      recursive: true,
      filter: file => !['.git', 'node_modules'].includes(path.basename(file))
    });
  }
  for (const file of ['_config.yml', '_config.butterfly.yml', 'package.json']) {
    await fs.copyFile(path.join(root, file), path.join(build, file));
  }
  await fs.mkdir(path.join(build, 'tools'), { recursive: true });
  await fs.copyFile(path.join(root, 'tools/check-site.cjs'), path.join(build, 'tools/check-site.cjs'));
  await fs.writeFile(path.join(build, post), article.content);
  for (const asset of article.assets) {
    await fs.mkdir(path.dirname(path.join(build, asset.relative)), { recursive: true });
    await fs.writeFile(path.join(build, asset.relative), asset.bytes);
  }
  try {
    await fs.symlink(path.join(root, 'node_modules'), path.join(build, 'node_modules'), process.platform === 'win32' ? 'junction' : 'dir');
  } catch (error) {
    if (error.code !== 'EEXIST') throw error;
  }

  const run = (command, args, cwd = build) => new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, windowsHide: true });
    let output = '';
    for (const stream of [child.stdout, child.stderr]) stream.on('data', data => {
      output += data; onProgress({ log: String(data) });
    });
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolve(output) : reject(new Error(output.trim().slice(-2500) || `命令执行失败：${command}`)));
  });

  onProgress({ stage: '构建本地预览网页' });
  const hexo = path.join(root, 'node_modules/hexo/bin/hexo');
  await run(process.execPath, [hexo, 'clean']);
  await run(process.execPath, [hexo, 'generate']);
  await run(process.execPath, [path.join(build, 'tools/check-site.cjs')]);

  // Locate the generated HTML page for this post
  const publicDir = path.join(build, 'public');
  const baseName = path.basename(article.filename, '.md');
  let postUrl = '/';

  const fsSync = require('node:fs');
  function findPost(dir) {
    for (const item of fsSync.readdirSync(dir)) {
      const full = path.join(dir, item);
      if (fsSync.statSync(full).isDirectory()) {
        findPost(full);
      } else if (item === 'index.html') {
        const rel = path.relative(publicDir, full).replace(/\\/g, '/');
        if (rel !== 'index.html' && (rel.includes(baseName) || rel.includes(encodeURIComponent(baseName)))) {
          postUrl = '/' + rel.replace(/\/index\.html$/, '/');
        }
      }
    }
  }
  findPost(publicDir);

  onProgress({ stage: '本地预览生成完毕', done: true });
  return {
    article,
    postUrl,
    publicDir,
    buildDir: build,
    sourceHash: article.sourceHash
  };
}

module.exports = { publishArticle, buildPreviewArticle };
