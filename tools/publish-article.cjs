const path = require('node:path');
const { prepareArticle } = require('./prepare-article.cjs');
const { publishArticle } = require('./publish-site.cjs');

async function main() {
  const [mode, input, ...rest] = process.argv.slice(2);
  if (!['--inspect', '--publish'].includes(mode) || !input) throw new Error('请双击“发布文章.cmd”选择 Typora 原稿；也可以将 .md 文件拖到该入口上。');
  const options = { root: path.resolve(__dirname, '..'), input };
  for (let index = 0; index < rest.length; index += 2) {
    if (rest[index] === '--category') options.category = rest[index + 1];
    else if (rest[index] === '--expected-hash') options.expectedHash = rest[index + 1];
    else throw new Error('无法识别的发布参数');
  }
  if (mode === '--inspect') {
    const article = await prepareArticle(options);
    console.log(JSON.stringify({ title: article.meta.title, date: article.meta.date, categories: article.meta.categories,
      filename: article.filename, exists: article.exists, sourceHash: article.sourceHash,
      localImages: article.localImages, copiedFiles: article.assets.length, externalImages: article.externalImages }));
  } else {
    await publishArticle({ ...options, onProgress: progress => {
      if (progress.stage) console.log('\n[进度] ' + progress.stage);
      if (progress.log) process.stdout.write(progress.log);
    } });
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
