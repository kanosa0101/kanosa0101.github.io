'use strict';

const MATH_STORAGE = new Map();

hexo.extend.filter.register('before_post_render', function(data) {
  if (!data.content) return data;

  const mathBlocks = [];
  const codeBlocks = [];

  // 1. Temporarily shelter code blocks from math extraction
  let content = data.content.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (match) => {
    const id = `HEXOCODEBLOCK${codeBlocks.length}END`;
    codeBlocks.push(match);
    return id;
  });

  // 2. Protect display math $$...$$
  content = content.replace(/\$\$([\s\S]+?)\$\$/g, (match) => {
    const id = `HEXOMATHBLOCK${mathBlocks.length}END`;
    mathBlocks.push(match);
    return id;
  });

  // 3. Protect inline math $...$
  content = content.replace(/(?<!\\|\$)\$(?!\$)([^\$\n]+?)(?<!\\|\$)\$(?!\$)/g, (match) => {
    const id = `HEXOMATHBLOCK${mathBlocks.length}END`;
    mathBlocks.push(match);
    return id;
  });

  // 4. Restore sheltered code blocks so markdown renderer can process them
  content = content.replace(/HEXOCODEBLOCK(\d+)END/g, (_, idx) => codeBlocks[Number(idx)]);

  data.content = content;
  const key = data._id || data.path || data.slug || data.title || String(Math.random());
  data._math_protect_key = key;
  MATH_STORAGE.set(key, mathBlocks);

  return data;
}, 1);

hexo.extend.filter.register('after_post_render', function(data) {
  if (!data.content) return data;

  const key = data._math_protect_key || data._id || data.path || data.slug || data.title;
  const mathBlocks = MATH_STORAGE.get(key);
  if (!mathBlocks || mathBlocks.length === 0) return data;

  data.content = data.content.replace(/HEXOMATHBLOCK(\d+)END/g, (_, idx) => {
    return mathBlocks[Number(idx)];
  });

  MATH_STORAGE.delete(key);
  return data;
}, 9999);
