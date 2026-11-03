/**
 * kanosa's blog - Hexagonal Portal Transition
 * 赛博六边形跃迁转场系统：自包含、轻量且高度兼容
 */
(function() {
  'use strict';

  let overlay = null;
  let isNavigating = false;

  function ensureOverlay() {
    if (document.getElementById('hex-portal-overlay')) {
      overlay = document.getElementById('hex-portal-overlay');
      return overlay;
    }

    overlay = document.createElement('div');
    overlay.id = 'hex-portal-overlay';
    overlay.className = 'hex-portal-wrap';
    overlay.setAttribute('aria-hidden', 'true');

    overlay.innerHTML = `
      <div class="hex-portal-backdrop"></div>
      <div class="hex-portal-stage">
        <div class="hex-energy-ring outer-ring">
          <svg class="hex-svg" viewBox="0 0 120 138.56">
            <polygon class="hex-poly outer-poly" points="60 0, 120 34.64, 120 103.92, 60 138.56, 0 103.92, 0 34.64" />
          </svg>
        </div>
        <div class="hex-energy-ring middle-ring">
          <svg class="hex-svg" viewBox="0 0 100 115.47">
            <polygon class="hex-poly middle-poly" points="50 0, 100 28.87, 100 86.6, 50 115.47, 0 86.6, 0 28.87" />
          </svg>
        </div>
        <div class="hex-energy-core">
          <div class="hex-core-icon-box">
            <i class="fas fa-cube hex-cube-icon"></i>
          </div>
        </div>
        <div class="hex-portal-hud">
          <div class="hex-hud-badge">
            <span class="hud-pulse-dot"></span>
            <span class="hud-tag">WARP DRIVE // 空间跃迁</span>
          </div>
          <div class="hex-hud-title" id="hex-portal-dest">INITIALIZING DOM...</div>
          <div class="hex-hud-circuit">
            <div class="circuit-rail"></div>
            <div class="circuit-beam"></div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(overlay);
    return overlay;
  }

  function formatPathName(path) {
    if (!path || path === '/') return 'HOME · 首页枢纽';
    if (path.includes('/about')) return 'ABOUT · 关于本人';
    if (path.includes('/archives')) return 'ARCHIVES · 历史归档';
    if (path.includes('/tags')) return 'TAGS · 标签聚合';
    if (path.includes('/categories')) return 'CATEGORIES · 分类导览';
    if (path.includes('/link')) return 'LINKS · 友链空间';
    return 'TARGET // ' + path.replace(/^\/|\/$/g, '').toUpperCase();
  }

  function triggerTransition(targetUrl) {
    if (isNavigating) return;
    isNavigating = true;

    const el = ensureOverlay();
    const destTitle = document.getElementById('hex-portal-dest');
    if (destTitle && targetUrl) {
      try {
        const u = new URL(targetUrl, window.location.origin);
        destTitle.textContent = formatPathName(u.pathname);
      } catch (e) {
        destTitle.textContent = 'CONNECTING // TARGET';
      }
    }

    try {
      sessionStorage.setItem('hex_warp_active', '1');
    } catch (e) {}

    el.classList.remove('is-leaving');
    el.classList.add('is-active');

    setTimeout(function() {
      window.location.href = targetUrl;
    }, 220);
  }

  // 暴露给外部模块（如 hex-wheel 轮盘）
  window.triggerHexTransition = triggerTransition;

  // 检查是否从上一页跃迁而来
  function checkEntryWarp() {
    try {
      if (sessionStorage.getItem('hex_warp_active') === '1') {
        sessionStorage.removeItem('hex_warp_active');
        const el = ensureOverlay();
        el.classList.add('is-active');
        // 瞬间触发离场扩散动画
        setTimeout(function() {
          el.classList.add('is-leaving');
          setTimeout(function() {
            el.classList.remove('is-active', 'is-leaving');
          }, 320);
        }, 120);
      }
    } catch (e) {}
  }

  // 监听站内链接点击
  document.addEventListener('click', function(e) {
    const link = e.target.closest('a');
    if (!link) return;

    const href = link.getAttribute('href');
    const target = link.getAttribute('target');

    if (!href || href.startsWith('#') || href.startsWith('javascript:') || target === '_blank') return;
    if (link.hasAttribute('download')) return;

    try {
      const u = new URL(href, window.location.origin);
      if (u.origin === window.location.origin) {
        // 如果是相同路径或锚点，不重复触发
        if (u.pathname === window.location.pathname && u.hash) return;
        if (u.pathname === window.location.pathname && !u.search && !u.hash) return;

        e.preventDefault();
        triggerTransition(u.href);
      }
    } catch (err) {}
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', checkEntryWarp);
  } else {
    checkEntryWarp();
  }
})();
