/**
 * kanosa's blog - Hexagonal Portal Page Transition
 * 赛博六边形跃迁转场系统：拒绝普通进度条，打造游戏级空间快门
 */
(function() {
  'use strict';

  let overlay = null;
  let isTransitioning = false;
  let transitionStartTime = 0;
  const MIN_TRANSITION_MS = 380; // 最小展示时间，确保视觉仪式感

  function createPortalOverlay() {
    if (document.getElementById('hex-portal-overlay')) {
      overlay = document.getElementById('hex-portal-overlay');
      return;
    }

    overlay = document.createElement('div');
    overlay.id = 'hex-portal-overlay';
    overlay.className = 'hex-portal-wrap';
    overlay.setAttribute('aria-hidden', 'true');

    overlay.innerHTML = `
      <div class="hex-portal-backdrop"></div>
      <div class="hex-portal-stage">
        <!-- 几何光圈与六边形能量环 -->
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

        <!-- 战术 HUD 状态信息 -->
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
  }

  function formatPathName(path) {
    if (!path || path === '/') return 'HOME · 首页枢纽';
    if (path.includes('/about')) return 'ABOUT · 关于本人';
    if (path.includes('/archives')) return 'ARCHIVES · 历史归档';
    if (path.includes('/tags')) return 'TAGS · 标签聚合';
    if (path.includes('/categories')) return 'CATEGORIES · 分类导览';
    if (path.includes('/link')) return 'LINKS · 友链空间';
    return 'SYNCING // ' + path.replace(/^\/|\/$/g, '').toUpperCase();
  }

  function startTransition(destUrl) {
    if (isTransitioning) return;
    isTransitioning = true;
    transitionStartTime = Date.now();

    if (!overlay) createPortalOverlay();

    const destTitle = document.getElementById('hex-portal-dest');
    if (destTitle && destUrl) {
      try {
        const urlObj = new URL(destUrl, window.location.origin);
        destTitle.textContent = formatPathName(urlObj.pathname);
      } catch (e) {
        destTitle.textContent = 'CONNECTING // TARGET';
      }
    }

    overlay.classList.remove('is-leaving');
    overlay.classList.add('is-active');
  }

  function endTransition() {
    if (!isTransitioning || !overlay) return;

    const elapsed = Date.now() - transitionStartTime;
    const remaining = Math.max(0, MIN_TRANSITION_MS - elapsed);

    setTimeout(() => {
      overlay.classList.add('is-leaving');
      setTimeout(() => {
        overlay.classList.remove('is-active', 'is-leaving');
        isTransitioning = false;
      }, 300);
    }, remaining);
  }

  // 暴露给外部模块（如 hex-wheel 轮盘）
  window.triggerHexTransition = function(url) {
    startTransition(url);
    if (window.pjax) {
      window.pjax.loadUrl(url);
    } else {
      setTimeout(() => {
        window.location.href = url;
      }, 160);
    }
  };

  // 绑定 PJAX 钩子
  document.addEventListener('pjax:send', function(e) {
    const targetUrl = (e && e.data && e.data.url) || window.location.href;
    startTransition(targetUrl);
  });

  document.addEventListener('pjax:complete', function() {
    endTransition();
  });

  document.addEventListener('pjax:error', function() {
    endTransition();
  });

  // 普通点击拦截降级（在未启用 PJAX 或某些链接无法走 PJAX 时）
  document.addEventListener('click', function(e) {
    const link = e.target.closest('a');
    if (!link) return;

    const href = link.getAttribute('href');
    const target = link.getAttribute('target');

    // 过滤外链、锚点、空白链接
    if (!href || href.startsWith('#') || href.startsWith('javascript:') || target === '_blank') return;
    if (link.hasAttribute('download')) return;

    try {
      const targetUrl = new URL(href, window.location.origin);
      if (targetUrl.origin === window.location.origin) {
        // 同源站内跳转
        if (!window.pjax) {
          e.preventDefault();
          startTransition(targetUrl.href);
          setTimeout(() => {
            window.location.href = targetUrl.href;
          }, 180);
        }
      }
    } catch (err) {
      // 忽略无法解析的地址
    }
  });

  // DOM 就绪检查
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createPortalOverlay);
  } else {
    createPortalOverlay();
  }
})();
