/**
 * kanosa's blog - Hexagonal Portal Transition
 * 赛博六边形跃迁转场系统：轻量、无文字阻滞、防卡顿无缝过渡
 */
(function() {
  'use strict';

  let overlay = null;
  let isNavigating = false;
  let watchdogTimer = null;

  function ensureOverlay() {
    if (overlay && document.getElementById('hex-portal-overlay')) {
      return overlay;
    }

    const existing = document.getElementById('hex-portal-overlay');
    if (existing) {
      overlay = existing;
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
      </div>
    `;

    document.body.appendChild(overlay);
    return overlay;
  }

  function dismissOverlay() {
    if (watchdogTimer) {
      clearTimeout(watchdogTimer);
      watchdogTimer = null;
    }
    isNavigating = false;
    try {
      sessionStorage.removeItem('hex_warp_active');
    } catch (e) {}

    const el = ensureOverlay();
    el.classList.add('is-leaving');
    setTimeout(function() {
      el.classList.remove('is-active', 'is-leaving');
    }, 280);
  }

  function triggerTransition(targetUrl) {
    if (isNavigating) return;
    isNavigating = true;

    const el = ensureOverlay();
    el.classList.remove('is-leaving');
    el.classList.add('is-active');

    try {
      sessionStorage.setItem('hex_warp_active', '1');
    } catch (e) {}

    // 防卡顿看门狗：800ms 内若页面未跳转成功或由于网络延迟留在原页，自动平滑隐去遮罩
    if (watchdogTimer) clearTimeout(watchdogTimer);
    watchdogTimer = setTimeout(function() {
      dismissOverlay();
    }, 800);

    setTimeout(function() {
      window.location.href = targetUrl;
    }, 160);
  }

  // 暴露给外部模块（如 hex-wheel 轮盘）
  window.triggerHexTransition = triggerTransition;

  // 检查是否从上一页跃迁而来，执行入场展开动画
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
            isNavigating = false;
          }, 260);
        }, 80);
      } else {
        dismissOverlay();
      }
    } catch (e) {
      dismissOverlay();
    }
  }

  // 浏览器历史回退/前进（bfcache）或页面激活时确保清除残留遮罩
  window.addEventListener('pageshow', function() {
    dismissOverlay();
  });
  window.addEventListener('popstate', function() {
    dismissOverlay();
  });

  // 监听站内链接点击
  document.addEventListener('click', function(e) {
    // 忽略右键、中键、组合键打开新标签等操作
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;

    const link = e.target.closest('a');
    if (!link) return;

    const href = link.getAttribute('href');
    const target = link.getAttribute('target');

    if (!href || href.startsWith('#') || href.startsWith('javascript:') || target === '_blank') return;
    if (link.hasAttribute('download')) return;

    try {
      const u = new URL(href, window.location.origin);
      if (u.origin === window.location.origin) {
        // 同路径同锚点或空变化，不触发
        if (u.pathname === window.location.pathname && (u.hash || (!u.search && !u.hash))) return;

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
