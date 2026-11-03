/**
 * kanosa's blog - Hexagonal Tactical Wheel Menu
 * Tab 键战术六边形轮盘导航系统
 */
(function() {
  'use strict';

  const WHEEL_ITEMS = [
    { id: 'home', name: '首页', en: 'HOME · 核心枢纽', path: '/', icon: 'fas fa-home', key: '1' },
    { id: 'archives', name: '归档', en: 'ARCHIVES · 历程沉淀', path: '/archives/', icon: 'fas fa-archive', key: '2' },
    { id: 'tags', name: '标签', en: 'TAGS · 知识聚合', path: '/tags/', icon: 'fas fa-tags', key: '3' },
    { id: 'categories', name: '分类', en: 'CATEGORIES · 结构专题', path: '/categories/', icon: 'fas fa-folder-open', key: '4' },
    { id: 'link', name: '友链', en: 'LINKS · 灵感星图', path: '/link/', icon: 'fas fa-link', key: '5' },
    { id: 'about', name: '关于', en: 'ABOUT · 架构档案', path: '/about/', icon: 'fas fa-user-astronaut', key: '6' }
  ];

  let wheelEl = null;
  let isOpen = false;
  let activeIndex = 0;

  function createWheelDOM() {
    if (document.getElementById('hex-wheel-overlay')) {
      wheelEl = document.getElementById('hex-wheel-overlay');
      return wheelEl;
    }

    wheelEl = document.createElement('div');
    wheelEl.id = 'hex-wheel-overlay';
    wheelEl.className = 'hex-wheel-backdrop';
    wheelEl.setAttribute('aria-hidden', 'true');

    const nodesHtml = WHEEL_ITEMS.map((item, idx) => `
      <div class="hex-nav-node node-pos-${idx}" data-index="${idx}" data-path="${item.path}" title="${item.name} (${item.key})">
        <div class="hex-node-body">
          <svg class="hex-vector" viewBox="0 0 100 115.47">
            <polygon class="hex-poly-bg" points="50 0, 100 28.87, 100 86.6, 50 115.47, 0 86.6, 0 28.87" />
            <polygon class="hex-poly-border" points="50 0, 100 28.87, 100 86.6, 50 115.47, 0 86.6, 0 28.87" />
          </svg>
          <div class="hex-node-content">
            <span class="hex-node-key">${item.key}</span>
            <i class="${item.icon} hex-node-icon"></i>
            <span class="hex-node-name">${item.name}</span>
          </div>
        </div>
      </div>
    `).join('');

    wheelEl.innerHTML = `
      <div class="hex-wheel-modal">
        <button class="hex-wheel-close-btn" title="关闭轮盘 (Esc)" aria-label="关闭">&times;</button>
        <div class="hex-wheel-glow"></div>
        <div class="hex-nodes-track">
          ${nodesHtml}
        </div>
        <div class="hex-center-hub" id="hex-wheel-hub" title="点击跳转">
          <div class="hub-hexagon-wrap">
            <svg class="hub-vector" viewBox="0 0 140 161.66">
              <polygon class="hub-poly-glow" points="70 0, 140 40.41, 140 121.24, 70 161.66, 0 121.24, 0 40.41" />
              <polygon class="hub-poly-inner" points="70 0, 140 40.41, 140 121.24, 70 161.66, 0 121.24, 0 40.41" />
            </svg>
            <div class="hub-data">
              <i class="fas fa-home hub-icon" id="wheel-hub-icon"></i>
              <div class="hub-title" id="wheel-hub-title">首页</div>
              <div class="hub-sub" id="wheel-hub-sub">HOME · 核心枢纽</div>
              <div class="hub-route" id="wheel-hub-route">/</div>
            </div>
          </div>
          <div class="hub-orbit-ring"></div>
        </div>
        <div class="hex-wheel-footer">
          <div class="footer-chip"><kbd>Tab</kbd> 顺时针轮转</div>
          <div class="footer-chip"><kbd>1-6</kbd> 快速激活</div>
          <div class="footer-chip"><kbd>Enter</kbd> / 点击 确认</div>
          <div class="footer-chip"><kbd>Esc</kbd> 退出</div>
        </div>
      </div>
    `;

    document.body.appendChild(wheelEl);
    bindEvents();
    return wheelEl;
  }

  function updateActiveState(index) {
    activeIndex = (index + WHEEL_ITEMS.length) % WHEEL_ITEMS.length;
    const item = WHEEL_ITEMS[activeIndex];

    if (!wheelEl) return;
    const nodes = wheelEl.querySelectorAll('.hex-nav-node');
    nodes.forEach((node, i) => {
      if (i === activeIndex) {
        node.classList.add('is-active');
      } else {
        node.classList.remove('is-active');
      }
    });

    const hubIcon = document.getElementById('wheel-hub-icon');
    const hubTitle = document.getElementById('wheel-hub-title');
    const hubSub = document.getElementById('wheel-hub-sub');
    const hubRoute = document.getElementById('wheel-hub-route');

    if (hubIcon) hubIcon.className = `${item.icon} hub-icon`;
    if (hubTitle) hubTitle.textContent = item.name;
    if (hubSub) hubSub.textContent = item.en;
    if (hubRoute) hubRoute.textContent = item.path;
  }

  function openWheel() {
    createWheelDOM();
    isOpen = true;
    wheelEl.classList.add('is-open');

    const curPath = window.location.pathname;
    let initialIdx = 0;
    if (curPath.includes('/about')) initialIdx = 5;
    else if (curPath.includes('/link')) initialIdx = 4;
    else if (curPath.includes('/categories')) initialIdx = 3;
    else if (curPath.includes('/tags')) initialIdx = 2;
    else if (curPath.includes('/archives')) initialIdx = 1;
    else initialIdx = 0;

    updateActiveState(initialIdx);
  }

  function closeWheel() {
    if (!isOpen || !wheelEl) return;
    wheelEl.classList.remove('is-open');
    isOpen = false;
  }

  function navigateToActive() {
    const item = WHEEL_ITEMS[activeIndex];
    closeWheel();
    if (window.triggerHexTransition) {
      window.triggerHexTransition(item.path);
    } else {
      window.location.href = item.path;
    }
  }

  function bindEvents() {
    wheelEl.addEventListener('click', function(e) {
      if (e.target === wheelEl || e.target.classList.contains('hex-wheel-modal')) {
        closeWheel();
      }
    });

    const closeBtn = wheelEl.querySelector('.hex-wheel-close-btn');
    if (closeBtn) {
      closeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        closeWheel();
      });
    }

    const nodes = wheelEl.querySelectorAll('.hex-nav-node');
    nodes.forEach(node => {
      node.addEventListener('mouseenter', function() {
        const idx = parseInt(this.getAttribute('data-index'), 10);
        updateActiveState(idx);
      });

      node.addEventListener('click', function(e) {
        e.stopPropagation();
        const idx = parseInt(this.getAttribute('data-index'), 10);
        activeIndex = idx;
        navigateToActive();
      });
    });

    const hub = document.getElementById('hex-wheel-hub');
    if (hub) {
      hub.addEventListener('click', function(e) {
        e.stopPropagation();
        navigateToActive();
      });
    }
  }

  // 暴露给全局调用
  window.openHexWheel = openWheel;
  window.closeHexWheel = closeWheel;

  // 全局键盘监听
  window.addEventListener('keydown', function(e) {
    const activeTagName = (document.activeElement && document.activeElement.tagName) || '';
    if (['INPUT', 'TEXTAREA'].includes(activeTagName) || (document.activeElement && document.activeElement.isContentEditable)) {
      return;
    }

    const isTab = (e.key === 'Tab' || e.code === 'Tab' || e.keyCode === 9);

    if (isTab) {
      e.preventDefault();
      e.stopPropagation();
      if (!isOpen) {
        openWheel();
      } else {
        updateActiveState(activeIndex + 1);
      }
      return;
    }

    if (!isOpen) return;

    if (e.key === 'Escape' || e.keyCode === 27) {
      e.preventDefault();
      closeWheel();
      return;
    }

    if (e.key === 'Enter' || e.keyCode === 13 || e.key === ' ') {
      e.preventDefault();
      navigateToActive();
      return;
    }

    if (['1', '2', '3', '4', '5', '6'].includes(e.key)) {
      e.preventDefault();
      const targetIdx = parseInt(e.key, 10) - 1;
      updateActiveState(targetIdx);
      setTimeout(function() {
        navigateToActive();
      }, 120);
      return;
    }

    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      updateActiveState(activeIndex + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      updateActiveState(activeIndex - 1);
    }
  }, true);

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createWheelDOM);
  } else {
    createWheelDOM();
  }
})();
