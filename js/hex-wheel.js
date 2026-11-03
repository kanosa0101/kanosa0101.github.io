/**
 * kanosa's blog - Hexagonal Tactical Wheel Menu
 * Tab 键触发六边形战术轮盘导航系统
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
      return;
    }

    wheelEl = document.createElement('div');
    wheelEl.id = 'hex-wheel-overlay';
    wheelEl.className = 'hex-wheel-backdrop';
    wheelEl.setAttribute('aria-hidden', 'true');

    // 组装 6 个六边形节点
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
        <!-- 背景科幻光晕 -->
        <div class="hex-wheel-glow"></div>

        <!-- 外围六边形节点容器 -->
        <div class="hex-nodes-track">
          ${nodesHtml}
        </div>

        <!-- 中央枢纽 Core Hub -->
        <div class="hex-center-hub" id="hex-wheel-hub">
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

        <!-- 底部快捷指令栏 -->
        <div class="hex-wheel-footer">
          <div class="footer-chip"><kbd>Tab</kbd> 顺时针轮转</div>
          <div class="footer-chip"><kbd>1-6</kbd> 快速激活</div>
          <div class="footer-chip"><kbd>Enter</kbd> / 点击确认</div>
          <div class="footer-chip"><kbd>Esc</kbd> 退出</div>
        </div>
      </div>
    `;

    document.body.appendChild(wheelEl);
    bindEvents();
  }

  function updateActiveState(index) {
    activeIndex = (index + WHEEL_ITEMS.length) % WHEEL_ITEMS.length;
    const item = WHEEL_ITEMS[activeIndex];

    // 更新外围节点样式
    const nodes = wheelEl.querySelectorAll('.hex-nav-node');
    nodes.forEach((node, i) => {
      if (i === activeIndex) {
        node.classList.add('is-active');
      } else {
        node.classList.remove('is-active');
      }
    });

    // 更新中央枢纽信息
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
    if (!wheelEl) createWheelDOM();
    isOpen = true;
    wheelEl.classList.add('is-open');

    // 智能推断当前所在路由对应的索引
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
    } else if (window.pjax) {
      window.pjax.loadUrl(item.path);
    } else {
      window.location.href = item.path;
    }
  }

  function bindEvents() {
    // 鼠标点击遮罩外部关闭
    wheelEl.addEventListener('click', function(e) {
      if (e.target === wheelEl || e.target.classList.contains('hex-wheel-modal')) {
        closeWheel();
      }
    });

    // 节点鼠标移入与点击
    const nodes = wheelEl.querySelectorAll('.hex-nav-node');
    nodes.forEach(node => {
      node.addEventListener('mouseenter', function() {
        const idx = parseInt(this.getAttribute('data-index'), 10);
        updateActiveState(idx);
      });

      node.addEventListener('click', function(e) {
        e.stopPropagation();
        navigateToActive();
      });
    });

    // 点击中央枢纽也可直接激活当前项
    const hub = document.getElementById('hex-wheel-hub');
    if (hub) {
      hub.addEventListener('click', function(e) {
        e.stopPropagation();
        navigateToActive();
      });
    }
  }

  // 全局键盘监听
  window.addEventListener('keydown', function(e) {
    // 如果焦点在输入框/富文本中，不拦截 Tab 键
    const activeTagName = (document.activeElement && document.activeElement.tagName) || '';
    if (['INPUT', 'TEXTAREA'].includes(activeTagName) || document.activeElement.isContentEditable) {
      return;
    }

    if (e.key === 'Tab') {
      e.preventDefault();
      if (!isOpen) {
        openWheel();
      } else {
        // 已开启时，按 Tab 顺时针轮转
        updateActiveState(activeIndex + 1);
      }
      return;
    }

    if (!isOpen) return;

    if (e.key === 'Escape') {
      e.preventDefault();
      closeWheel();
      return;
    }

    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      navigateToActive();
      return;
    }

    // 支持数字键 1 - 6 直达
    if (['1', '2', '3', '4', '5', '6'].includes(e.key)) {
      e.preventDefault();
      const targetIdx = parseInt(e.key, 10) - 1;
      updateActiveState(targetIdx);
      setTimeout(() => {
        navigateToActive();
      }, 120);
      return;
    }

    // 方向键支持
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      e.preventDefault();
      updateActiveState(activeIndex + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      e.preventDefault();
      updateActiveState(activeIndex - 1);
    }
  }, true);

  // 初始化
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createWheelDOM);
  } else {
    createWheelDOM();
  }
})();
