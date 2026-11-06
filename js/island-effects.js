/**
 * kanosa's blog - Island Architecture Interactive Effects
 * - Mechanism B: Independent Scroll Reveal (IntersectionObserver)
 * - Mechanism C: Radial Spotlight Hover Border & Surface Tracking
 * - About Side TOC: Side Outline Rail Scroll-Spy & Smooth Navigation
 */
(function () {
  'use strict';

  function initIslandScrollReveal() {
    var cards = document.querySelectorAll('.island-card:not(.island-revealed)');
    if (!cards || cards.length === 0) return;

    if (!('IntersectionObserver' in window)) {
      cards.forEach(function (card) {
        card.classList.add('island-revealed');
      });
      return;
    }

    var observer = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('island-revealed');
          obs.unobserve(entry.target);
        }
      });
    }, {
      root: null,
      rootMargin: '0px 0px -40px 0px',
      threshold: 0.08
    });

    cards.forEach(function (card) {
      observer.observe(card);
    });
  }

  function initSpotlightCards() {
    var spotlightCards = document.querySelectorAll('.island-project-card, .status-island, .marquee-island, .timeline-island');
    if (!spotlightCards || spotlightCards.length === 0) return;

    spotlightCards.forEach(function (card) {
      if (card.__hasSpotlight) return;
      card.__hasSpotlight = true;

      card.addEventListener('mousemove', function (e) {
        var rect = card.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        card.style.setProperty('--mouse-x', x + 'px');
        card.style.setProperty('--mouse-y', y + 'px');
      }, { passive: true });

      card.addEventListener('mouseleave', function () {
        card.style.setProperty('--mouse-x', '-999px');
        card.style.setProperty('--mouse-y', '-999px');
      }, { passive: true });
    });
  }

  function initProjectCardFlip() {
    var cards = document.querySelectorAll('.island-project-card.flip-card');
    if (!cards || cards.length === 0) return;

    cards.forEach(function (card) {
      if (card.__hasFlipListener) return;
      card.__hasFlipListener = true;

      card.addEventListener('click', function (e) {
        // 若点击的是外部跳转链接（且非翻转触发器），保持原链接正常打开，不触发卡片翻转
        if (e.target.closest('a') && !e.target.closest('.flip-trigger')) {
          return;
        }
        card.classList.toggle('is-flipped');
      });

      card.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          if (e.target.tagName !== 'A') {
            e.preventDefault();
            card.classList.toggle('is-flipped');
          }
        }
      });
    });
  }

  function initSideTocScrollSpy() {
    var sideToc = document.getElementById('about-side-toc');
    if (!sideToc || sideToc.__hasSpy) return;
    sideToc.__hasSpy = true;

    var items = Array.prototype.slice.call(sideToc.querySelectorAll('.side-toc-item'));
    if (!items.length) return;

    function getTargets() {
      return items.map(function (item) {
        var id = item.getAttribute('data-target');
        var el = document.getElementById(id);
        return { id: id, item: item, el: el };
      }).filter(function (t) { return !!t.el; });
    }

    var targets = getTargets();
    if (!targets.length) return;

    var isTicking = false;
    function updateActive() {
      var scrollPos = window.pageYOffset || document.documentElement.scrollTop || 0;
      var viewportHeight = window.innerHeight || 800;
      var docHeight = document.documentElement.scrollHeight || 1000;

      var activeTarget = targets[0];

      // 关键：当滚动到达或极度接近文档底部时，直接高亮最后一个板块（发展轨迹），解决因页面末尾高度不足无法激活的问题
      if (docHeight - (scrollPos + viewportHeight) <= 60) {
        activeTarget = targets[targets.length - 1];
      } else {
        var probeY = scrollPos + 180;
        for (var i = 0; i < targets.length; i++) {
          if (probeY >= targets[i].el.offsetTop) {
            activeTarget = targets[i];
          }
        }
      }

      items.forEach(function (it) { it.classList.remove('active'); });
      if (activeTarget && activeTarget.item) {
        activeTarget.item.classList.add('active');
      }
      isTicking = false;
    }

    window.addEventListener('scroll', function () {
      if (!isTicking) {
        isTicking = true;
        requestAnimationFrame(updateActive);
      }
    }, { passive: true });

    updateActive();

    items.forEach(function (item) {
      item.addEventListener('click', function (e) {
        var id = item.getAttribute('data-target');
        var el = document.getElementById(id);
        if (el) {
          e.preventDefault();
          var targetTop = el.getBoundingClientRect().top + (window.pageYOffset || document.documentElement.scrollTop);
          var maxScroll = document.documentElement.scrollHeight - window.innerHeight;
          var desiredScroll = Math.min(Math.max(0, targetTop - 90), maxScroll);
          window.scrollTo({
            top: desiredScroll,
            behavior: 'smooth'
          });
        }
      });
    });
  }

  function initAll() {
    initIslandScrollReveal();
    initSpotlightCards();
    initProjectCardFlip();
    initSideTocScrollSpy();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

  // Re-run on PJAX or custom transitions if any
  document.addEventListener('pjax:complete', initAll);
  window.addEventListener('hexo:page-loaded', initAll);
})();
