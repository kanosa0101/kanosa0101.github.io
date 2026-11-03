/**
 * kanosa's blog - Island Architecture Interactive Effects
 * - Mechanism B: Independent Scroll Reveal (IntersectionObserver)
 * - Mechanism C: Radial Spotlight Hover Border & Surface Tracking
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

  function initAll() {
    initIslandScrollReveal();
    initSpotlightCards();
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
