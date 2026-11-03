/**
 * kanosa's blog - Island Architecture Interactive Effects
 * - Mechanism B: Independent Scroll Reveal (IntersectionObserver)
 * - Mechanism C: Radial Spotlight Hover Border & Surface Tracking
 * - About Subnav: Sticky Pill Dock Scroll-Spy & Smooth Jump
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

  function initSubnavScrollSpy() {
    var subnav = document.getElementById('about-subnav');
    if (!subnav || subnav.__hasSpy) return;
    subnav.__hasSpy = true;

    var items = Array.prototype.slice.call(subnav.querySelectorAll('.subnav-item'));
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
      var scrollPos = window.scrollY + 170;
      var activeTarget = targets[0];

      for (var i = 0; i < targets.length; i++) {
        if (scrollPos >= targets[i].el.offsetTop) {
          activeTarget = targets[i];
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
          var offsetTop = el.getBoundingClientRect().top + window.pageYOffset - 125;
          window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
          });
        }
      });
    });
  }

  function initAll() {
    initIslandScrollReveal();
    initSpotlightCards();
    initSubnavScrollSpy();
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
