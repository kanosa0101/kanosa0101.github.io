/**
 * kanosa's blog - Fluid Particle Cursor Trail (Mechanism A)
 * Lightweight canvas-based pointer trail inspired by modern creative portfolios.
 * Automatically disabled on mobile/touch screens and when reduced motion is preferred.
 */
(function () {
  'use strict';

  // Only run on fine pointer devices (desktop mouse) and when motion is allowed
  if (typeof window === 'undefined') return;
  if (!window.matchMedia || !window.matchMedia('(pointer: fine)').matches) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  var canvas, ctx;
  var width = window.innerWidth;
  var height = window.innerHeight;
  var particles = [];
  var isRunning = false;
  var lastX = -100, lastY = -100;
  var colors = [
    'rgba(19, 194, 194, ',  // Teal / Cyan
    'rgba(78, 142, 247, ',  // Electric Blue
    'rgba(99, 102, 241, ',  // Indigo
    'rgba(247, 186, 78, '   // Soft Gold
  ];

  function initCanvas() {
    if (canvas) return;
    canvas = document.createElement('canvas');
    canvas.id = 'pointer-trail-canvas';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
    canvas.style.pointerEvents = 'none';
    canvas.style.zIndex = '99998';
    document.body.appendChild(canvas);

    ctx = canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas, { passive: true });
    window.addEventListener('pointermove', onPointerMove, { passive: true });
  }

  function resizeCanvas() {
    if (!canvas) return;
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;
  }

  function onPointerMove(e) {
    var x = e.clientX;
    var y = e.clientY;

    // Calculate distance moved
    var dx = x - lastX;
    var dy = y - lastY;
    var dist = Math.sqrt(dx * dx + dy * dy);

    if (dist > 4) {
      var count = Math.min(Math.floor(dist / 6), 3);
      for (var i = 0; i < count; i++) {
        var baseColor = colors[Math.floor(Math.random() * colors.length)];
        particles.push({
          x: x + (Math.random() - 0.5) * 8,
          y: y + (Math.random() - 0.5) * 8,
          vx: (Math.random() - 0.5) * 1.2,
          vy: (Math.random() - 0.5) * 1.2,
          size: Math.random() * 2.8 + 1.2,
          life: 1,
          decay: Math.random() * 0.035 + 0.025,
          color: baseColor
        });
      }
      lastX = x;
      lastY = y;

      if (!isRunning) {
        isRunning = true;
        requestAnimationFrame(renderLoop);
      }
    }
  }

  function renderLoop() {
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);

    for (var i = particles.length - 1; i >= 0; i--) {
      var p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= p.decay;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
      ctx.fillStyle = p.color + (p.life * 0.7) + ')';
      ctx.shadowColor = p.color + '0.5)';
      ctx.shadowBlur = 6;
      ctx.fill();
    }

    if (particles.length > 0) {
      requestAnimationFrame(renderLoop);
    } else {
      isRunning = false;
      ctx.clearRect(0, 0, width, height);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCanvas);
  } else {
    initCanvas();
  }
})();
