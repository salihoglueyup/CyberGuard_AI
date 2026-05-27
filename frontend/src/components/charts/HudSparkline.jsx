import { useRef, useEffect } from 'react';

export default function HudSparkline({
  data = [],
  width = 120,
  height = 32,
  color = 'var(--hud-cyan)',
  fillOpacity = 0.15,
  strokeWidth = 1.5,
  showDot = true,
  animated = true,
  className = '',
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const values = data.map(d => (typeof d === 'number' ? d : d.value ?? 0));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const pad = 4;

    function getPoint(i) {
      const x = pad + (i / (values.length - 1)) * (width - pad * 2);
      const y = height - pad - ((values[i] - min) / range) * (height - pad * 2);
      return [x, y];
    }

    function draw(progress) {
      ctx.clearRect(0, 0, width, height);
      const count = Math.floor(values.length * progress);
      if (count < 2) return;

      // Fill gradient
      const grad = ctx.createLinearGradient(0, 0, 0, height);
      grad.addColorStop(0, color + Math.round(fillOpacity * 255).toString(16).padStart(2, '0'));
      grad.addColorStop(1, color + '00');

      ctx.beginPath();
      const [x0, y0] = getPoint(0);
      ctx.moveTo(x0, y0);
      for (let i = 1; i < count; i++) {
        const [x, y] = getPoint(i);
        ctx.lineTo(x, y);
      }
      // Close for fill
      const [lastX] = getPoint(count - 1);
      ctx.lineTo(lastX, height);
      ctx.lineTo(x0, height);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();

      // Line
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      for (let i = 1; i < count; i++) {
        const [x, y] = getPoint(i);
        ctx.lineTo(x, y);
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = strokeWidth;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // End dot
      if (showDot && count === values.length) {
        const [dx, dy] = getPoint(values.length - 1);
        ctx.beginPath();
        ctx.arc(dx, dy, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 6;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    if (animated) {
      let frame = 0;
      const totalFrames = 30;
      function step() {
        frame++;
        draw(frame / totalFrames);
        if (frame < totalFrames) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    } else {
      draw(1);
    }
  }, [data, width, height, color, fillOpacity, strokeWidth, showDot, animated]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width, height, display: 'block' }}
    />
  );
}
