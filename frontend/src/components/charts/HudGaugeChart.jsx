import { useRef, useEffect, useMemo } from 'react';

const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#ff5252',
  medium: '#ffab00',
  low: '#10b981',
  info: 'var(--hud-cyan)',
};

export default function HudGaugeChart({
  value = 0,
  max = 100,
  min = 0,
  label = '',
  unit = '%',
  severity,
  color,
  size = 180,
  strokeWidth = 14,
  showTicks = true,
  animated = true,
  className = '',
}) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const currentVal = useRef(0);

  const gaugeColor = color || (severity ? SEVERITY_COLORS[severity] : 'var(--hud-cyan)');
  const percent = Math.max(0, Math.min(1, (value - min) / (max - min)));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * 0.7 * dpr;
    ctx.scale(dpr, dpr);

    const cx = size / 2;
    const cy = size * 0.6;
    const r = size / 2 - strokeWidth - 4;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    const targetVal = percent;

    function draw(p) {
      ctx.clearRect(0, 0, size, size * 0.7);

      // Background arc
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, endAngle);
      ctx.strokeStyle = 'rgba(56,189,248,0.08)';
      ctx.lineWidth = strokeWidth;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Ticks
      if (showTicks) {
        for (let i = 0; i <= 10; i++) {
          const angle = startAngle + (endAngle - startAngle) * (i / 10);
          const x1 = cx + Math.cos(angle) * (r + strokeWidth / 2 + 2);
          const y1 = cy + Math.sin(angle) * (r + strokeWidth / 2 + 2);
          const x2 = cx + Math.cos(angle) * (r + strokeWidth / 2 + 6);
          const y2 = cy + Math.sin(angle) * (r + strokeWidth / 2 + 6);
          ctx.beginPath();
          ctx.moveTo(x1, y1);
          ctx.lineTo(x2, y2);
          ctx.strokeStyle = 'rgba(56,189,248,0.15)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Value arc
      const valAngle = startAngle + (endAngle - startAngle) * p;
      ctx.beginPath();
      ctx.arc(cx, cy, r, startAngle, valAngle);
      ctx.strokeStyle = gaugeColor;
      ctx.lineWidth = strokeWidth;
      ctx.lineCap = 'round';
      ctx.shadowColor = gaugeColor;
      ctx.shadowBlur = 12;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Needle
      const needleAngle = startAngle + (endAngle - startAngle) * p;
      const nx = cx + Math.cos(needleAngle) * (r - 8);
      const ny = cy + Math.sin(needleAngle) * (r - 8);
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fillStyle = gaugeColor;
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.strokeStyle = gaugeColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Value text
      const displayVal = Math.round(min + (max - min) * p);
      ctx.font = `bold ${size * 0.14}px "JetBrains Mono", monospace`;
      ctx.fillStyle = gaugeColor;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${displayVal}${unit}`, cx, cy - r * 0.2);

      // Label
      if (label) {
        ctx.font = `${size * 0.065}px "JetBrains Mono", monospace`;
        ctx.fillStyle = '#4a5568';
        ctx.fillText(label.toUpperCase(), cx, cy - r * 0.2 + size * 0.12);
      }

      // Min/Max
      ctx.font = `${size * 0.055}px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#2d3748';
      ctx.textAlign = 'left';
      ctx.fillText(`${min}`, cx - r - 2, cy + 14);
      ctx.textAlign = 'right';
      ctx.fillText(`${max}`, cx + r + 2, cy + 14);
    }

    if (animated) {
      const start = currentVal.current;
      const dur = 800;
      let t0;
      function step(ts) {
        if (!t0) t0 = ts;
        const prog = Math.min(1, (ts - t0) / dur);
        const ease = 1 - Math.pow(1 - prog, 3);
        const p = start + (targetVal - start) * ease;
        draw(p);
        if (prog < 1) {
          animRef.current = requestAnimationFrame(step);
        } else {
          currentVal.current = targetVal;
        }
      }
      animRef.current = requestAnimationFrame(step);
    } else {
      draw(targetVal);
      currentVal.current = targetVal;
    }

    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [value, max, min, percent, gaugeColor, size, strokeWidth, showTicks, label, unit, animated]);

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <canvas
        ref={canvasRef}
        style={{ width: size, height: size * 0.7 }}
      />
    </div>
  );
}
