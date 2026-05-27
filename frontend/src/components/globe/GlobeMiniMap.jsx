import { useRef, useEffect } from 'react';

export default function GlobeMiniMap({
  attacks = [],
  centerLat = 39,
  centerLng = 35,
  size = 140,
  className = '',
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * 0.5 * dpr;
    ctx.scale(dpr, dpr);
    const w = size;
    const h = size * 0.5;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = 'rgba(6,10,20,0.9)';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(56,189,248,0.06)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 6; i++) {
      ctx.beginPath();
      ctx.moveTo((i / 6) * w, 0);
      ctx.lineTo((i / 6) * w, h);
      ctx.stroke();
    }
    for (let i = 0; i <= 3; i++) {
      ctx.beginPath();
      ctx.moveTo(0, (i / 3) * h);
      ctx.lineTo(w, (i / 3) * h);
      ctx.stroke();
    }

    function toXY(lat, lng) {
      const x = ((lng + 180) / 360) * w;
      const y = ((90 - lat) / 180) * h;
      return [x, y];
    }

    // Attack dots
    attacks.forEach(a => {
      if (!a.source?.lat) return;
      const [x, y] = toXY(a.source.lat, a.source.lng);
      const colors = {
        critical: '#ef4444',
        high: '#ff6a00',
        medium: '#ffc400',
        low: '#10b981',
        info: 'var(--hud-cyan)',
      };
      ctx.beginPath();
      ctx.arc(x, y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = colors[a.severity] || '#38bdf8';
      ctx.globalAlpha = 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
    });

    // Center crosshair (current camera)
    const [cx, cy] = toXY(centerLat, centerLng);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.6;
    // Viewport box
    const vw = w * 0.25;
    const vh = h * 0.3;
    ctx.strokeRect(cx - vw / 2, cy - vh / 2, vw, vh);
    // Crosshair
    ctx.beginPath();
    ctx.moveTo(cx - 4, cy);
    ctx.lineTo(cx + 4, cy);
    ctx.moveTo(cx, cy - 4);
    ctx.lineTo(cx, cy + 4);
    ctx.stroke();
    ctx.globalAlpha = 1;

    // Border
    ctx.strokeStyle = 'rgba(56,189,248,0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);
  }, [attacks, centerLat, centerLng, size]);

  return (
    <div className={`absolute bottom-3 left-3 z-10 ${className}`}>
      <canvas
        ref={canvasRef}
        style={{
          width: size,
          height: size * 0.5,
          borderRadius: 6,
          border: '1px solid rgba(56,189,248,0.15)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
        }}
      />
      <div className="font-mono text-[8px] text-center mt-0.5" style={{ color: 'var(--hud-text-dim)', letterSpacing: '2px' }}>
        WORLD VIEW
      </div>
    </div>
  );
}
