import { useRef, useEffect, useState, useMemo } from 'react';
import { Brain, TrendingUp } from 'lucide-react';

export default function AnomalyDetector({ data = [], threshold = 2.0, className = '' }) {
  const canvasRef = useRef(null);
  const [simData, setSimData] = useState(() => generateSimData());

  useEffect(() => {
    if (data.length > 0) return;
    const iv = setInterval(() => setSimData(generateSimData()), 4000);
    return () => clearInterval(iv);
  }, [data]);

  const points = data.length > 0 ? data : simData;
  const anomalies = useMemo(() => {
    const vals = points.map(p => p.value);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    return points.map(p => ({
      ...p,
      isAnomaly: Math.abs(p.value - mean) > threshold * std,
      zScore: std > 0 ? (p.value - mean) / std : 0,
    }));
  }, [points, threshold]);

  const anomalyCount = anomalies.filter(a => a.isAnomaly).length;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || anomalies.length === 0) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const maxVal = Math.max(...anomalies.map(a => a.value));
    const minVal = Math.min(...anomalies.map(a => a.value));
    const range = maxVal - minVal || 1;

    // Grid
    ctx.strokeStyle = 'rgba(0,229,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = (i / 3) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Threshold bands
    const vals = anomalies.map(a => a.value);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length);
    const upperY = h - ((mean + threshold * std - minVal) / range) * h * 0.85 - h * 0.05;
    const lowerY = h - ((mean - threshold * std - minVal) / range) * h * 0.85 - h * 0.05;
    ctx.fillStyle = 'rgba(255,0,60,0.04)';
    ctx.fillRect(0, 0, w, Math.max(0, upperY));
    ctx.fillRect(0, Math.min(h, lowerY), w, h - Math.min(h, lowerY));

    // Dashed threshold lines
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = 'rgba(255,0,60,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, upperY); ctx.lineTo(w, upperY); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, lowerY); ctx.lineTo(w, lowerY); ctx.stroke();
    ctx.setLineDash([]);

    // Line
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0,229,255,0.6)';
    ctx.lineWidth = 1.5;
    anomalies.forEach((a, i) => {
      const x = (i / (anomalies.length - 1)) * w;
      const y = h - ((a.value - minVal) / range) * h * 0.85 - h * 0.05;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Anomaly dots
    anomalies.forEach((a, i) => {
      if (!a.isAnomaly) return;
      const x = (i / (anomalies.length - 1)) * w;
      const y = h - ((a.value - minVal) / range) * h * 0.85 - h * 0.05;
      // Glow
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,0,60,0.15)';
      ctx.fill();
      // Dot
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ff003c';
      ctx.fill();
    });
  }, [anomalies, threshold]);

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <Brain size={12} style={{ color: anomalyCount > 0 ? 'var(--hud-red)' : 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>ANOMALY DETECT</span>
        </div>
        <span className="font-mono text-[9px] font-bold" style={{
          color: anomalyCount > 0 ? 'var(--hud-red)' : 'var(--hud-emerald)',
        }}>
          {anomalyCount} ANOMALI
        </span>
      </div>
      <canvas ref={canvasRef} className="w-full" style={{ height: 90, display: 'block' }} />
      <div className="flex justify-between px-3 py-1.5" style={{ borderTop: '1px solid var(--hud-border-subtle)' }}>
        <span className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>
          σ threshold: {threshold}
        </span>
        <span className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>
          {anomalies.length} samples
        </span>
      </div>
    </div>
  );
}

function generateSimData() {
  const arr = [];
  for (let i = 0; i < 60; i++) {
    let value = 50 + Math.sin(i * 0.15) * 15 + (Math.random() - 0.5) * 10;
    // Inject anomalies
    if (i === 12 || i === 38 || i === 51) value += 45 + Math.random() * 20;
    if (i === 25) value -= 40;
    arr.push({ ts: i, value });
  }
  return arr;
}
