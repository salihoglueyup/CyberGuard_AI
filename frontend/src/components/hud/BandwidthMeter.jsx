import { useRef, useEffect, useState } from 'react';
import { ArrowUpDown } from 'lucide-react';

export default function BandwidthMeter({ inbound = 0, outbound = 0, maxBandwidth = 1000, className = '' }) {
  const canvasRef = useRef(null);
  const historyRef = useRef({ inArr: [], outArr: [] });
  const [currentIn, setCurrentIn] = useState(inbound);
  const [currentOut, setCurrentOut] = useState(outbound);

  // Simulate fluctuations if no real data
  useEffect(() => {
    if (inbound > 0) {
      setCurrentIn(inbound);
      setCurrentOut(outbound);
      return;
    }
    const iv = setInterval(() => {
      setCurrentIn(Math.random() * maxBandwidth * 0.7 + maxBandwidth * 0.1);
      setCurrentOut(Math.random() * maxBandwidth * 0.4 + maxBandwidth * 0.05);
    }, 1500);
    return () => clearInterval(iv);
  }, [inbound, outbound, maxBandwidth]);

  useEffect(() => {
    const h = historyRef.current;
    h.inArr.push(currentIn);
    h.outArr.push(currentOut);
    if (h.inArr.length > 60) h.inArr.shift();
    if (h.outArr.length > 60) h.outArr.shift();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const hh = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = hh * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, hh);

    // Grid
    ctx.strokeStyle = 'rgba(0,229,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < 4; i++) {
      const y = (i / 3) * hh;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    function drawLine(arr, color) {
      if (arr.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      arr.forEach((v, i) => {
        const x = (i / 59) * w;
        const y = hh - (v / maxBandwidth) * hh * 0.9;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Fill under
      ctx.lineTo(w, hh);
      ctx.lineTo(0, hh);
      ctx.closePath();
      ctx.fillStyle = color.replace('1)', '0.08)');
      ctx.fill();
    }

    drawLine(h.inArr, 'rgba(0,229,255,1)');
    drawLine(h.outArr, 'rgba(0,230,118,1)');
  }, [currentIn, currentOut, maxBandwidth]);

  const formatBW = (v) => v >= 1000 ? `${(v / 1000).toFixed(1)} Gbps` : `${v.toFixed(0)} Mbps`;

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <ArrowUpDown size={12} style={{ color: 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
            BANDWIDTH
          </span>
        </div>
        <div className="flex gap-3">
          <span className="font-mono text-[9px]" style={{ color: 'var(--hud-cyan)' }}>▼ {formatBW(currentIn)}</span>
          <span className="font-mono text-[9px]" style={{ color: 'var(--hud-emerald)' }}>▲ {formatBW(currentOut)}</span>
        </div>
      </div>
      <canvas ref={canvasRef} className="w-full" style={{ height: 80, display: 'block' }} />
    </div>
  );
}
