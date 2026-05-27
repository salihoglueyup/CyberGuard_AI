import { useRef, useEffect, useMemo } from 'react';
import { Crosshair } from 'lucide-react';

const ATTACK_CATEGORIES = [
  { name: 'Network', children: [
    { name: 'DDoS', count: 234, severity: 'critical' },
    { name: 'Port Scan', count: 891, severity: 'medium' },
    { name: 'DNS Tunnel', count: 67, severity: 'high' },
  ]},
  { name: 'Web', children: [
    { name: 'SQLi', count: 156, severity: 'critical' },
    { name: 'XSS', count: 342, severity: 'high' },
    { name: 'CSRF', count: 45, severity: 'medium' },
    { name: 'Path Traversal', count: 78, severity: 'high' },
  ]},
  { name: 'Auth', children: [
    { name: 'Brute Force', count: 1203, severity: 'high' },
    { name: 'Credential Stuff', count: 456, severity: 'critical' },
    { name: 'Session Hijack', count: 23, severity: 'critical' },
  ]},
  { name: 'Malware', children: [
    { name: 'Ransomware', count: 12, severity: 'critical' },
    { name: 'Trojan', count: 34, severity: 'high' },
    { name: 'Cryptominer', count: 89, severity: 'medium' },
  ]},
];

const SEV_COLORS = { critical: '#ff003c', high: '#ff6a00', medium: '#ffc400', low: '#00e676' };

export default function AttackVector3D({ categories = ATTACK_CATEGORIES, className = '' }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const centerX = w / 2;
    const centerY = h / 2;
    const maxRadius = Math.min(w, h) * 0.42;

    // Background circles
    for (let r = 1; r <= 3; r++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (r / 3) * maxRadius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(0,229,255,0.06)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }

    // Center dot
    ctx.beginPath();
    ctx.arc(centerX, centerY, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,229,255,0.4)';
    ctx.fill();
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillStyle = 'rgba(0,229,255,0.5)';
    ctx.textAlign = 'center';
    ctx.fillText('TARGET', centerX, centerY + 14);

    const totalCount = categories.flatMap(c => c.children).reduce((a, c) => a + c.count, 0);
    let angleOffset = -Math.PI / 2;
    const categoryAngle = (Math.PI * 2) / categories.length;

    categories.forEach((cat, ci) => {
      const catAngle = angleOffset + ci * categoryAngle;
      const catMid = catAngle + categoryAngle / 2;

      // Category label
      const labelR = maxRadius + 16;
      const lx = centerX + Math.cos(catMid) * labelR;
      const ly = centerY + Math.sin(catMid) * labelR;
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.fillStyle = 'rgba(200,214,229,0.8)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(cat.name.toUpperCase(), lx, ly);

      // Sector line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.cos(catAngle) * maxRadius, centerY + Math.sin(catAngle) * maxRadius);
      ctx.strokeStyle = 'rgba(0,229,255,0.08)';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Child nodes
      const childAngleStep = categoryAngle / (cat.children.length + 1);
      cat.children.forEach((child, j) => {
        const childAngle = catAngle + (j + 1) * childAngleStep;
        const dist = (child.count / totalCount) * maxRadius * 2;
        const r = Math.min(maxRadius * 0.9, Math.max(maxRadius * 0.2, dist));
        const cx = centerX + Math.cos(childAngle) * r;
        const cy = centerY + Math.sin(childAngle) * r;
        const dotSize = Math.max(3, Math.min(10, child.count / 80));
        const color = SEV_COLORS[child.severity] || '#00e5ff';

        // Connection line
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(cx, cy);
        ctx.strokeStyle = `${color}30`;
        ctx.lineWidth = 1;
        ctx.stroke();

        // Glow
        ctx.beginPath();
        ctx.arc(cx, cy, dotSize + 4, 0, Math.PI * 2);
        ctx.fillStyle = `${color}15`;
        ctx.fill();

        // Node
        ctx.beginPath();
        ctx.arc(cx, cy, dotSize, 0, Math.PI * 2);
        ctx.fillStyle = `${color}cc`;
        ctx.fill();

        // Label
        ctx.font = '7px JetBrains Mono, monospace';
        ctx.fillStyle = `${color}aa`;
        ctx.textAlign = 'center';
        ctx.fillText(child.name, cx, cy - dotSize - 4);
        ctx.fillStyle = 'rgba(150,170,190,0.5)';
        ctx.fillText(child.count.toString(), cx, cy + dotSize + 8);
      });
    });
  }, [categories]);

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <Crosshair size={12} style={{ color: 'var(--hud-red)' }} />
        <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
          ATTACK VECTORS
        </span>
      </div>
      <canvas ref={canvasRef} className="w-full" style={{ height: 220, display: 'block' }} />
    </div>
  );
}
