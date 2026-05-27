import { useRef, useEffect, useState } from 'react';
import { Cpu } from 'lucide-react';

export default function CpuMonitor({ cores = 8, className = '' }) {
  const [coreLoads, setCoreLoads] = useState(() => Array(cores).fill(0).map(() => Math.random() * 60 + 10));

  useEffect(() => {
    const iv = setInterval(() => {
      setCoreLoads(prev => prev.map(v => {
        const delta = (Math.random() - 0.5) * 30;
        return Math.max(2, Math.min(100, v + delta));
      }));
    }, 2000);
    return () => clearInterval(iv);
  }, [cores]);

  const avg = coreLoads.reduce((a, b) => a + b, 0) / coreLoads.length;

  const getColor = (v) => {
    if (v > 90) return 'var(--hud-red)';
    if (v > 70) return 'var(--hud-amber)';
    if (v > 40) return 'var(--hud-cyan)';
    return 'var(--hud-emerald)';
  };

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <Cpu size={12} style={{ color: getColor(avg) }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>CPU MONITOR</span>
        </div>
        <span className="font-mono text-[11px] font-bold" style={{ color: getColor(avg) }}>
          {avg.toFixed(0)}%
        </span>
      </div>
      <div className="p-3 grid gap-1.5" style={{ gridTemplateColumns: `repeat(${Math.min(cores, 4)}, 1fr)` }}>
        {coreLoads.map((load, i) => (
          <div key={i} className="text-center">
            <div className="font-mono text-[7px] mb-1" style={{ color: 'var(--hud-text-dim)' }}>C{i}</div>
            <div className="h-10 rounded-sm overflow-hidden relative" style={{
              background: 'rgba(0,229,255,0.04)',
              border: '1px solid var(--hud-border-subtle)',
            }}>
              <div className="absolute bottom-0 w-full transition-all duration-700" style={{
                height: `${load}%`,
                background: `linear-gradient(to top, ${getColor(load)}, ${getColor(load)}88)`,
              }} />
            </div>
            <div className="font-mono text-[8px] mt-0.5" style={{ color: getColor(load) }}>
              {load.toFixed(0)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
