import { useState, useEffect } from 'react';
import { HardDrive } from 'lucide-react';

const DEFAULT_DISKS = [
  { label: '/ (SSD)', total: 512, used: 287, type: 'ssd' },
  { label: '/data (HDD)', total: 2048, used: 1423, type: 'hdd' },
  { label: '/logs (SSD)', total: 256, used: 198, type: 'ssd' },
];

export default function DiskUsage({ disks = DEFAULT_DISKS, className = '' }) {
  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
    }}>
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <HardDrive size={12} style={{ color: 'var(--hud-cyan)' }} />
        <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
          DISK USAGE
        </span>
      </div>
      <div className="p-3 space-y-3">
        {disks.map((d, i) => {
          const pct = (d.used / d.total) * 100;
          const color = pct > 90 ? 'var(--hud-red)' : pct > 70 ? 'var(--hud-amber)' : 'var(--hud-cyan)';
          return (
            <div key={i}>
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-[9px]" style={{ color: 'var(--hud-text)' }}>{d.label}</span>
                <span className="font-mono text-[8px]" style={{ color }}>
                  {formatSize(d.used)} / {formatSize(d.total)}
                </span>
              </div>
              <div className="h-2 rounded-full overflow-hidden" style={{ background: 'rgba(0,229,255,0.06)' }}>
                <div className="h-full rounded-full transition-all duration-700" style={{
                  width: `${pct}%`,
                  background: `linear-gradient(90deg, ${color}88, ${color})`,
                  boxShadow: `0 0 8px ${color}40`,
                }} />
              </div>
              <div className="flex justify-between mt-0.5">
                <span className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>{d.type?.toUpperCase()}</span>
                <span className="font-mono text-[8px] font-bold" style={{ color }}>{pct.toFixed(0)}%</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatSize(gb) {
  if (gb >= 1024) return `${(gb / 1024).toFixed(1)} TB`;
  return `${gb} GB`;
}
