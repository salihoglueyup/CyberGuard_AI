import { useMemo } from 'react';
import { Globe } from 'lucide-react';

const DEFAULT_REGIONS = [
  { name: 'Avrupa', code: 'EU', attacks: 342, color: '#00e5ff' },
  { name: 'Asya', code: 'AS', attacks: 891, color: '#ff6a00' },
  { name: 'K.Amerika', code: 'NA', attacks: 234, color: '#ffc400' },
  { name: 'G.Amerika', code: 'SA', attacks: 67, color: '#00e676' },
  { name: 'Afrika', code: 'AF', attacks: 45, color: '#b388ff' },
  { name: 'Okyanusya', code: 'OC', attacks: 23, color: '#448aff' },
];

export default function GeoDistribution({ regions = DEFAULT_REGIONS, className = '' }) {
  const total = useMemo(() => regions.reduce((a, r) => a + r.attacks, 0), [regions]);
  const sorted = useMemo(() => [...regions].sort((a, b) => b.attacks - a.attacks), [regions]);
  const max = sorted[0]?.attacks || 1;

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <Globe size={12} style={{ color: 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>GEO DISTRIBUTION</span>
        </div>
        <span className="font-mono text-[9px]" style={{ color: 'var(--hud-text)' }}>{total} total</span>
      </div>
      <div className="p-3 space-y-2">
        {sorted.map(r => {
          const pct = (r.attacks / total) * 100;
          const barWidth = (r.attacks / max) * 100;
          return (
            <div key={r.code}>
              <div className="flex items-center justify-between mb-0.5">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-sm" style={{ background: r.color }} />
                  <span className="font-mono text-[9px]" style={{ color: 'var(--hud-text)' }}>{r.name}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[9px] font-bold" style={{ color: r.color }}>{r.attacks}</span>
                  <span className="font-mono text-[8px]" style={{ color: 'var(--hud-text-dim)' }}>{pct.toFixed(0)}%</span>
                </div>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'rgba(0,229,255,0.04)' }}>
                <div className="h-full rounded-full transition-all duration-700" style={{
                  width: `${barWidth}%`,
                  background: `linear-gradient(90deg, ${r.color}66, ${r.color})`,
                }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
