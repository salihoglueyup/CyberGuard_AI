import { useMemo } from 'react';
import { Zap, TrendingUp, TrendingDown, Minus, Clock } from 'lucide-react';

const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#ff6a00',
  medium: '#ffc400',
  low: '#10b981',
  info: 'var(--hud-cyan)',
};

export default function GlobeStats({ attacks = [], className = '' }) {
  const stats = useMemo(() => {
    const total = attacks.length;
    const byServ = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
    const countries = new Set();
    let mlThreats = 0;

    attacks.forEach(a => {
      if (byServ[a.severity] !== undefined) byServ[a.severity]++;
      if (a.source?.country) countries.add(a.source.country);
      if (a.ml_prediction?.is_threat) mlThreats++;
    });

    // Recent rate (last minute)
    const now = Date.now();
    const recentCount = attacks.filter(a => {
      const t = a.timestamp ? new Date(a.timestamp).getTime() : now;
      return now - t < 60000;
    }).length;

    return { total, byServ, countries: countries.size, mlThreats, recentRate: recentCount };
  }, [attacks]);

  return (
    <div className={`absolute top-3 left-3 z-10 flex flex-col gap-1.5 ${className}`}>
      {/* Main count */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg" style={{
        background: 'var(--hud-surface-elevated)',
        border: '1px solid var(--hud-border)',
        boxShadow: 'var(--hud-shadow)',
      }}>
        <Zap size={14} style={{ color: 'var(--hud-cyan)' }} />
        <div>
          <div className="font-mono text-[8px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>CANLI SALDIRI</div>
          <div className="font-mono text-lg font-bold" style={{ color: 'var(--hud-cyan)' }}>{stats.total}</div>
        </div>
      </div>

      {/* Severity mini bars */}
      <div className="px-3 py-2 rounded-lg" style={{
        background: 'var(--hud-surface-elevated)',
        border: '1px solid var(--hud-border)',
        boxShadow: 'var(--hud-shadow)',
      }}>
        {Object.entries(stats.byServ).map(([sev, count]) => (
          <div key={sev} className="flex items-center gap-2 py-0.5">
            <div className="w-1.5 h-1.5 rounded-full" style={{ background: SEVERITY_COLORS[sev] }} />
            <span className="font-mono text-[9px] flex-1" style={{ color: 'var(--hud-text-muted)' }}>
              {sev.charAt(0).toUpperCase() + sev.slice(1)}
            </span>
            <span className="font-mono text-[10px] font-bold" style={{ color: SEVERITY_COLORS[sev] }}>
              {count}
            </span>
          </div>
        ))}
      </div>

      {/* Quick stats */}
      <div className="px-3 py-2 rounded-lg flex items-center gap-3" style={{
        background: 'var(--hud-surface-elevated)',
        border: '1px solid var(--hud-border)',
        boxShadow: 'var(--hud-shadow)',
      }}>
        <div className="text-center">
          <div className="font-mono text-[8px] tracking-wider" style={{ color: 'var(--hud-text-muted)' }}>ULKE</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-amber)' }}>{stats.countries}</div>
        </div>
        <div className="w-px h-6" style={{ background: 'var(--hud-border)' }} />
        <div className="text-center">
          <div className="font-mono text-[8px] tracking-wider" style={{ color: 'var(--hud-text-muted)' }}>AI</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-purple)' }}>{stats.mlThreats}</div>
        </div>
        <div className="w-px h-6" style={{ background: 'var(--hud-border)' }} />
        <div className="text-center">
          <div className="font-mono text-[8px] tracking-wider" style={{ color: 'var(--hud-text-muted)' }}>/DK</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-emerald)' }}>{stats.recentRate}</div>
        </div>
      </div>
    </div>
  );
}
