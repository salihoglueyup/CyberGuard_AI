import { useMemo } from 'react';

const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#ff6a00',
  medium: '#ffc400',
  low: '#10b981',
  info: 'var(--hud-cyan)',
};

export default function GlobeAttackFeed({ attacks = [], maxItems = 8, className = '' }) {
  const recent = useMemo(() => attacks.slice(0, maxItems), [attacks, maxItems]);

  return (
    <div className={`absolute bottom-3 right-3 z-10 w-64 ${className}`}>
      <div className="rounded-lg overflow-hidden" style={{
        background: 'var(--hud-surface-elevated)',
        border: '1px solid var(--hud-border)',
        boxShadow: 'var(--hud-shadow-lg)',
      }}>
        <div className="px-3 py-1.5 flex items-center justify-between border-b" style={{ borderColor: 'var(--hud-border)' }}>
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
            CANLI AKIS
          </span>
          <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: '#10b981' }} />
        </div>
        <div className="max-h-52 overflow-y-auto" style={{ scrollbarWidth: 'thin' }}>
          {recent.map((a, i) => (
            <div
              key={a.id || i}
              className="flex items-center gap-2 px-3 py-1.5 border-b transition-colors hover:bg-white/[0.02]"
              style={{
                borderColor: 'var(--hud-border-subtle)',
                animation: i === 0 ? 'fadeSlideIn 0.3s ease-out' : undefined,
              }}
            >
              <div className="w-1 h-6 rounded-full flex-shrink-0" style={{ background: SEVERITY_COLORS[a.severity] || 'var(--hud-cyan)' }} />
              <div className="flex-1 min-w-0">
                <div className="font-mono text-[9px] truncate" style={{ color: 'var(--hud-text)' }}>
                  {a.attack_type || a.type || 'Unknown'}
                </div>
                <div className="font-mono text-[8px]" style={{ color: 'var(--hud-text-dim)' }}>
                  {a.source?.country || '??'} → TR
                </div>
              </div>
              <div className="text-right flex-shrink-0">
                <div className="font-mono text-[8px]" style={{ color: SEVERITY_COLORS[a.severity] }}>
                  {a.severity?.charAt(0).toUpperCase()}
                </div>
                {a.ml_prediction?.confidence && (
                  <div className="font-mono text-[7px]" style={{ color: 'var(--hud-purple)' }}>
                    {(a.ml_prediction.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            </div>
          ))}
          {recent.length === 0 && (
            <div className="font-mono text-[10px] text-center py-6" style={{ color: 'var(--hud-text-dim)' }}>
              Bekleniyor...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
