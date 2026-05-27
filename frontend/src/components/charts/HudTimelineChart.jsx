import { useMemo } from 'react';

const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#ff5252',
  medium: '#ffab00',
  low: '#10b981',
  info: 'var(--hud-cyan)',
};

export default function HudTimelineChart({
  events = [],
  height = 200,
  showConnectors = true,
  title,
  className = '',
}) {
  const sorted = useMemo(() =>
    [...events].sort((a, b) => new Date(a.time) - new Date(b.time)),
    [events]
  );

  if (!sorted.length) return null;

  const minT = new Date(sorted[0].time).getTime();
  const maxT = new Date(sorted[sorted.length - 1].time).getTime();
  const range = maxT - minT || 1;

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <div style={{ position: 'relative', height, overflowX: 'auto', overflowY: 'hidden' }}>
        {/* Baseline */}
        <div style={{
          position: 'absolute',
          top: height / 2,
          left: 20,
          right: 20,
          height: 1,
          background: 'rgba(56,189,248,0.15)',
        }} />

        {sorted.map((evt, i) => {
          const t = new Date(evt.time).getTime();
          const x = 20 + ((t - minT) / range) * (100 - 4); // percent
          const color = SEVERITY_COLORS[evt.severity] || 'var(--hud-cyan)';
          const isTop = i % 2 === 0;

          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${x}%`,
                top: isTop ? 8 : height / 2 + 8,
                transform: 'translateX(-50%)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                maxWidth: 120,
              }}
            >
              {/* Connector line */}
              {showConnectors && (
                <div style={{
                  width: 1,
                  height: isTop ? height / 2 - 20 : height / 2 - 20,
                  background: `${color}40`,
                  order: isTop ? 2 : 0,
                }} />
              )}
              {/* Node dot */}
              <div style={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: color,
                boxShadow: `0 0 8px ${color}60`,
                flexShrink: 0,
                order: isTop ? 3 : -1,
                position: 'relative',
                top: isTop ? 0 : -4,
              }} />
              {/* Label */}
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 9,
                color: '#c8d6e5',
                textAlign: 'center',
                lineHeight: 1.3,
                order: isTop ? 1 : 1,
                padding: '2px 4px',
              }}>
                <div style={{ color, fontWeight: 'bold', fontSize: 10 }}>{evt.label}</div>
                <div style={{ color: '#4a5568' }}>{evt.time}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
