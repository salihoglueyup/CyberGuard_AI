import { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444', '#448aff'];

function HudTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(6,10,20,0.95)',
      border: '1px solid rgba(56,189,248,0.2)',
      borderRadius: 4,
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
    }}>
      <p style={{ color: '#c8d6e5', marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, margin: 0 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</strong>
        </p>
      ))}
    </div>
  );
}

export default function HudLineChart({
  data = [],
  dataKeys = ['value'],
  xKey = 'name',
  colors = HUD_COLORS,
  height = 300,
  showGrid = true,
  showLegend = false,
  showDots = false,
  curved = true,
  strokeWidth = 2,
  animated = true,
  dashed = [],
  title,
  className = '',
}) {
  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
          {showGrid && (
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)" vertical={false} />
          )}
          <XAxis
            dataKey={xKey}
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          />
          <YAxis
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          />
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
          )}
          {dataKeys.map((key, i) => (
            <Line
              key={key}
              type={curved ? 'monotone' : 'linear'}
              dataKey={key}
              stroke={colors[i % colors.length]}
              strokeWidth={strokeWidth}
              strokeDasharray={dashed.includes(key) ? '6 3' : undefined}
              dot={showDots ? { r: 3, fill: '#060a14', stroke: colors[i % colors.length], strokeWidth: 2 } : false}
              activeDot={{ r: 5, fill: colors[i % colors.length], stroke: '#060a14', strokeWidth: 2 }}
              isAnimationActive={animated}
              animationDuration={1200}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
