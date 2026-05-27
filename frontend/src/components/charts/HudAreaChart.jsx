import { useMemo } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
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

export default function HudAreaChart({
  data = [],
  dataKeys = ['value'],
  xKey = 'name',
  colors = HUD_COLORS,
  height = 300,
  showGrid = true,
  showLegend = false,
  stacked = false,
  gradientOpacity = [0.4, 0.02],
  strokeWidth = 2,
  animated = true,
  title,
  className = '',
}) {
  const gradients = useMemo(() => dataKeys.map((key, i) => ({
    id: `grad-area-${key}`,
    color: colors[i % colors.length],
  })), [dataKeys, colors]);

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && (
        <div className="hud-chart-title">{title}</div>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
          <defs>
            {gradients.map(g => (
              <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={g.color} stopOpacity={gradientOpacity[0]} />
                <stop offset="100%" stopColor={g.color} stopOpacity={gradientOpacity[1]} />
              </linearGradient>
            ))}
          </defs>
          {showGrid && (
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(56,189,248,0.06)"
              vertical={false}
            />
          )}
          <XAxis
            dataKey={xKey}
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: 'rgba(56,189,248,0.15)' }}
          />
          <YAxis
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
            axisLine={{ stroke: 'rgba(56,189,248,0.15)' }}
          />
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend
              wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#c8d6e5' }}
            />
          )}
          {dataKeys.map((key, i) => (
            <Area
              key={key}
              type="monotone"
              dataKey={key}
              stackId={stacked ? 'stack' : undefined}
              stroke={colors[i % colors.length]}
              strokeWidth={strokeWidth}
              fill={`url(#grad-area-${key})`}
              isAnimationActive={animated}
              animationDuration={1200}
              dot={false}
              activeDot={{ r: 4, stroke: colors[i % colors.length], strokeWidth: 2, fill: '#060a14' }}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
