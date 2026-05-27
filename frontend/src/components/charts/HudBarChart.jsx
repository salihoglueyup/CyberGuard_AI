import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell
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
        <p key={i} style={{ color: p.color || p.fill, margin: 0 }}>
          {p.name}: <strong>{typeof p.value === 'number' ? p.value.toLocaleString() : p.value}</strong>
        </p>
      ))}
    </div>
  );
}

export default function HudBarChart({
  data = [],
  dataKeys = ['value'],
  xKey = 'name',
  colors = HUD_COLORS,
  height = 300,
  showGrid = true,
  showLegend = false,
  stacked = false,
  horizontal = false,
  barSize = 20,
  radius = [4, 4, 0, 0],
  animated = true,
  colorByItem = false,
  title,
  className = '',
}) {
  const Chart = horizontal ? BarChart : BarChart;
  const layout = horizontal ? 'vertical' : 'horizontal';

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout={layout} margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
          {showGrid && (
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(56,189,248,0.06)"
              vertical={!horizontal}
              horizontal={horizontal || true}
            />
          )}
          {horizontal ? (
            <>
              <XAxis type="number" stroke="rgba(56,189,248,0.15)" tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
              <YAxis dataKey={xKey} type="category" stroke="rgba(56,189,248,0.15)" tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }} width={80} />
            </>
          ) : (
            <>
              <XAxis dataKey={xKey} stroke="rgba(56,189,248,0.15)" tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
              <YAxis stroke="rgba(56,189,248,0.15)" tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }} />
            </>
          )}
          <Tooltip content={<HudTooltip />} cursor={{ fill: 'rgba(56,189,248,0.04)' }} />
          {showLegend && (
            <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
          )}
          {dataKeys.map((key, i) => (
            <Bar
              key={key}
              dataKey={key}
              stackId={stacked ? 'stack' : undefined}
              fill={colors[i % colors.length]}
              barSize={barSize}
              radius={radius}
              isAnimationActive={animated}
              animationDuration={800}
            >
              {colorByItem && data.map((_, idx) => (
                <Cell key={idx} fill={colors[idx % colors.length]} />
              ))}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
