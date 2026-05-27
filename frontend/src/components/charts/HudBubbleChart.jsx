import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444'];

function HudTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  return (
    <div style={{
      background: 'rgba(6,10,20,0.95)',
      border: '1px solid rgba(56,189,248,0.2)',
      borderRadius: 4,
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
    }}>
      {d?.name && <p style={{ color: 'var(--hud-cyan)', marginBottom: 4, fontWeight: 'bold' }}>{d.name}</p>}
      <p style={{ color: '#c8d6e5', margin: 0 }}>X: {d?.x} / Y: {d?.y}</p>
      {d?.z != null && <p style={{ color: '#4a5568', margin: 0 }}>Size: {d.z}</p>}
    </div>
  );
}

export default function HudBubbleChart({
  data = [],
  groups = [],
  xKey = 'x',
  yKey = 'y',
  zKey = 'z',
  colors = HUD_COLORS,
  height = 350,
  zRange = [30, 400],
  showGrid = true,
  showLegend = false,
  animated = true,
  title,
  className = '',
}) {
  // If no groups, treat data as single group
  const datasets = groups.length > 0 ? groups : [{ name: 'data', data }];

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <ScatterChart margin={{ top: 8, right: 8, left: -10, bottom: 0 }}>
          {showGrid && (
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(56,189,248,0.06)"
            />
          )}
          <XAxis
            dataKey={xKey}
            type="number"
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          />
          <YAxis
            dataKey={yKey}
            type="number"
            stroke="rgba(56,189,248,0.15)"
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
          />
          <ZAxis dataKey={zKey} type="number" range={zRange} />
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
          )}
          {datasets.map((group, i) => (
            <Scatter
              key={group.name || i}
              name={group.name}
              data={group.data || data}
              fill={colors[i % colors.length]}
              fillOpacity={0.5}
              stroke={colors[i % colors.length]}
              strokeWidth={1}
              isAnimationActive={animated}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
