import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, Legend
} from 'recharts';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444'];

function HudTooltip({ active, payload }) {
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
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color, margin: 0 }}>
          {p.name}: <strong>{p.value}</strong>
        </p>
      ))}
    </div>
  );
}

export default function HudRadarChart({
  data = [],
  dataKeys = ['value'],
  angleKey = 'name',
  colors = HUD_COLORS,
  height = 350,
  showLegend = false,
  animated = true,
  fillOpacity = 0.15,
  domain = [0, 100],
  title,
  className = '',
}) {
  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <RadarChart data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid
            stroke="rgba(56,189,248,0.1)"
            gridType="polygon"
          />
          <PolarAngleAxis
            dataKey={angleKey}
            tick={{ fill: '#4a5568', fontSize: 10, fontFamily: 'var(--font-mono)' }}
            stroke="rgba(56,189,248,0.15)"
          />
          <PolarRadiusAxis
            angle={90}
            domain={domain}
            tick={{ fill: '#2d3748', fontSize: 9, fontFamily: 'var(--font-mono)' }}
            stroke="rgba(56,189,248,0.08)"
          />
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11 }} />
          )}
          {dataKeys.map((key, i) => (
            <Radar
              key={key}
              name={key}
              dataKey={key}
              stroke={colors[i % colors.length]}
              fill={colors[i % colors.length]}
              fillOpacity={fillOpacity}
              strokeWidth={2}
              isAnimationActive={animated}
              animationDuration={1000}
              dot={{ r: 3, fill: colors[i % colors.length], stroke: '#060a14', strokeWidth: 1 }}
            />
          ))}
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
