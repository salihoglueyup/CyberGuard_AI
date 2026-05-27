import {
  RadialBarChart, RadialBar, ResponsiveContainer, Legend, Tooltip
} from 'recharts';

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
      <p style={{ color: d?.fill || 'var(--hud-cyan)', margin: 0 }}>
        {d?.name}: <strong>{d?.value}</strong>
      </p>
    </div>
  );
}

export default function HudRadialBar({
  data = [],
  height = 300,
  innerRadius = '30%',
  outerRadius = '90%',
  barSize = 12,
  showLegend = true,
  animated = true,
  startAngle = 180,
  endAngle = -180,
  background = true,
  title,
  className = '',
}) {
  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <RadialBarChart
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          data={data}
          startAngle={startAngle}
          endAngle={endAngle}
          barSize={barSize}
        >
          <RadialBar
            background={background ? { fill: 'rgba(56,189,248,0.04)' } : false}
            dataKey="value"
            isAnimationActive={animated}
            animationDuration={1000}
            cornerRadius={6}
          />
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 10, lineHeight: '20px' }}
              formatter={(value, entry) => (
                <span style={{ color: entry.color }}>{value}</span>
              )}
            />
          )}
        </RadialBarChart>
      </ResponsiveContainer>
    </div>
  );
}
