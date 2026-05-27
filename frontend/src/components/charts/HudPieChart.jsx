import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444', '#448aff', '#80deea', '#69f0ae'];

function HudTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div style={{
      background: 'rgba(6,10,20,0.95)',
      border: '1px solid rgba(56,189,248,0.2)',
      borderRadius: 4,
      padding: '8px 12px',
      fontFamily: 'var(--font-mono)',
      fontSize: 11,
    }}>
      <p style={{ color: d.payload?.fill || '#c8d6e5', margin: 0 }}>
        {d.name}: <strong>{d.value?.toLocaleString()}</strong>
        {d.payload?.percent != null && ` (${(d.payload.percent * 100).toFixed(1)}%)`}
      </p>
    </div>
  );
}

function CenterLabel({ viewBox, label, value }) {
  const { cx, cy } = viewBox;
  return (
    <g>
      <text x={cx} y={cy - 8} textAnchor="middle" fill="#4a5568" fontFamily="var(--font-mono)" fontSize={10}>
        {label}
      </text>
      <text x={cx} y={cy + 14} textAnchor="middle" fill="var(--hud-cyan)" fontFamily="var(--font-mono)" fontSize={20} fontWeight="bold">
        {value}
      </text>
    </g>
  );
}

export default function HudPieChart({
  data = [],
  dataKey = 'value',
  nameKey = 'name',
  colors = HUD_COLORS,
  height = 300,
  donut = true,
  innerRadius = '55%',
  outerRadius = '80%',
  showLegend = true,
  animated = true,
  centerLabel,
  centerValue,
  paddingAngle = 2,
  title,
  className = '',
}) {
  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            dataKey={dataKey}
            nameKey={nameKey}
            cx="50%"
            cy="50%"
            innerRadius={donut ? innerRadius : 0}
            outerRadius={outerRadius}
            paddingAngle={paddingAngle}
            stroke="rgba(6,10,20,0.8)"
            strokeWidth={2}
            isAnimationActive={animated}
            animationDuration={1000}
            label={!donut ? ({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%` : false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={colors[i % colors.length]} style={{ filter: `drop-shadow(0 0 4px ${colors[i % colors.length]}40)` }} />
            ))}
            {donut && centerLabel && (
              <CenterLabel label={centerLabel} value={centerValue} />
            )}
          </Pie>
          <Tooltip content={<HudTooltip />} />
          {showLegend && (
            <Legend
              layout="vertical"
              align="right"
              verticalAlign="middle"
              wrapperStyle={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: '#c8d6e5', paddingLeft: 16 }}
              formatter={(value) => <span style={{ color: '#c8d6e5' }}>{value}</span>}
            />
          )}
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
