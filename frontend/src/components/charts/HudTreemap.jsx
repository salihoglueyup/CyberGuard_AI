import {
  Treemap, ResponsiveContainer, Tooltip
} from 'recharts';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444', '#448aff', '#80deea', '#69f0ae'];

function CustomContent({ x, y, width, height, name, value, depth, index, colors }) {
  if (width < 4 || height < 4) return null;
  const color = colors[index % colors.length];
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={3}
        fill={color}
        fillOpacity={0.2}
        stroke={color}
        strokeWidth={1}
        strokeOpacity={0.5}
      />
      {width > 50 && height > 30 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 6}
            textAnchor="middle"
            fill={color}
            fontSize={Math.min(11, width / 8)}
            fontFamily="var(--font-mono)"
            fontWeight="bold"
          >
            {name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 10}
            textAnchor="middle"
            fill="#4a5568"
            fontSize={Math.min(10, width / 9)}
            fontFamily="var(--font-mono)"
          >
            {value?.toLocaleString()}
          </text>
        </>
      )}
    </g>
  );
}

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
      <p style={{ color: 'var(--hud-cyan)', margin: 0 }}>
        {d?.name}: <strong>{d?.value?.toLocaleString()}</strong>
      </p>
    </div>
  );
}

export default function HudTreemap({
  data = [],
  dataKey = 'value',
  nameKey = 'name',
  colors = HUD_COLORS,
  height = 300,
  animated = true,
  title,
  className = '',
}) {
  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <ResponsiveContainer width="100%" height={height}>
        <Treemap
          data={data}
          dataKey={dataKey}
          nameKey={nameKey}
          stroke="rgba(6,10,20,0.9)"
          isAnimationActive={animated}
          animationDuration={800}
          content={<CustomContent colors={colors} />}
        >
          <Tooltip content={<HudTooltip />} />
        </Treemap>
      </ResponsiveContainer>
    </div>
  );
}
