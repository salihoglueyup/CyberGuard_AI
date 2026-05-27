import { useMemo } from 'react';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444'];

export default function HudWaterfall({
  data = [],
  height = 300,
  barWidth = 36,
  gap = 8,
  positiveColor = '#10b981',
  negativeColor = '#ef4444',
  totalColor = 'var(--hud-cyan)',
  title,
  className = '',
}) {
  const layout = useMemo(() => {
    let running = 0;
    return data.map((d, i) => {
      const isTotal = d.isTotal;
      const val = d.value;
      const base = isTotal ? 0 : running;
      const top = isTotal ? val : running + val;
      if (!isTotal) running += val;
      return {
        ...d,
        base: Math.min(base, top),
        top: Math.max(base, top),
        positive: val >= 0,
        isTotal,
      };
    });
  }, [data]);

  const maxVal = Math.max(...layout.map(d => d.top), 0);
  const minVal = Math.min(...layout.map(d => d.base), 0);
  const range = maxVal - minVal || 1;
  const chartW = layout.length * (barWidth + gap) + 40;
  const chartH = height - 40;
  const padTop = 20;

  function yScale(v) {
    return padTop + (1 - (v - minVal) / range) * chartH;
  }

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <div style={{ overflowX: 'auto' }}>
        <svg width={chartW} height={height} style={{ fontFamily: 'var(--font-mono)' }}>
          {/* Zero line */}
          <line
            x1={20}
            y1={yScale(0)}
            x2={chartW - 10}
            y2={yScale(0)}
            stroke="rgba(56,189,248,0.15)"
            strokeDasharray="4 2"
          />

          {layout.map((d, i) => {
            const x = 30 + i * (barWidth + gap);
            const y1 = yScale(d.top);
            const y2 = yScale(d.base);
            const h = y2 - y1;
            const color = d.isTotal ? totalColor : (d.positive ? positiveColor : negativeColor);

            return (
              <g key={i}>
                {/* Connector */}
                {i > 0 && !d.isTotal && (
                  <line
                    x1={x - gap}
                    y1={yScale(d.base + (d.positive ? 0 : d.value))}
                    x2={x}
                    y2={yScale(d.base + (d.positive ? 0 : d.value))}
                    stroke="rgba(56,189,248,0.12)"
                    strokeDasharray="2 2"
                  />
                )}
                <rect
                  x={x}
                  y={y1}
                  width={barWidth}
                  height={Math.max(h, 1)}
                  rx={3}
                  fill={color}
                  fillOpacity={0.3}
                  stroke={color}
                  strokeWidth={1}
                />
                {/* Value label */}
                <text
                  x={x + barWidth / 2}
                  y={y1 - 5}
                  textAnchor="middle"
                  fill={color}
                  fontSize={9}
                  fontWeight="bold"
                >
                  {d.positive && !d.isTotal ? '+' : ''}{d.value}
                </text>
                {/* Name label */}
                <text
                  x={x + barWidth / 2}
                  y={height - 4}
                  textAnchor="middle"
                  fill="#4a5568"
                  fontSize={8}
                >
                  {d.name}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
