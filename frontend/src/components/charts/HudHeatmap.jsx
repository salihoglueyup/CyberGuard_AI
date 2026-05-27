import { useMemo } from 'react';

const HUD_COLORS = ['#38bdf8', '#10b981', '#ffab00', '#b388ff', '#ef4444', '#448aff', '#80deea', '#69f0ae'];

export default function HudHeatmap({
  data = [],
  xLabels = [],
  yLabels = [],
  colorRange = ['#060a14', '#38bdf8'],
  cellSize = 28,
  gap = 2,
  showValues = true,
  maxValue,
  title,
  className = '',
}) {
  const { max, cells } = useMemo(() => {
    let mx = maxValue ?? 0;
    if (!maxValue) {
      data.forEach(row => row.forEach(v => { if (v > mx) mx = v; }));
    }
    const cells = data.map((row, y) =>
      row.map((v, x) => ({
        x, y, value: v,
        opacity: mx > 0 ? v / mx : 0,
      }))
    );
    return { max: mx, cells };
  }, [data, maxValue]);

  const width = xLabels.length * (cellSize + gap) + 50;
  const height = yLabels.length * (cellSize + gap) + 30;

  function interpolateColor(t) {
    const r0 = parseInt(colorRange[0].slice(1, 3), 16);
    const g0 = parseInt(colorRange[0].slice(3, 5), 16);
    const b0 = parseInt(colorRange[0].slice(5, 7), 16);
    const r1 = parseInt(colorRange[1].slice(1, 3), 16);
    const g1 = parseInt(colorRange[1].slice(3, 5), 16);
    const b1 = parseInt(colorRange[1].slice(5, 7), 16);
    const r = Math.round(r0 + (r1 - r0) * t);
    const g = Math.round(g0 + (g1 - g0) * t);
    const b = Math.round(b0 + (b1 - b0) * t);
    return `rgb(${r},${g},${b})`;
  }

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <div style={{ overflowX: 'auto' }}>
        <svg width={width} height={height} style={{ fontFamily: 'var(--font-mono)' }}>
          {/* Y labels */}
          {yLabels.map((label, i) => (
            <text
              key={`y-${i}`}
              x={46}
              y={28 + i * (cellSize + gap) + cellSize / 2 + 4}
              textAnchor="end"
              fill="#4a5568"
              fontSize={9}
            >
              {label}
            </text>
          ))}
          {/* X labels */}
          {xLabels.map((label, i) => (
            <text
              key={`x-${i}`}
              x={50 + i * (cellSize + gap) + cellSize / 2}
              y={18}
              textAnchor="middle"
              fill="#4a5568"
              fontSize={9}
            >
              {label}
            </text>
          ))}
          {/* Cells */}
          {cells.flat().map((cell, i) => (
            <g key={i}>
              <rect
                x={50 + cell.x * (cellSize + gap)}
                y={28 + cell.y * (cellSize + gap)}
                width={cellSize}
                height={cellSize}
                rx={3}
                fill={interpolateColor(cell.opacity)}
                stroke="rgba(56,189,248,0.06)"
                strokeWidth={0.5}
              >
                <title>{`${yLabels[cell.y]} / ${xLabels[cell.x]}: ${cell.value}`}</title>
              </rect>
              {showValues && cellSize >= 24 && (
                <text
                  x={50 + cell.x * (cellSize + gap) + cellSize / 2}
                  y={28 + cell.y * (cellSize + gap) + cellSize / 2 + 3}
                  textAnchor="middle"
                  fill={cell.opacity > 0.5 ? '#060a14' : '#4a5568'}
                  fontSize={8}
                  fontWeight={cell.opacity > 0.5 ? 'bold' : 'normal'}
                >
                  {cell.value}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>
    </div>
  );
}
